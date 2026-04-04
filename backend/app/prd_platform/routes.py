"""FastAPI routes: /api/v1/prd-platform — unauthenticated."""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from openai import AzureOpenAI
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.llm_routing import llm_provider_is_local
from app.prd_platform.clarifying import CAPABILITIES, get_capability
from app.prd_platform.database import (
    clear_chunks,
    delete_chat_thread,
    delete_project,
    get_chat_thread_full,
    get_project,
    insert_chunks,
    save_chat_thread,
    update_project_status,
    upsert_project,
)
from app.prd_platform.embeddings import embed_texts, prd_embedding_configuration_error
from app.prd_platform.faiss_store import build_index, save_index
from app.prd_platform.generation import generate
from app.prd_platform.ingest import extract_and_chunk_zip
from app.prd_platform.neo4j_graph import sync_project_graph_to_neo4j

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/prd-platform", tags=["PRD Platform"])


class GenerateRequest(BaseModel):
    capability: str
    answers: dict[str, Any] = Field(default_factory=dict)
    project_id: Optional[str] = None


CHATBOT_SYSTEM = (
    "You are OgesAssistant, a helpful assistant for software documentation, code navigation, and clear explanations. "
    "Be concise unless the user asks for detail."
)

CONVERSATION_SUMMARY_MAX_CHARS = 32_768
ROLLUP_EXCHANGES_MAX = 24

ROLLUP_SYSTEM = (
    "You maintain a running summary of a conversation. Merge the prior summary (if any) with the new user/assistant "
    "exchanges. Output ONLY the updated summary — no title, no preamble. Use short bullets or compact paragraphs. "
    "Preserve facts, decisions, entities, and open questions. Drop pleasantries; avoid repeating redundant detail."
)


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=500_000)


class ChatbotRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500_000)
    history: list[ChatHistoryItem] = Field(default_factory=list)
    conversation_summary: Optional[str] = Field(default=None, max_length=CONVERSATION_SUMMARY_MAX_CHARS)


class ChatbotResponse(BaseModel):
    answer: str
    model: str = "local_gguf"


class ConversationRollupExchange(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=500_000)


class ConversationRollupRequest(BaseModel):
    prior_summary: str = Field(default="", max_length=CONVERSATION_SUMMARY_MAX_CHARS)
    exchanges: list[ConversationRollupExchange] = Field(..., min_length=1, max_length=ROLLUP_EXCHANGES_MAX)
    max_output_tokens: Optional[int] = Field(default=None, ge=32, le=4096)


class ConversationRollupResponse(BaseModel):
    summary: str
    model: str = "local_gguf"


CHAT_THREAD_ID_MAX_LEN = 256


class ChatPersistMessageItem(BaseModel):
    id: int = Field(default=0, ge=0, le=2**53)
    role: Literal["user", "assistant", "error"]
    content: str = Field(..., max_length=500_000)
    meta: Optional[dict[str, Any]] = None


class ChatThreadSaveRequest(BaseModel):
    mode: Literal["chatbot", "hybrid_rag"]
    rag_document_id: Optional[str] = Field(default=None, max_length=CHAT_THREAD_ID_MAX_LEN)
    conversation_summary: str = Field(default="", max_length=CONVERSATION_SUMMARY_MAX_CHARS)
    rollup_message_index: int = Field(default=0, ge=0)
    rag_indexed: bool = False
    rag_last_file_name: str = Field(default="", max_length=512)
    messages: list[ChatPersistMessageItem] = Field(default_factory=list, max_length=2000)


class ChatThreadResponse(BaseModel):
    thread_id: str
    mode: Literal["chatbot", "hybrid_rag"]
    rag_document_id: Optional[str] = None
    conversation_summary: str
    rollup_message_index: int
    rag_indexed: bool
    rag_last_file_name: str
    messages: list[ChatPersistMessageItem]


def _validate_thread_id(thread_id: str) -> str:
    tid = (thread_id or "").strip()
    if not tid or len(tid) > CHAT_THREAD_ID_MAX_LEN:
        raise HTTPException(status_code=400, detail="Invalid thread_id")
    return tid


def _row_to_chat_response(row: dict[str, Any]) -> ChatThreadResponse:
    tid = str(row["id"])
    msgs = [
        ChatPersistMessageItem(
            id=int(m.get("id") or 0),
            role=m["role"],
            content=m.get("content") or "",
            meta=m.get("meta") if isinstance(m.get("meta"), dict) else None,
        )
        for m in row.get("messages") or []
    ]
    return ChatThreadResponse(
        thread_id=tid,
        mode=row["mode"],
        rag_document_id=row.get("rag_document_id"),
        conversation_summary=row.get("conversation_summary") or "",
        rollup_message_index=int(row.get("rollup_message_index") or 0),
        rag_indexed=bool(row.get("rag_indexed")),
        rag_last_file_name=row.get("rag_last_file_name") or "",
        messages=msgs,
    )


def _chatbot_history_cap() -> int:
    """Max prior user+assistant messages to include (bounded for local CPU)."""
    n = max(8, int(settings.LOCAL_CHAT_CPU_HISTORY_MESSAGES) * 2)
    return min(n, 32)


def _trim_chat_content(text: str) -> str:
    cap = int(settings.LOCAL_CHAT_CPU_MSG_CHAR_CAP)
    if cap <= 0 or len(text) <= cap:
        return text
    return text[:cap] + "\n…"


def _trim_conversation_summary_for_system(text: str) -> str:
    """Keep system prompt bounded when injecting a long running summary."""
    cap = int(settings.LOCAL_CHAT_CPU_MSG_CHAR_CAP)
    if cap <= 0:
        limit = 16_000
    else:
        limit = min(CONVERSATION_SUMMARY_MAX_CHARS, max(cap * 2, 8_000))
    t = (text or "").strip()
    if len(t) <= limit:
        return t
    return t[:limit] + "\n…"


@router.post("/conversation/rollup", response_model=ConversationRollupResponse)
async def conversation_rollup(body: ConversationRollupRequest):
    """
    Merge prior_summary with recent user/assistant exchanges into one updated summary (local GGUF or Azure).
    """
    lines: list[str] = []
    ps = (body.prior_summary or "").strip()
    if ps:
        lines.append(f"Prior summary:\n{ps}")
    lines.append("New exchanges:")
    for ex in body.exchanges:
        role = ex.role.upper()
        c = (ex.content or "").strip()
        if settings.LOCAL_CHAT_CPU_MSG_CHAR_CAP > 0:
            c = _trim_chat_content(c)
        lines.append(f"{role}: {c}")
    user_block = "\n\n".join(lines)

    max_out = body.max_output_tokens
    if max_out is None:
        max_out = min(512, max(128, settings.LOCAL_CHAT_MAX_TOKENS))

    messages: list[dict[str, str]] = [
        {"role": "system", "content": ROLLUP_SYSTEM},
        {"role": "user", "content": user_block},
    ]

    try:
        if llm_provider_is_local():
            from app.core.local_llm import local_chat_complete_messages, local_chat_gguf_configured

            if not local_chat_gguf_configured():
                raise HTTPException(
                    status_code=503,
                    detail="Local chat model not configured (set LOCAL_CHAT_GGUF_PATH to your .gguf file)",
                )
            summary = await asyncio.to_thread(local_chat_complete_messages, messages, max_out)
            return ConversationRollupResponse(summary=(summary or "").strip(), model="local_gguf")

        if not settings.AZURE_GPT_KEY or not settings.AZURE_GPT_ENDPOINT:
            raise HTTPException(status_code=503, detail="Azure OpenAI not configured (AZURE_GPT_KEY / AZURE_GPT_ENDPOINT)")
        if not settings.AZURE_GPT_DEPLOYMENT:
            raise HTTPException(status_code=503, detail="Azure deployment not configured (AZURE_GPT_DEPLOYMENT)")

        client = AzureOpenAI(
            azure_endpoint=settings.AZURE_GPT_ENDPOINT.rstrip("/"),
            api_key=settings.AZURE_GPT_KEY,
            api_version=settings.AZURE_GPT_API_VERSION,
        )
        resp = client.chat.completions.create(
            model=settings.AZURE_GPT_DEPLOYMENT,
            messages=messages,
            max_completion_tokens=min(4096, max(64, max_out)),
        )
        content = (resp.choices[0].message.content or "").strip()
        return ConversationRollupResponse(summary=content, model=settings.AZURE_GPT_DEPLOYMENT)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("conversation rollup failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/chatbot", response_model=ChatbotResponse)
async def chatbot(body: ChatbotRequest):
    """
    Stateless multi-turn chat using LOCAL_CHAT_GGUF_PATH (local) or Azure OpenAI (when LLM_PROVIDER=azure).
    """
    msg = body.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message is required")

    cap = _chatbot_history_cap()
    hist = body.history[-cap:] if len(body.history) > cap else list(body.history)

    system_text = CHATBOT_SYSTEM
    raw_summary = (body.conversation_summary or "").strip()
    if raw_summary:
        system_text = (
            f"{CHATBOT_SYSTEM}\n\nEarlier conversation (summary):\n{_trim_conversation_summary_for_system(raw_summary)}"
        )

    messages: list[dict[str, str]] = [{"role": "system", "content": system_text}]
    if llm_provider_is_local() and settings.LOCAL_CHAT_CPU_MSG_CHAR_CAP > 0:
        for h in hist:
            messages.append(
                {"role": h.role, "content": _trim_chat_content((h.content or "").strip())}
            )
    else:
        for h in hist:
            messages.append({"role": h.role, "content": (h.content or "").strip()})
    messages.append({"role": "user", "content": msg})

    try:
        if llm_provider_is_local():
            from app.core.local_llm import local_chat_complete_messages, local_chat_gguf_configured

            if not local_chat_gguf_configured():
                raise HTTPException(
                    status_code=503,
                    detail="Local chat model not configured (set LOCAL_CHAT_GGUF_PATH to your .gguf file)",
                )
            answer = await asyncio.to_thread(
                local_chat_complete_messages,
                messages,
                settings.LOCAL_CHAT_MAX_TOKENS,
            )
            return ChatbotResponse(answer=(answer or "").strip(), model="local_gguf")

        if not settings.AZURE_GPT_KEY or not settings.AZURE_GPT_ENDPOINT:
            raise HTTPException(status_code=503, detail="Azure OpenAI not configured (AZURE_GPT_KEY / AZURE_GPT_ENDPOINT)")
        if not settings.AZURE_GPT_DEPLOYMENT:
            raise HTTPException(status_code=503, detail="Azure deployment not configured (AZURE_GPT_DEPLOYMENT)")

        client = AzureOpenAI(
            azure_endpoint=settings.AZURE_GPT_ENDPOINT.rstrip("/"),
            api_key=settings.AZURE_GPT_KEY,
            api_version=settings.AZURE_GPT_API_VERSION,
        )
        resp = client.chat.completions.create(
            model=settings.AZURE_GPT_DEPLOYMENT,
            messages=messages,
            max_completion_tokens=min(4096, max(256, settings.LOCAL_CHAT_MAX_TOKENS)),
        )
        content = (resp.choices[0].message.content or "").strip()
        return ChatbotResponse(answer=content, model=settings.AZURE_GPT_DEPLOYMENT)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("chatbot failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/chat/threads/{thread_id}", response_model=ChatThreadResponse)
async def get_persisted_chat_thread(thread_id: str):
    """Load chatbot or Hybrid RAG thread state from SQLite (PRD platform DB)."""
    tid = _validate_thread_id(thread_id)
    row = get_chat_thread_full(tid)
    if not row:
        raise HTTPException(status_code=404, detail="Thread not found")
    return _row_to_chat_response(row)


@router.put("/chat/threads/{thread_id}", response_model=ChatThreadResponse)
async def put_persisted_chat_thread(thread_id: str, body: ChatThreadSaveRequest):
    """Create or replace thread + messages (full snapshot)."""
    tid = _validate_thread_id(thread_id)
    if body.mode == "hybrid_rag":
        if body.rag_document_id and body.rag_document_id.strip() != tid:
            raise HTTPException(
                status_code=400,
                detail="For hybrid_rag, rag_document_id must match the path thread_id",
            )
        rag_doc = tid
    else:
        rag_doc = None
    msgs = [m.model_dump() for m in body.messages]
    save_chat_thread(
        tid,
        body.mode,
        rag_doc,
        body.conversation_summary,
        body.rollup_message_index,
        body.rag_indexed,
        body.rag_last_file_name,
        msgs,
    )
    row = get_chat_thread_full(tid)
    if not row:
        raise HTTPException(status_code=500, detail="Failed to read saved thread")
    return _row_to_chat_response(row)


@router.delete("/chat/threads/{thread_id}")
async def delete_persisted_chat_thread(thread_id: str):
    tid = _validate_thread_id(thread_id)
    delete_chat_thread(tid)
    return {"ok": True, "thread_id": tid}


@router.get("/capabilities")
async def list_capabilities():
    return {"capabilities": CAPABILITIES}


@router.get("/projects/{project_id}")
async def project_status(project_id: str):
    row = get_project(project_id)
    if not row:
        raise HTTPException(status_code=404, detail="Project not found")
    return {
        "project_id": project_id,
        "status": row["status"],
        "chunk_count": row.get("chunk_count", 0),
        "embedding_dim": row.get("embedding_dim"),
        "error_message": row.get("error_message"),
    }


@router.post("/projects/upload")
async def upload_project(
    file: UploadFile = File(...),
    project_id: Optional[str] = Form(None),
):
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Upload a .zip file")

    raw = await file.read()
    max_b = settings.PRD_PLATFORM_MAX_ZIP_MB * 1024 * 1024
    if len(raw) > max_b:
        raise HTTPException(
            status_code=413,
            detail=f"ZIP exceeds {settings.PRD_PLATFORM_MAX_ZIP_MB} MB limit",
        )

    emb_err = prd_embedding_configuration_error()
    if emb_err:
        logger.warning("PRD upload rejected (embeddings not configured): %s", emb_err)
        raise HTTPException(status_code=503, detail=emb_err)

    pid = (project_id or "").strip() or str(uuid.uuid4())
    if get_project(pid):
        delete_project(pid)

    root = Path(settings.UPLOAD_DIR) / "prd_projects" / pid
    src = root / "src"
    faiss_path = root / "index.faiss"

    upsert_project(
        pid,
        "indexing",
        str(src.resolve()),
        str(faiss_path.resolve()),
        0,
        0,
        None,
    )

    try:
        chunk_pairs = extract_and_chunk_zip(raw, src)
        if not chunk_pairs:
            update_project_status(pid, "error", "No indexable text files in archive")
            raise HTTPException(
                status_code=400,
                detail="No indexable text files found (check paths and file types)",
            )

        texts = [t for _, t in chunk_pairs]
        logger.info(
            "PRD upload: %s — %d chunks to embed + FAISS build (local GGUF or Azure per LLM_PROVIDER)",
            pid,
            len(texts),
        )
        vectors, dim = embed_texts(texts)
        index = build_index(vectors, dim)
        save_index(index, faiss_path)

        clear_chunks(pid)
        rows = [(i, fp, txt) for i, (fp, txt) in enumerate(chunk_pairs)]
        insert_chunks(pid, rows)

        upsert_project(
            pid,
            "ready",
            str(src.resolve()),
            str(faiss_path.resolve()),
            dim,
            len(rows),
            None,
        )
        sync_project_graph_to_neo4j(pid, src)
        return {
            "project_id": pid,
            "chunk_count": len(rows),
            "embedding_dim": dim,
            "message": f"Indexed {len(rows)} chunks from {file.filename}",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("PRD upload failed")
        update_project_status(pid, "error", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/generate")
async def generate_doc(body: GenerateRequest):
    cap_id = body.capability.strip()
    answers = body.answers or {}
    project_id = body.project_id

    if not cap_id:
        raise HTTPException(status_code=400, detail="capability is required")

    cap = get_capability(cap_id)
    if not cap:
        raise HTTPException(status_code=400, detail="Unknown capability")

    pid = (project_id or "").strip() or None
    if cap["requires_codebase"]:
        if not pid:
            raise HTTPException(status_code=400, detail="project_id required for this capability")
        row = get_project(pid)
        if not row or row["status"] != "ready":
            raise HTTPException(status_code=400, detail="Project not indexed or not ready")

    # required questions
    for q in cap.get("clarifying_questions", []):
        if q.get("required") and not str(answers.get(q["id"], "")).strip():
            raise HTTPException(status_code=400, detail=f"Missing answer: {q.get('label', q['id'])}")

    str_answers = {str(k): str(v) for k, v in answers.items()}

    logger.info(
        "PRD platform generate started (capability=%s, project_id=%s) — "
        "local llama.cpp may take several minutes (GPU if LLAMA_N_GPU_LAYERS is not 0 and CUDA build; "
        "PRD indexing uses LLAMA_N_GPU_LAYERS_EMBED with partial-GPU fallback LLAMA_N_GPU_LAYERS_EMBED_PARTIAL)",
        cap_id,
        pid or "(none)",
    )
    try:
        # llama.cpp is synchronous; run in a thread so it does not block the event loop (GPU or CPU per .env).
        out = await asyncio.to_thread(generate, cap_id, str_answers, pid)
        logger.info("PRD platform generate finished (capability=%s)", cap_id)
        return out
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("PRD generate failed")
        raise HTTPException(status_code=500, detail=str(e))
