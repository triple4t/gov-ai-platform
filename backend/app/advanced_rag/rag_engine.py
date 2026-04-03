
from __future__ import annotations

import asyncio
import os
import re
import threading
import time
from pathlib import Path
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator

from app.advanced_rag.file_ingest import extract_text_for_rag
from app.core.config import settings

# ---- LangGraph ----
from langgraph.graph import StateGraph, START, END

# ---- Text splitting ----
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---- BM25 ----
from rank_bm25 import BM25Okapi

# ---- ChromaDB ----
import chromadb
from chromadb.config import Settings

# ---- Jina embeddings v4 (GGUF via llama-cpp-python; lazy-loaded) ----
from huggingface_hub import hf_hub_download
from llama_cpp import Llama, LLAMA_POOLING_TYPE_MEAN

load_dotenv()


def _gov_settings_fallback_paths() -> tuple[str, str]:
    """Use main backend LOCAL_* GGUF paths when RAG-specific env vars are unset."""
    try:
        from app.core.config import settings as _s

        return (_s.LOCAL_CHAT_GGUF_PATH or "").strip(), (_s.LOCAL_EMBEDDING_GGUF_PATH or "").strip()
    except Exception:
        return "", ""


_fb_rag_chat_gguf, _fb_rag_embed_gguf = _gov_settings_fallback_paths()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# RAG chat model — local GGUF via llama-cpp-python only.
# Prefer RAG_LLM_GGUF_PATH; else LOCAL_CHAT_GGUF_PATH from root config; else HF download defaults.
RAG_LLM_GGUF_PATH   = os.getenv("RAG_LLM_GGUF_PATH", "").strip() or _fb_rag_chat_gguf
RAG_LLM_HF_REPO     = (os.getenv("RAG_LLM_HF_REPO") or "unsloth/Qwen3.5-9B-GGUF").strip()
RAG_LLM_HF_FILENAME = (os.getenv("RAG_LLM_HF_FILENAME") or "Qwen3.5-9B-Q6_K.gguf").strip()
# Default 8192: same order of magnitude as main chat LLM; 32k + 9B + Jina embed on one GPU often fails llama_context.
RAG_LLM_N_CTX       = int(os.getenv("RAG_LLM_N_CTX", "8192"))
RAG_LLM_N_GPU       = int(os.getenv("RAG_LLM_N_GPU_LAYERS", "-1"))
# After full-GPU (-1) fails, try this many layers before CPU (0 = skip).
RAG_LLM_N_GPU_PARTIAL = int(os.getenv("RAG_LLM_N_GPU_LAYERS_PARTIAL", "32"))
RAG_LLM_CHAT_FORMAT = (os.getenv("RAG_LLM_CHAT_FORMAT", "qwen") or "qwen").strip()

# Jina text-retrieval GGUF (IQ3_M — memory-efficient; embed on OCR text)
JINA_EMBED_REPO     = os.getenv("JINA_EMBED_REPO", "jinaai/jina-embeddings-v4-text-retrieval-GGUF")
JINA_EMBED_FILENAME = os.getenv("JINA_EMBED_FILENAME", "jina-embeddings-v4-text-retrieval-IQ3_M.gguf")
JINA_EMBED_N_CTX    = int(os.getenv("JINA_EMBED_N_CTX", "8192"))
JINA_EMBED_N_BATCH  = int(os.getenv("JINA_EMBED_N_BATCH", "2048"))
# -1 = all layers on GPU (same semantics as llama.cpp). Set 0 to force CPU embeds.
JINA_EMBED_N_GPU    = int(os.getenv("JINA_EMBED_N_GPU_LAYERS", "-1"))
# Prefer JINA_EMBED_LOCAL_PATH; else LOCAL_EMBEDDING_GGUF_PATH; else hf_hub_download.
JINA_EMBED_LOCAL_PATH = os.getenv("JINA_EMBED_LOCAL_PATH", "").strip() or _fb_rag_embed_gguf

EMBEDDING_MODEL_LABEL = (
    Path(JINA_EMBED_LOCAL_PATH).name
    if JINA_EMBED_LOCAL_PATH and Path(JINA_EMBED_LOCAL_PATH).is_file()
    else f"{JINA_EMBED_REPO}:{JINA_EMBED_FILENAME}"
)

# Chunks per Jina embed() call (keep small for peak RAM with IQ quant)
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "4"))
BM25_TOP_K          = 15           # BM25 candidate pool
VECTOR_TOP_K        = 15           # ChromaDB candidate pool
RRF_K               = 60           # RRF constant
FINAL_TOP_K         = 8            # chunks sent to LLM after RRF
MAX_RETRY_COUNT     = 2            # max query rewrites in LangGraph
CHUNK_SIZE          = 1000         # characters per chunk
CHUNK_OVERLAP       = 200          # character overlap between chunks
MAX_GRADING_WORKERS = 8            # parallel threads for document grading
# RAG context budget for the generator (chars). Size alongside RAG_LLM_N_CTX.
MAX_RAG_CONTEXT_CHARS = int(os.getenv("MAX_RAG_CONTEXT_CHARS", "64000"))

# ---------------------------------------------------------------------------
# Lazy Jina GGUF embedder (loaded on first ingest/query — saves RAM at OCR-only startup)
# ---------------------------------------------------------------------------
_jina_embed_llm: Optional[Llama] = None
_jina_embed_load_error: Optional[str] = None
_jina_lock = threading.Lock()


def _jina_llama_from_path(model_path: str, n_gpu_layers: int) -> Llama:
    return Llama(
        model_path=model_path,
        embedding=True,
        pooling_type=LLAMA_POOLING_TYPE_MEAN,
        n_ctx=JINA_EMBED_N_CTX,
        n_batch=JINA_EMBED_N_BATCH,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )


def _get_jina_embedder() -> Llama:
    """
    Load jina-embeddings-v4-text-retrieval GGUF once (embedding=True, mean pooling).
    Retrieval variant requires 'Query: ' for questions and 'Passage: ' for document chunks.
    """
    global _jina_embed_llm, _jina_embed_load_error
    if _jina_embed_llm is not None:
        return _jina_embed_llm
    if _jina_embed_load_error is not None:
        raise RuntimeError(
            f"Jina embedding model failed earlier: {_jina_embed_load_error}. "
            "Fix the error, restart the server, or set JINA_EMBED_REPO / JINA_EMBED_FILENAME."
        )
    with _jina_lock:
        if _jina_embed_llm is not None:
            return _jina_embed_llm
        try:
            if JINA_EMBED_LOCAL_PATH and Path(JINA_EMBED_LOCAL_PATH).is_file():
                model_path = str(Path(JINA_EMBED_LOCAL_PATH).resolve())
                label = Path(model_path).name
            else:
                raw_path = hf_hub_download(repo_id=JINA_EMBED_REPO, filename=JINA_EMBED_FILENAME)
                model_path = str(Path(raw_path).resolve())
                label = JINA_EMBED_FILENAME
            effective_gpu = JINA_EMBED_N_GPU
            print(
                f"⏳ Loading Jina embeddings GGUF ({label}) "
                f"[n_ctx={JINA_EMBED_N_CTX}, n_gpu_layers={effective_gpu}] ...\n"
                f"   path: {model_path}"
            )
            try:
                _jina_embed_llm = _jina_llama_from_path(model_path, effective_gpu)
            except Exception as gpu_err:
                if effective_gpu == 0:
                    raise
                print(
                    f"⚠️  Jina load failed with n_gpu_layers={effective_gpu}: {gpu_err}\n"
                    f"   Retrying on CPU (n_gpu_layers=0). "
                    f"Set JINA_EMBED_N_GPU_LAYERS=0 in backend/.env if you do not have a CUDA GPU."
                )
                effective_gpu = 0
                _jina_embed_llm = _jina_llama_from_path(model_path, 0)
            _jina_device = (
                "GPU (all layers)"
                if effective_gpu == -1
                else ("GPU" if effective_gpu > 0 else "CPU")
            )
            print(
                f"✅ Jina embeddings ready [{_jina_device}, n_gpu_layers={effective_gpu}]: "
                f"{EMBEDDING_MODEL_LABEL}"
            )
        except Exception as e:
            _jina_embed_load_error = str(e)
            raise RuntimeError(
                f"Failed to load Jina embedding GGUF: {e}. "
                f"If the file may be incomplete, delete the repo folder under ~/.cache/huggingface/hub "
                f"for `{JINA_EMBED_REPO}` and retry."
            ) from e
        return _jina_embed_llm

# ---------------------------------------------------------------------------
# Lazy Qwen GGUF chat model for RAG (grading / generation)
# ---------------------------------------------------------------------------
_rag_chat_llm: Optional[Llama] = None
_rag_chat_load_error: Optional[str] = None
_rag_chat_lock = threading.Lock()


def _rag_llm_model_label() -> str:
    if RAG_LLM_GGUF_PATH:
        return str(Path(RAG_LLM_GGUF_PATH).expanduser())
    return f"{RAG_LLM_HF_REPO}:{RAG_LLM_HF_FILENAME}"


def _resolve_rag_chat_gguf_path() -> str:
    if RAG_LLM_GGUF_PATH:
        path = Path(RAG_LLM_GGUF_PATH).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"RAG_LLM_GGUF_PATH is not a file: {path}. "
                "Point it at your Qwen .gguf, or clear RAG_LLM_GGUF_PATH to use Hugging Face "
                f"(RAG_LLM_HF_REPO / RAG_LLM_HF_FILENAME, default {RAG_LLM_HF_REPO})."
            )
        return str(path)
    raw = hf_hub_download(repo_id=RAG_LLM_HF_REPO, filename=RAG_LLM_HF_FILENAME)
    return str(Path(raw).resolve())


def _rag_llama_from_path(model_path: str, n_gpu_layers: int) -> Llama:
    return Llama(
        model_path=model_path,
        chat_format=RAG_LLM_CHAT_FORMAT,
        n_ctx=RAG_LLM_N_CTX,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )


def _rag_reuses_warmed_local_chat_llama() -> bool:
    """
    When RAG points at the same .gguf as LOCAL_CHAT_GGUF_PATH, reuse get_chat_llama()
    (already loaded on GPU at startup) instead of a second Llama() that usually OOMs next to Jina.
    """
    if not RAG_LLM_GGUF_PATH:
        return False
    try:
        from app.core.config import settings

        lp = (settings.LOCAL_CHAT_GGUF_PATH or "").strip()
        if not lp or not os.path.isfile(lp):
            return False
        rp = str(Path(RAG_LLM_GGUF_PATH).expanduser().resolve())
        lp_res = str(Path(lp).expanduser().resolve())
        return os.path.normcase(rp) == os.path.normcase(lp_res)
    except OSError:
        return False


def _rag_llm_gpu_fallback_chain() -> List[int]:
    """Try primary GPU setting, then partial GPU if primary was -1, then CPU."""
    primary = RAG_LLM_N_GPU
    seen: set[int] = set()
    chain: List[int] = []
    for x in (primary, RAG_LLM_N_GPU_PARTIAL if primary == -1 else None, 0):
        if x is None:
            continue
        if x in seen:
            continue
        seen.add(x)
        chain.append(x)
    return chain


def _get_rag_chat_llm() -> Llama:
    global _rag_chat_llm, _rag_chat_load_error
    if _rag_chat_llm is not None:
        return _rag_chat_llm
    if _rag_chat_load_error is not None:
        raise RuntimeError(
            f"RAG chat GGUF failed earlier: {_rag_chat_load_error}. "
            "Fix RAG_LLM_GGUF_PATH / RAG_LLM_HF_* or lower RAG_LLM_N_CTX / set RAG_LLM_N_GPU_LAYERS=0 in backend/.env."
        )
    with _rag_chat_lock:
        if _rag_chat_llm is not None:
            return _rag_chat_llm
        if _rag_chat_load_error is not None:
            raise RuntimeError(
                f"RAG chat GGUF failed earlier: {_rag_chat_load_error}. "
                "Fix RAG_LLM_GGUF_PATH / RAG_LLM_HF_* or lower RAG_LLM_N_CTX / set RAG_LLM_N_GPU_LAYERS=0 in backend/.env."
            )
        if _rag_reuses_warmed_local_chat_llama():
            try:
                from app.core.local_llm import get_chat_llama

                _rag_chat_llm = get_chat_llama()
                print(
                    "✅ RAG reuses warmed local chat GGUF (same file as LOCAL_CHAT_GGUF_PATH) — "
                    "no second load; inference uses the GPU copy from startup."
                )
                return _rag_chat_llm
            except Exception as e:
                print(f"⚠️  RAG shared-chat reuse failed ({e}); loading a dedicated RAG Llama…")

        try:
            model_path = _resolve_rag_chat_gguf_path()
            last_err: Optional[Exception] = None
            for effective_gpu in _rag_llm_gpu_fallback_chain():
                print(
                    f"⏳ Loading RAG chat GGUF [n_ctx={RAG_LLM_N_CTX}, "
                    f"n_gpu_layers={effective_gpu}, chat_format={RAG_LLM_CHAT_FORMAT}] ...\n"
                    f"   path: {model_path}"
                )
                try:
                    _rag_chat_llm = _rag_llama_from_path(model_path, effective_gpu)
                    print(f"✅ RAG chat GGUF ready (n_gpu_layers={effective_gpu}).")
                    return _rag_chat_llm
                except Exception as e:
                    last_err = e
                    print(
                        f"⚠️  RAG chat GGUF load failed with n_gpu_layers={effective_gpu}: {e}\n"
                        "   Trying next n_gpu_layers fallback (see RAG_LLM_N_GPU_LAYERS / RAG_LLM_N_GPU_LAYERS_PARTIAL)…"
                    )
            assert last_err is not None
            _rag_chat_load_error = str(last_err)
            raise RuntimeError(f"Failed to load RAG chat GGUF: {last_err}") from last_err
        except RuntimeError:
            raise
        except Exception as e:
            _rag_chat_load_error = str(e)
            raise RuntimeError(f"Failed to load RAG chat GGUF: {e}") from e


# ---------------------------------------------------------------------------
# In-memory store: document_id → { "bm25": BM25Okapi, "chunks": List[str] }
# ---------------------------------------------------------------------------
_bm25_store: Dict[str, Dict[str, Any]] = {}

# Project workspace: stacked sources per RAG id (sanitized project_id). Rebuilt on each append/zip batch.
_project_source_blocks: Dict[str, List[str]] = {}


def sanitize_rag_id(raw: str) -> str:
    """Chroma collection / BM25 key (alphanumeric, underscore, hyphen; max 63)."""
    s = re.sub(r"[^a-zA-Z0-9_-]", "_", (raw or "").strip())[:63]
    return s if s else "project"


def clear_project_sources(doc_id: str) -> None:
    _project_source_blocks.pop(doc_id, None)


def _format_project_source_block(logical_path: str, text: str, content_kind: Optional[str]) -> str:
    path = (logical_path or "unknown").strip() or "unknown"
    header = f"=== SOURCE: {path} ===\n"
    ck: Optional[str] = None
    if content_kind:
        k = str(content_kind).strip().lower().replace(" ", "_").replace("-", "_")
        if k == "cdg":
            k = "dependency_graph"
        if k in _RAG_CONTENT_KIND_TAGS:
            ck = k
    if ck:
        header += _RAG_CONTENT_KIND_TAGS[ck] + "\n\n"
    return header + (text or "").strip()


def _reindex_project_corpus(doc_id: str) -> IngestResponse:
    blocks = _project_source_blocks.get(doc_id) or []
    if not blocks:
        raise ValueError("No sources in project — append at least one source first.")
    full_corpus = "\n\n---SOURCE_SEPARATOR---\n\n".join(blocks)
    return ingest_document(full_corpus, doc_id, None)


def append_project_source_and_reindex(
    project_id_raw: str,
    logical_path: str,
    text: str,
    content_kind: Optional[str] = None,
) -> IngestResponse:
    """
    Append one logical file to a project corpus, then re-embed the full stacked corpus.
    All sources share one BM25 + Chroma collection keyed by sanitized project_id.
    """
    doc_id = sanitize_rag_id(project_id_raw)
    if not (text or "").strip():
        raise ValueError("text cannot be empty.")
    block = _format_project_source_block(logical_path, text, content_kind)
    lst = _project_source_blocks.setdefault(doc_id, [])
    lst.append(block)
    print(f"\n📎 [Project] append source '{logical_path}' → project_id='{doc_id}' (total sources={len(lst)})")
    return _reindex_project_corpus(doc_id)


def extend_project_sources_and_reindex(
    project_id_raw: str,
    items: List[tuple[str, str, Optional[str]]],
) -> IngestResponse:
    """Append many (path, text, content_kind) then single reindex (e.g. ZIP extract)."""
    doc_id = sanitize_rag_id(project_id_raw)
    lst = _project_source_blocks.setdefault(doc_id, [])
    n = 0
    for logical_path, text, content_kind in items:
        if not (text or "").strip():
            continue
        lst.append(_format_project_source_block(logical_path, text, content_kind))
        n += 1
    if n == 0:
        raise ValueError("No extractable text in batch.")
    print(f"\n📎 [Project] batch append {n} sources → project_id='{doc_id}' (total sources={len(lst)})")
    return _reindex_project_corpus(doc_id)


def replace_project_from_full_text(
    project_id_raw: str,
    logical_path: str,
    text: str,
    content_kind: Optional[str] = None,
) -> IngestResponse:
    """Reset stacked sources to a single block (e.g. legacy full replace ingest)."""
    doc_id = sanitize_rag_id(project_id_raw)
    _project_source_blocks[doc_id] = [_format_project_source_block(logical_path, text, content_kind)]
    return _reindex_project_corpus(doc_id)

# Single persistent ChromaDB client (in-memory, no disk needed)
_chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))

# ---------------------------------------------------------------------------
# Pydantic models for the API
# ---------------------------------------------------------------------------

# Optional tags for PRD-platform style docs (flow diagrams, SOPs, architecture, etc.)
_RAG_CONTENT_KIND_TAGS: Dict[str, str] = {
    "flowchart": "[Content kind: project flow diagram / flowchart — OCR from image]",
    "diagram": "[Content kind: technical diagram — OCR from image]",
    "architecture": "[Content kind: system or code architecture description / diagram]",
    "prd": "[Content kind: product requirements document (PRD)]",
    "sop": "[Content kind: standard operating procedure (SOP)]",
    "code_docs": "[Content kind: technical / code documentation]",
    "code_review": "[Content kind: code review notes]",
    "dependency_graph": "[Content kind: code dependency graph (CDG)]",
    "general": "[Content kind: general document]",
}


class IngestRequest(BaseModel):
    text: str
    document_id: str  # typically the filename stem
    content_kind: Optional[str] = Field(
        default=None,
        description=(
            "Document profile for retrieval (e.g. flowchart, diagram, prd, sop, architecture). "
            "Prepends a short tag to indexed text. Omit or use 'general' for default."
        ),
    )

    @field_validator("content_kind", mode="before")
    @classmethod
    def _normalize_content_kind(cls, v: Optional[str]) -> Optional[str]:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        k = str(v).strip().lower().replace(" ", "_").replace("-", "_")
        if k == "cdg":
            return "dependency_graph"
        if k not in _RAG_CONTENT_KIND_TAGS:
            raise ValueError(
                f"content_kind must be one of {sorted(_RAG_CONTENT_KIND_TAGS.keys())} or null"
            )
        return k

class IngestResponse(BaseModel):
    status: str
    document_id: str
    chunks_indexed: int
    bm25_ready: bool
    vector_ready: bool
    elapsed_seconds: float


class ProjectAppendRequest(BaseModel):
    """Append one extracted source into a project workspace and re-embed the full corpus."""

    project_id: str
    logical_path: str = Field(..., min_length=1, max_length=2048, description="e.g. docs/prd.pdf or src/app.py")
    text: str
    content_kind: Optional[str] = Field(
        default=None,
        description="Optional chunk tag: prd, code_docs, architecture, … (same as /rag/ingest).",
    )

    @field_validator("content_kind", mode="before")
    @classmethod
    def _normalize_ck(cls, v: Optional[str]) -> Optional[str]:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        k = str(v).strip().lower().replace(" ", "_").replace("-", "_")
        if k == "cdg":
            return "dependency_graph"
        if k not in _RAG_CONTENT_KIND_TAGS:
            raise ValueError(
                f"content_kind must be one of {sorted(_RAG_CONTENT_KIND_TAGS.keys())} or null"
            )
        return k


class ProjectBatchItem(BaseModel):
    logical_path: str
    text: str
    content_kind: Optional[str] = None

    @field_validator("content_kind", mode="before")
    @classmethod
    def _normalize_ck(cls, v: Optional[str]) -> Optional[str]:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        k = str(v).strip().lower().replace(" ", "_").replace("-", "_")
        if k == "cdg":
            return "dependency_graph"
        if k not in _RAG_CONTENT_KIND_TAGS:
            raise ValueError(
                f"content_kind must be one of {sorted(_RAG_CONTENT_KIND_TAGS.keys())} or null"
            )
        return k


class ProjectBatchIngestRequest(BaseModel):
    project_id: str
    items: List[ProjectBatchItem] = Field(..., min_length=1, max_length=500)


class QueryRequest(BaseModel):
    question: str
    document_id: str
    top_k: Optional[int] = FINAL_TOP_K

class QueryResponse(BaseModel):
    answer: str
    corrected_question: str
    sources: List[str]
    retrieval_scores: Optional[List[float]] = None
    retries: int
    elapsed_seconds: float


# AI-Based PRD Platform — structured artifact generation (RAG + Qwen GGUF)
ARTIFACT_TOP_K_DEFAULT = 12

_ARTIFACT_CONFIG: Dict[str, Dict[str, str]] = {
    "prd": {
        "retrieval_query": (
            "product requirements PRD stakeholders goals scope features user stories "
            "acceptance criteria milestones roadmap success metrics risks assumptions"
        ),
        "grade_question": (
            "Does this text help draft or update a Product Requirements Document (PRD) "
            "(goals, scope, features, users, constraints)?"
        ),
        "system_prompt": (
            "You are a senior product manager. Using ONLY the provided context from project documents, "
            "produce a clear Product Requirements Document in Markdown.\n\n"
            "Include these sections (omit a section only if the context has absolutely no basis for it; "
            "then state 'Not specified in source documents.'):\n"
            "1. Executive summary\n2. Problem statement & goals\n3. Users / personas\n"
            "4. Scope (in / out)\n5. Functional requirements (bulleted, with IDs FR-001…)\n"
            "6. Non-functional requirements\n7. Success metrics & KPIs\n"
            "8. Milestones / phases (if inferable)\n9. Dependencies & assumptions\n10. Open questions\n\n"
            "Be concrete. Do not invent facts; infer cautiously and label inferences as such."
        ),
    },
    "code_docs": {
        "retrieval_query": (
            "technical documentation API module class function endpoint database schema "
            "configuration deployment build test code structure"
        ),
        "grade_question": (
            "Does this text help write technical or code documentation (APIs, modules, behavior, setup)?"
        ),
        "system_prompt": (
            "You are a staff engineer. Using ONLY the provided context, write technical documentation in Markdown.\n\n"
            "Cover: overview; architecture touchpoints; main modules or components; "
            "key APIs or interfaces (if any); data models or storage; configuration & environment; "
            "build/run notes if present; testing notes if present.\n"
            "Use headings, tables where useful, and ``` code fences for any code-like snippets from the context."
        ),
    },
    "flow_diagram": {
        "retrieval_query": (
            "workflow process flowchart steps sequence decision swimlane user journey "
            "state transition pipeline stages arrows"
        ),
        "grade_question": (
            "Does this text describe a process, workflow, user journey, or flow that could be diagrammed?"
        ),
        "system_prompt": (
            "You are a solutions architect. Using ONLY the provided context, describe the project or process flow.\n\n"
            "Output Markdown with:\n"
            "1. A short narrative of the flow (bulleted).\n"
            "2. A Mermaid **flowchart** in a fenced block: ```mermaid ... ```\n"
            "   Use `flowchart TD` or `flowchart LR`, concise node IDs, and labels grounded in the context.\n"
            "   CRITICAL Mermaid syntax rule: use ONLY the pipe label style for ALL arrows.\n"
            "   CORRECT:   A -->|Yes| B       A -->|No - Redirect| C\n"
            "   WRONG:     A -- Yes --> B      A -- No -->|Redirect| C\n"
            "   Never mix '-- label -->' and '-->|label|' on the same arrow.\n"
            "3. Notes on actors/systems if identifiable.\n"
            "If the context is too thin, say so and produce a minimal diagram labeled 'hypothesis — verify'."
        ),
    },
    "sop": {
        "retrieval_query": (
            "standard operating procedure SOP steps checklist roles responsibilities "
            "approval escalation maintenance incident runbook procedure policy"
        ),
        "grade_question": (
            "Does this text support writing a standard operating procedure (steps, roles, checks)?"
        ),
        "system_prompt": (
            "You are a compliance-aware operations lead. Using ONLY the provided context, write an SOP in Markdown.\n\n"
            "Include: title; purpose; scope; roles & responsibilities; prerequisites; "
            "numbered procedure steps; verification / quality checks; exceptions; references to sections in the source if clear.\n"
            "Use imperative voice for steps. Do not fabricate regulatory claims."
        ),
    },
    "code_review": {
        "retrieval_query": (
            "code review pull request defect bug security performance refactor quality "
            "lint test coverage technical debt risk"
        ),
        "grade_question": (
            "Does this text relate to code quality, review feedback, defects, or implementation risks?"
        ),
        "system_prompt": (
            "You are a principal engineer doing a code review-style analysis based ONLY on the provided context "
            "(which may include OCR'd code fragments, specs, or review notes).\n\n"
            "Produce Markdown with: summary; strengths; issues / risks (severity: High/Med/Low); "
            "test & observability gaps; concrete recommendations.\n"
            "If there is no actual code, frame findings as 'documentation/spec review' and stay honest about limits."
        ),
    },
    "architecture": {
        "retrieval_query": (
            "system architecture components services layers deployment infrastructure integration "
            "data flow diagram context diagram boundaries scalability"
        ),
        "grade_question": (
            "Does this text describe system structure, components, deployment, or technical architecture?"
        ),
        "system_prompt": (
            "You are a software architect. Using ONLY the provided context, describe the project architecture in Markdown.\n\n"
            "Include: context / system overview; logical components & responsibilities; "
            "data flows; external integrations; deployment or runtime view (if inferable); "
            "key technology choices mentioned; risks and evolution notes.\n"
            "Optionally add a Mermaid **C4-style or component** diagram in ```mermaid``` if the context supports it."
        ),
    },
    "dependency_graph": {
        "retrieval_query": (
            "dependency import require module package library service calls graph "
            "depends on references links between components"
        ),
        "grade_question": (
            "Does this text mention modules, packages, services, or dependencies between parts of the system?"
        ),
        "system_prompt": (
            "You are a tooling engineer documenting dependencies. Using ONLY the provided context, output Markdown with:\n\n"
            "1. A bullet list of modules/packages/services and what they depend on (as stated or clearly implied).\n"
            "2. A Mermaid **graph** in a fenced block: ```mermaid\ngraph TD\n ... ``` "
            "or ```mermaid\nflowchart LR\n ... ``` showing nodes and directed edges. "
            "Label edges briefly.\n"
            "   CRITICAL Mermaid syntax rule: use ONLY the pipe label style for ALL arrows.\n"
            "   CORRECT:   A -->|imports| B\n"
            "   WRONG:     A -- imports --> B\n"
            "   Never mix '-- label -->' and '-->|label|' on the same arrow.\n"
            "3. Legend if some relationships are uncertain (mark as dashed or note 'inferred').\n"
            "Do not invent dependencies not grounded in the context."
        ),
    },
    "full_summary": {
        "retrieval_query": (
            "project overview purpose directory structure components modules "
            "architecture services features codebase summary high level tech stack"
        ),
        "grade_question": (
            "Does this text describe the project's purpose, structure, components, or overall functionality?"
        ),
        "system_prompt": (
            "You are a senior engineer writing a comprehensive project summary report. "
            "Using ONLY the provided context from the uploaded repository, "
            "produce a thorough Markdown summary.\n\n"
            "Include these sections:\n"
            "1. **Project Overview** — purpose, goals, and target users\n"
            "2. **Tech Stack** — languages, frameworks, and key libraries\n"
            "3. **Directory Structure** — overview of top-level folders and their roles\n"
            "4. **Core Modules & Responsibilities** — what each major component does\n"
            "5. **Key Data Flows** — how data moves through the system\n"
            "6. **External Dependencies & Integrations** — third-party services, APIs, databases\n"
            "7. **Deployment & Configuration Notes** — if inferable from Dockerfiles, configs, etc.\n\n"
            "Adapt the depth based on the user's instruction (executive overview vs detailed breakdown). "
            "Be concrete and grounded in what the code actually shows."
        ),
    },
}


# ---------------------------------------------------------------------------
# Generative Config — used when ZIP is skipped (conceptual modules only)
# These are standalone prompts that build docs from user-provided text inputs.
# ---------------------------------------------------------------------------

_GENERATIVE_CONFIG: Dict[str, Dict[str, str]] = {
    "prd": {
        "system_prompt": (
            "You are an expert product manager. The user has provided a set of structured inputs "
            "describing their product or initiative. Based ONLY on these inputs, write a complete, "
            "professional Product Requirements Document (PRD) in Markdown format.\n\n"
            "Structure the PRD with these sections:\n"
            "1. **Executive Summary**\n"
            "2. **Problem Statement**\n"
            "3. **Solution Overview**\n"
            "4. **Primary Users & Personas**\n"
            "5. **Key Features** (with priority: P0/P1/P2)\n"
            "6. **Non-Functional Requirements**\n"
            "7. **Success Metrics & KPIs**\n"
            "8. **Out of Scope**\n"
            "9. **Assumptions & Dependencies**\n"
            "10. **Risks & Edge Cases**\n"
            "11. **Open Questions**\n\n"
            "Be thorough, structured, and professional. Do not invent facts beyond what was provided."
        ),
    },
    "flow_diagram": {
        "system_prompt": (
            "You are an expert solutions architect. The user has described a process or system flow. "
            "Based ONLY on the provided description, produce two things in Markdown:\n\n"
            "1. A brief **narrative description** of the flow (2–4 bullet points per major stage).\n"
            "2. A complete **Mermaid flowchart** in a fenced code block (```mermaid ... ```)\n"
            "   - Use `flowchart TD` (top-down) or `flowchart LR` (left-right) as appropriate\n"
            "   - Use descriptive node labels from the user's description\n"
            "   - Show decision points (diamonds), start/end nodes, and all actors/systems mentioned\n"
            "   - Keep node IDs short (A, B, C or descriptive short codes)\n"
            "   - CRITICAL Mermaid syntax rule: use ONLY the pipe label style for ALL arrows.\n"
            "     CORRECT:   A -->|Yes| B       A -->|No - Redirect| C\n"
            "     WRONG:     A -- Yes --> B      A -- No -->|Redirect| C\n"
            "     Never mix '-- label -->' and '-->|label|' on the same arrow.\n\n"
            "If actors/systems were specified, use swimlanes (subgraph syntax) where appropriate. "
            "Label all arrows to show what triggers each transition."
        ),
    },
    "sop": {
        "system_prompt": (
            "You are an expert operations lead with experience writing Standard Operating Procedures. "
            "The user has provided a topic and context. Based ONLY on these inputs, "
            "write a complete, professional SOP in Markdown format.\n\n"
            "Structure the SOP with these sections:\n"
            "1. **Title & Version**\n"
            "2. **Purpose** — what this SOP achieves\n"
            "3. **Scope** — who this applies to and when\n"
            "4. **Roles & Responsibilities**\n"
            "5. **Prerequisites** — tools, access, knowledge required\n"
            "6. **Procedure** — numbered step-by-step instructions (use imperative voice)\n"
            "7. **Verification & Quality Checks** — how to confirm success\n"
            "8. **Exceptions & Escalation** — what to do when things go wrong\n"
            "9. **References**\n\n"
            "Be precise and actionable. Each step should be clear enough for someone new to follow."
        ),
    },
    "code_docs": {
        "system_prompt": (
            "You are a senior software engineer and technical writer. "
            "The user has described a software system or service using structured inputs. "
            "Based ONLY on the information provided, write complete, professional Technical Documentation "
            "in Markdown format targeted at the specified audience.\n\n"
            "Structure the document as follows:\n"
            "1. **Overview** — what this system does and why it exists\n"
            "2. **Tech Stack** — languages, frameworks, databases, infrastructure\n"
            "3. **Architecture Overview** — high-level description of how it works\n"
            "4. **Modules & Components** — describe each component's responsibility\n"
            "5. **API Reference** — document each endpoint (method, path, request/response)\n"
            "6. **Data Models** — explain key schemas and their fields\n"
            "7. **Authentication & Security**\n"
            "8. **Setup & Quickstart** — step-by-step local development guide\n"
            "9. **Configuration** — key environment variables and their purpose\n"
            "10. **Common Issues & Troubleshooting**\n\n"
            "Adapt depth and detail to the user's requested audience and depth level. "
            "Be concrete and precise. Do not invent details not provided by the user."
        ),
    },
    "architecture": {
        "system_prompt": (
            "You are a senior solutions architect with deep expertise in distributed systems design. "
            "The user has described their system's components, communication patterns, and constraints. "
            "Based ONLY on these inputs, produce a comprehensive Architecture Description document "
            "in Markdown format.\n\n"
            "Structure the output as follows:\n"
            "1. **Executive Summary** — 2-3 sentence overview of the architecture\n"
            "2. **Architecture Style** — chosen pattern and rationale\n"
            "3. **System Components** — table of components with responsibilities\n"
            "4. **Component Interactions** — how services communicate (sync vs async)\n"
            "5. **Data Layer** — storage technologies and their roles\n"
            "6. **Security Architecture** — auth, encryption, network boundaries\n"
            "7. **Deployment Architecture** — infrastructure and hosting\n"
            "8. **External Integrations** — third-party services and APIs\n"
            "9. **Quality Attributes** — how the architecture meets NFRs\n"
            "10. **Architecture Diagram** — produce a Mermaid diagram matching the requested viewpoint.\n\n"
            "For the Mermaid diagram:\n"
            "- Use ```mermaid ... ``` fenced block\n"
            "- Use `graph TD` or `graph LR` or C4-style with `graph TB` + subgraphs\n"
            "- Group related services into subgraphs (e.g. Frontend, Backend, Data Layer)\n"
            "- Show data flow with labeled arrows\n"
            "- External systems in a separate subgraph\n\n"
            "Be thorough. A well-designed architecture doc should be persuasive and precise."
        ),
    },
}


def run_generative_artifact(
    capability_id: str,
    form_data: Dict[str, str],
) -> "GenerateArtifactResponse":
    """
    Generate documentation purely from user-provided text inputs (no RAG retrieval).
    Used when the user skips the ZIP upload on conceptual modules (PRD, Flow Diagram, SOP).

    Args:
        capability_id: one of 'prd', 'flow_diagram', 'sop'
        form_data: dict of field_name -> value from the clarifying questions form

    Returns:
        GenerateArtifactResponse with the generated content
    """
    cid = capability_id.strip().lower().replace("-", "_")
    cfg = _GENERATIVE_CONFIG.get(cid)
    if not cfg:
        allowed = ", ".join(sorted(_GENERATIVE_CONFIG.keys()))
        raise ValueError(
            f"Generative mode is only available for: {allowed}. "
            f"'{capability_id}' requires a codebase ZIP upload."
        )

    t0 = time.time()
    print(f"\n🖊️ [Generative] capability={cid} | form_data keys={list(form_data.keys())}")

    # Build a structured user message from all form fields
    fields_text = "\n".join(
        f"**{key.replace('_', ' ').title()}:** {value}"
        for key, value in form_data.items()
        if value and str(value).strip()
    )
    user_content = (
        f"The user has provided the following information:\n\n"
        f"{fields_text}\n\n"
        f"Please generate the requested documentation based strictly on the information above."
    )

    try:
        content = _call_rag_llm(
            system_prompt=cfg["system_prompt"],
            user_content=user_content,
            max_tokens=4096,
        )
    except Exception as e:
        content = f"Error generating artifact: {e}"

    elapsed = round(time.time() - t0, 2)
    print(f"  🏁 Generative '{cid}' done in {elapsed}s ({len(content)} chars)")

    return GenerateArtifactResponse(
        artifact_type=cid,
        document_id="generative_no_rag",
        content=content or "(empty model response)",
        elapsed_seconds=elapsed,
    )


class GenerateArtifactRequest(BaseModel):
    document_id: str
    artifact_type: str = Field(
        ...,
        description=(
            "One of: prd, code_docs, flow_diagram, sop, code_review, architecture, dependency_graph"
        ),
    )
    user_brief: Optional[str] = Field(
        default=None,
        description="Optional extra instructions (audience, tone, sections to emphasize).",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=4,
        le=24,
        description="Chunks to retrieve before generation (default 12).",
    )


class GenerateArtifactResponse(BaseModel):
    artifact_type: str
    document_id: str
    content: str
    elapsed_seconds: float


class ClearResponse(BaseModel):
    status: str
    document_id: str

# ---------------------------------------------------------------------------
# FastAPI router (mounted in app.py)
# ---------------------------------------------------------------------------
rag_router = APIRouter(prefix="/rag", tags=["Advanced RAG"])

# ===========================================================================
# LOW-LEVEL HELPERS
# ===========================================================================

def _call_rag_llm(
    *,
    system_prompt: str,
    user_content: str,
    max_tokens: int = 1024,
) -> str:
    """RAG chat via local Qwen GGUF (llama-cpp-python)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    llm = _get_rag_chat_llm()
    from app.core.local_llm import llama_chat_completion_lock

    with llama_chat_completion_lock():
        data = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.6,
        )
    raw = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
    # Qwen thinking models emit <think>...</think> blocks — strip them before returning
    import re as _re
    clean = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
    # Auto-fix invalid Mermaid syntax that LLMs commonly generate
    clean = _sanitize_mermaid_in_content(clean)
    return clean


def _sanitize_mermaid_in_content(text: str) -> str:
    """
    Post-process any ```mermaid ... ``` blocks in LLM output to fix common
    syntax errors before they reach the frontend renderer.

    Known issue: Mermaid 11.x does NOT allow mixing both edge-label styles
    on the same arrow, e.g.:
        A -- label -->|pipe-label| B   ← INVALID (parse error)

    Fix strategy:
        A -- No -->|Redirect| B  →  A -->|No - Redirect| B
        A -- Yes --> B           →  A -->|Yes| B
        A -->|label| B           →  kept as-is (already valid)
    """
    import re as _re

    def _fix_block(m: re.Match) -> str:
        block = m.group(0)          # full ```mermaid ... ``` fence
        inner = m.group(1)           # diagram text inside the fence

        # Pattern: A -- label -->|extra| B   (mixed style — INVALID)
        # Group 1: source + leading dash-space, Group 2: dash-label, Group 3: pipe-label, Group 4: target
        mixed = _re.compile(
            r'([ \t]*\S[^\n]*?)\s*--\s+([^-\n>|]+?)\s*-->\|([^|\n]+)\|'
        )
        def _merge_mixed(mm):
            prefix = mm.group(1)
            dash_label = mm.group(2).strip()
            pipe_label = mm.group(3).strip()
            combined = f"{dash_label} - {pipe_label}" if pipe_label else dash_label
            return f"{prefix} -->|{combined}|"
        inner = mixed.sub(_merge_mixed, inner)

        # Pattern: A -- label --> B   (dash label, NO pipe) — convert to pipe style
        dash_only = _re.compile(
            r'([ \t]*\S[^\n]*?)\s*--\s+([^-\n>|]+?)\s*-->(?!\|)'
        )
        def _dash_to_pipe(dm):
            prefix = dm.group(1)
            label = dm.group(2).strip()
            return f"{prefix} -->|{label}|"
        inner = dash_only.sub(_dash_to_pipe, inner)

        return f'```mermaid\n{inner}\n```'

    # Match every ```mermaid ... ``` fenced block (non-greedy)
    pattern = _re.compile(r'```mermaid\n([\s\S]*?)```', _re.IGNORECASE)
    return pattern.sub(_fix_block, text)


# ---------------------------------------------------------------------------
# Jina GGUF embedding helpers (Query: / Passage: prefixes per model card)
# ---------------------------------------------------------------------------

def _jina_prefix_query(text: str) -> str:
    t = text.strip()
    if t.lower().startswith("query:"):
        return t
    return f"Query: {t}"


def _jina_prefix_passage(text: str) -> str:
    t = text.strip()
    if t.lower().startswith("passage:"):
        return t
    return f"Passage: {t}"


def _get_embedding(text: str) -> List[float]:
    """Embed the user question with Jina retrieval prefix (Query:)."""
    model = _get_jina_embedder()
    with _jina_lock:
        emb = model.embed(_jina_prefix_query(text), normalize=True)
    # llama-cpp-python: str input → single vector List[float]; list input → List[List[float]]
    if emb and isinstance(emb[0], (int, float)):
        return emb
    return emb[0]


def _embed_batch_passages(texts: List[str]) -> List[List[float]]:
    """Embed document chunks with Jina retrieval prefix (Passage:)."""
    if not texts:
        return []
    model = _get_jina_embedder()
    # One string per embed() call: batched multi-sequence decode often hits llama_decode -1
    # on Windows with Jina v4 GGUF + llama_cpp 0.3.x; sequential is slower but stable.
    out: List[List[float]] = []
    with _jina_lock:
        for t in texts:
            emb = model.embed(_jina_prefix_passage(t), normalize=True)
            if emb and isinstance(emb[0], (int, float)):
                out.append(emb)
            else:
                out.append(emb[0])
    return out


def _embed_all_chunks_batched(chunks: List[str]) -> List[List[float]]:
    """
    Embed all chunks in small batches to cap peak RAM (IQ3 + shared llama.cpp runtime).
    """
    batches = [
        chunks[i : i + EMBEDDING_BATCH_SIZE]
        for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE)
    ]
    all_embeddings: List[List[float]] = []
    for b_idx, batch in enumerate(batches):
        vectors = _embed_batch_passages(batch)
        all_embeddings.extend(vectors)
        print(f"    Embedded batch {b_idx + 1}/{len(batches)} ({len(batch)} chunks)")
    return all_embeddings


def _tokenize(text: str) -> List[str]:
    """Simple whitespace+punctuation tokenizer for BM25."""
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def _clean_ocr_text(text: str) -> str:
    """
    Light cleanup of OCR-extracted text before chunking.
    Removes excessive whitespace and artifact markers but preserves structure.
    """
    # Remove markdown image artifacts from OCR
    text = re.sub(r"!\[[^\]]*\]\([^\)]*\)", " ", text)
    text = re.sub(r"!\[[^\]]*\]", " ", text)
    text = re.sub(r"\bimg-\d+\.(?:png|jpg|jpeg|gif)\b", " ", text, flags=re.IGNORECASE)
    # Normalise excessive blank lines (keep max 2)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Normalise tabs and trailing spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ===========================================================================
# INDEXER
# ===========================================================================

def _build_bm25(chunks: List[str]) -> BM25Okapi:
    """Build a BM25Okapi index from a list of text chunks."""
    tokenized = [_tokenize(chunk) for chunk in chunks]
    return BM25Okapi(tokenized)


def ingest_document(
    text: str,
    document_id: str,
    content_kind: Optional[str] = None,
) -> IngestResponse:
    """
    Ingest OCR text for a document:
      1. Clean OCR artifacts
      2. Split into chunks (RecursiveCharacterTextSplitter)
      3. Build BM25 index (fast, CPU-only)
      4. Embed chunks with Jina v4 retrieval GGUF (Passage:) → upsert to ChromaDB
    """
    t0 = time.time()

    print(f"\n📥 [RAG Ingest] document_id='{document_id}' | raw text length={len(text):,} chars")

    # Step 1: Clean
    clean_text = _clean_ocr_text(text)
    print(f"  ✓ Cleaned text: {len(clean_text):,} chars")

    # Step 2: Chunk, then tag each chunk (so every vector/BM25 hit carries document profile)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_text(clean_text)
    tag = _RAG_CONTENT_KIND_TAGS.get(content_kind) if content_kind else None
    if tag:
        chunks = [f"{tag}\n\n{c}" for c in chunks]
    print(f"  ✓ Split into {len(chunks):,} chunks")

    if not chunks:
        raise ValueError("No chunks produced — text may be empty or too short.")

    # Step 3: BM25 (pure Python, very fast)
    bm25 = _build_bm25(chunks)
    _bm25_store[document_id] = {"bm25": bm25, "chunks": chunks}
    print(f"  ✓ BM25 index built ({len(chunks)} docs)")

    # Step 4: ChromaDB — embed + upsert
    # Delete existing collection for this document_id if it exists
    try:
        _chroma_client.delete_collection(document_id)
    except Exception:
        pass  # doesn't exist yet

    collection = _chroma_client.create_collection(
        name=document_id,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"  ⏳ Embedding {len(chunks)} chunks with Jina GGUF (batch size={EMBEDDING_BATCH_SIZE})...")
    embeddings = _embed_all_chunks_batched(chunks)

    # Upsert all chunks at once
    ids       = [f"{document_id}__chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"chunk_index": i, "document_id": document_id} for i in range(len(chunks))]
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"  ✓ ChromaDB: {len(chunks)} chunks indexed")

    elapsed = round(time.time() - t0, 2)
    print(f"  🏁 Ingest complete in {elapsed}s")

    return IngestResponse(
        status="indexed",
        document_id=document_id,
        chunks_indexed=len(chunks),
        bm25_ready=True,
        vector_ready=True,
        elapsed_seconds=elapsed,
    )


# ===========================================================================
# RETRIEVAL — BM25 + ChromaDB + RRF
# ===========================================================================

def _bm25_retrieve(query: str, document_id: str, top_k: int = BM25_TOP_K) -> List[tuple[int, float]]:
    """
    Run BM25 search.
    Returns list of (chunk_index, bm25_score) sorted descending.
    """
    store = _bm25_store.get(document_id)
    if not store:
        return []

    bm25: BM25Okapi = store["bm25"]
    q_tokens = _tokenize(query)
    scores = bm25.get_scores(q_tokens)

    # Get top_k indices
    indexed_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(idx, float(score)) for idx, score in indexed_scores[:top_k]]


def _vector_retrieve(query: str, document_id: str, top_k: int = VECTOR_TOP_K) -> List[tuple[int, float]]:
    """
    Run ChromaDB semantic search using Jina retrieval embeddings (Query:).
    Returns list of (chunk_index, distance_score) sorted descending by similarity.
    """
    try:
        collection = _chroma_client.get_collection(document_id)
    except Exception:
        return []

    query_embedding = _get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["metadatas", "distances"],
    )

    if not results["metadatas"] or not results["metadatas"][0]:
        return []

    items = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        chunk_idx = meta["chunk_index"]
        # Cosine distance → similarity score (0=identical, 2=opposite)
        similarity = 1.0 - (dist / 2.0)
        items.append((chunk_idx, float(similarity)))

    return sorted(items, key=lambda x: x[1], reverse=True)


def _rrf_fusion(
    bm25_results:   List[tuple[int, float]],
    vector_results: List[tuple[int, float]],
    k: int = RRF_K,
    top_k: int = FINAL_TOP_K,
) -> List[tuple[int, float]]:
    """
    Reciprocal Rank Fusion.
    score(doc) = Σ 1/(k + rank_in_list)
    Returns sorted list of (chunk_index, rrf_score).
    """
    rrf_scores: Dict[int, float] = {}

    for rank, (chunk_idx, _) in enumerate(bm25_results):
        rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + 1.0 / (k + rank + 1)

    for rank, (chunk_idx, _) in enumerate(vector_results):
        rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + 1.0 / (k + rank + 1)

    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]


def hybrid_retrieve(
    query: str,
    document_id: str,
    top_k: int = FINAL_TOP_K,
) -> tuple[List[str], List[float]]:
    """
    Run BM25 and ChromaDB retrieval IN PARALLEL, then fuse with RRF.
    Returns (list_of_chunk_texts, list_of_rrf_scores).
    """
    store = _bm25_store.get(document_id)
    if not store:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not indexed. POST /rag/ingest first.")

    chunks: List[str] = store["chunks"]

    # ---- Parallel retrieval ----
    bm25_results:   List[tuple[int, float]] = []
    vector_results: List[tuple[int, float]] = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        f_bm25   = executor.submit(_bm25_retrieve,  query, document_id, BM25_TOP_K)
        f_vector = executor.submit(_vector_retrieve, query, document_id, VECTOR_TOP_K)
        bm25_results   = f_bm25.result()
        vector_results = f_vector.result()

    print(f"  BM25 hits: {len(bm25_results)} | Vector hits: {len(vector_results)}")

    # ---- RRF Fusion ----
    fused = _rrf_fusion(bm25_results, vector_results, k=RRF_K, top_k=top_k)

    if not fused:
        # Fallback: return first `top_k` chunks
        return chunks[:top_k], [0.0] * min(top_k, len(chunks))

    result_chunks = [chunks[idx] for idx, _ in fused]
    result_scores = [score for _, score in fused]

    return result_chunks, result_scores


# ===========================================================================
# LANGGRAPH WORKFLOW STATE
# ===========================================================================

class RAGState(TypedDict):
    question:           str
    document_id:        str
    top_k:              int
    documents:          List[str]
    doc_scores:         List[float]
    relevant_docs:      List[str]
    answer:             str
    retry_count:        int


# ===========================================================================
# LANGGRAPH NODES
# ===========================================================================

def node_hybrid_retrieve(state: RAGState) -> RAGState:
    """
    Node 1: Hybrid retrieval — BM25 ∥ ChromaDB → RRF fusion.
    Uses the question for retrieval.
    """
    query       = state["question"]
    document_id = state["document_id"]
    top_k       = state["top_k"]

    print(f"\n[Node: hybrid_retrieve] query='{query[:80]}...' top_k={top_k}")

    docs, scores = hybrid_retrieve(query, document_id, top_k=top_k)

    print(f"  ✓ Retrieved {len(docs)} chunks via RRF")
    return {**state, "documents": docs, "doc_scores": scores}


def node_grade_documents(state: RAGState) -> RAGState:
    """
    Node 2: Grade each retrieved chunk for relevance using Qwen3 GGUF (PARALLEL).
    Irrelevant chunks are filtered out before answer generation.
    """
    question  = state["question"]
    documents = state["documents"]

    print(f"\n[Node: grade_documents] Grading {len(documents)} chunks in parallel...")

    def grade_single(chunk: str) -> bool:
        try:
            verdict = _call_rag_llm(
                system_prompt=(
                    "You are a document relevance grader. "
                    "Given a user question and a document chunk, decide if the chunk "
                    "contains ANY information at all that is relevant to answering the question. "
                    "Be VERY LIBERAL — if the chunk might help answer even part of the question, say YES. "
                    "Only say NO if the chunk is completely unrelated. "
                    "Reply with ONLY 'yes' or 'no'. No explanation."
                ),
                user_content=(
                    f"Question: {question}\n\n"
                    f"Document chunk:\n{chunk}"
                ),
                max_tokens=10,
            )
            result = verdict.strip().lower()
            print(f"    Grader verdict: '{result[:20]}' for chunk starting: '{chunk[:60].strip()}'")
            return result.startswith("y")
        except Exception as e:
            print(f"    Grader error (keeping chunk): {e}")
            return True  # default: keep chunk if grading fails

    relevant_docs: List[str] = []
    with ThreadPoolExecutor(max_workers=min(len(documents), MAX_GRADING_WORKERS)) as executor:
        futures = {executor.submit(grade_single, chunk): chunk for chunk in documents}
        for future in as_completed(futures):
            chunk = futures[future]
            if future.result():
                relevant_docs.append(chunk)

    # Preserve original order
    relevant_docs = [d for d in documents if d in relevant_docs]
    print(f"  ✓ {len(relevant_docs)}/{len(documents)} chunks deemed relevant")

    # SAFETY NET: if grading wiped ALL chunks but we had retrievals, keep top half
    if len(relevant_docs) == 0 and len(documents) > 0:
        print(f"  ⚠️  All {len(documents)} chunks were graded irrelevant — keeping top retrieved chunks as fallback")
        relevant_docs = documents[:max(1, len(documents) // 2)]

    return {**state, "relevant_docs": relevant_docs}


def node_generate_answer(state: RAGState) -> RAGState:
    """
    Node 3: Generate the final answer from relevant chunks using Qwen3 GGUF.
    """
    question      = state["question"]
    relevant_docs = state["relevant_docs"]

    print(f"\n[Node: generate] Generating answer from {len(relevant_docs)} relevant chunks...")

    # Build context — join all relevant chunks with separators
    context_parts = [f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(relevant_docs)]
    context = "\n\n---\n\n".join(context_parts)

    # Keep assembled RAG context within MAX_RAG_CONTEXT_CHARS (~64k default)
    if len(context) > MAX_RAG_CONTEXT_CHARS:
        context = context[:MAX_RAG_CONTEXT_CHARS] + "\n\n[...context truncated for length...]"

    try:
        answer = _call_rag_llm(
            system_prompt=(
                "You are a precise, helpful assistant answering questions about documents. "
                "The context below is extracted from an OCR-processed document. "
                "It may include flowcharts or diagrams transcribed as text (boxes, steps, arrows like A -> B). "
                "Use ONLY the context to answer. "
                "Be specific — cite the chunk number when referencing specific information. "
                "If the answer is not in the context, say exactly: "
                "'I cannot find this information in the document.'"
            ),
            user_content=(
                f"Question: {question}\n\n"
                f"Context:\n{context}\n\n"
                "Answer the question based only on the context above."
            ),
            max_tokens=2048,
        )
    except Exception as e:
        answer = f"Error generating answer: {e}"

    print(f"  ✓ Answer generated ({len(answer)} chars)")
    return {**state, "answer": answer}


def node_rewrite_query(state: RAGState) -> RAGState:
    """
    Node 4: Rewrite the query when no relevant docs were found.
    Aims to paraphrase / disambiguate to improve retrieval on the next attempt.
    """
    question = state["question"]
    retry    = state["retry_count"]

    print(f"\n[Node: rewrite_query] Retry #{retry+1} — rewriting query...")

    try:
        rewritten = _call_rag_llm(
            system_prompt=(
                "You are a query rewriter. "
                "The original query retrieved no relevant results from the document. "
                "Rephrase the query by simplifying it, or substituting terms that might appear in a formal document, "
                "to improve document retrieval. Do NOT output a list of synonyms. Just a single natural question or keyword phrase. "
                "Return ONLY the rewritten query — no explanation."
            ),
            user_content=f"Original query: {question}",
            max_tokens=256,
        )
        rewritten = rewritten.strip() or question
    except Exception:
        rewritten = question

    print(f"  ✓ Rewritten: '{rewritten}'")
    return {**state, "question": rewritten, "retry_count": retry + 1}


# ===========================================================================
# CONDITIONAL ROUTING
# ===========================================================================

def route_after_grading(state: RAGState) -> str:
    """
    After grading:
      - If relevant docs found → generate answer
      - If no relevant docs AND retries remaining → rewrite query
      - If no relevant docs AND retries exhausted → generate with empty context (graceful)
    """
    has_relevant = len(state.get("relevant_docs", [])) > 0
    retry_count  = state.get("retry_count", 0)

    if has_relevant:
        return "generate"
    elif retry_count < MAX_RETRY_COUNT:
        return "rewrite"
    else:
        print("  ⚠️ No relevant docs after max retries, generating graceful not-found answer.")
        return "generate"


# ===========================================================================
# BUILD LANGGRAPH
# ===========================================================================

def _build_rag_graph() -> Any:
    """Compile the LangGraph RAG workflow."""
    builder = StateGraph(RAGState)

    # Add nodes
    builder.add_node("hybrid_retrieve", node_hybrid_retrieve)
    builder.add_node("grade_documents", node_grade_documents)
    builder.add_node("generate",        node_generate_answer)
    builder.add_node("rewrite_query",   node_rewrite_query)

    # Edges
    builder.add_edge(START, "hybrid_retrieve")
    builder.add_edge("hybrid_retrieve", "grade_documents")

    # Conditional: after grading → generate OR rewrite
    builder.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite":  "rewrite_query",
        },
    )

    # After rewrite → go back to retrieve (retry loop)
    builder.add_edge("rewrite_query", "hybrid_retrieve")

    # Generate → END
    builder.add_edge("generate", END)

    return builder.compile()


# Singleton compiled graph (built once at module import)
_rag_graph = _build_rag_graph()


def _grade_chunk_for_artifact(question: str, chunk: str) -> bool:
    """Parallel-friendly: is this chunk useful for the artifact we're generating?"""
    try:
        verdict = _call_rag_llm(
            system_prompt=(
                "You are a document relevance grader. Given a criterion question and a document chunk, "
                "reply ONLY 'yes' or 'no'. Say yes if the chunk could help satisfy the criterion even partially. "
                "Be liberal; only no if completely unrelated."
            ),
            user_content=f"Criterion: {question}\n\nChunk:\n{chunk}",
            max_tokens=10,
        )
        return verdict.strip().lower().startswith("y")
    except Exception as e:
        print(f"    Artifact grader error (keeping chunk): {e}")
        return True


def _filter_chunks_for_artifact(question: str, documents: List[str]) -> List[str]:
    if not documents:
        return []
    relevant: List[str] = []
    with ThreadPoolExecutor(max_workers=min(len(documents), MAX_GRADING_WORKERS)) as executor:
        futures = {executor.submit(_grade_chunk_for_artifact, question, chunk): chunk for chunk in documents}
        for future in as_completed(futures):
            chunk = futures[future]
            if future.result():
                relevant.append(chunk)
    relevant = [d for d in documents if d in relevant]
    if not relevant and documents:
        relevant = documents[: max(1, len(documents) // 2)]
    return relevant


def run_artifact_generation(
    document_id: str,
    artifact_type: str,
    user_brief: Optional[str] = None,
    top_k: Optional[int] = None,
) -> GenerateArtifactResponse:
    """
    Retrieve chunks tailored to artifact type, filter by relevance, then generate with Qwen GGUF.
    """
    at = artifact_type.strip().lower().replace("-", "_")
    if at == "cdg":
        at = "dependency_graph"
    if at == "flowchart" or at == "project_flow":
        at = "flow_diagram"
    cfg = _ARTIFACT_CONFIG.get(at)
    if not cfg:
        allowed = ", ".join(sorted(_ARTIFACT_CONFIG.keys()))
        raise ValueError(f"artifact_type must be one of: {allowed}")

    tk = top_k if top_k is not None else ARTIFACT_TOP_K_DEFAULT
    if document_id not in _bm25_store:
        raise LookupError(f"Document '{document_id}' is not indexed")

    t0 = time.time()
    retrieval_query = cfg["retrieval_query"]
    print(f"\n📄 [Artifact] type={at} doc='{document_id}' top_k={tk}")

    docs, _scores = hybrid_retrieve(retrieval_query, document_id, top_k=tk)
    grade_q = cfg["grade_question"]
    relevant = _filter_chunks_for_artifact(grade_q, docs)
    print(f"  ✓ Artifact retrieval: {len(relevant)} chunks after grading")

    context_parts = [f"[Chunk {i+1}]\n{c}" for i, c in enumerate(relevant)]
    context = "\n\n---\n\n".join(context_parts)
    if len(context) > MAX_RAG_CONTEXT_CHARS:
        context = context[:MAX_RAG_CONTEXT_CHARS] + "\n\n[...context truncated...]"

    brief = (user_brief or "").strip()
    user_content = (
        (f"Additional instructions from the user:\n{brief}\n\n" if brief else "")
        + "Source context (from indexed documents):\n"
        + context
        + "\n\nProduce the deliverable now. Use Markdown."
    )

    try:
        content = _call_rag_llm(
            system_prompt=cfg["system_prompt"],
            user_content=user_content,
            max_tokens=4096,
        )
    except Exception as e:
        content = f"Error generating artifact: {e}"

    elapsed = round(time.time() - t0, 2)
    print(f"  🏁 Artifact '{at}' done in {elapsed}s ({len(content)} chars)")
    return GenerateArtifactResponse(
        artifact_type=at,
        document_id=document_id,
        content=content or "(empty model response)",
        elapsed_seconds=elapsed,
    )


def run_rag_pipeline(
    question:    str,
    document_id: str,
    top_k:       int = FINAL_TOP_K,
) -> RAGState:
    """
    Run the full LangGraph RAG pipeline synchronously.
    Returns the final state dict.
    """
    initial_state: RAGState = {
        "question":      question,
        "document_id":   document_id,
        "top_k":         top_k,
        "documents":     [],
        "doc_scores":    [],
        "relevant_docs": [],
        "answer":        "",
        "retry_count":   0,
    }

    final_state = _rag_graph.invoke(initial_state)
    return final_state


# ===========================================================================
# FASTAPI ENDPOINTS
# ===========================================================================

@rag_router.post("/ingest", response_model=IngestResponse)
async def rag_ingest(payload: IngestRequest):
    """
    Index OCR-extracted text for a document.

    Call this AFTER OCR extraction. Pass the full extracted text and a unique document_id
    (e.g. the filename without extension). Supports texts of 1,000 – 100,000+ lines.

    Optional `content_kind` tags the document for hybrid retrieval (e.g. flowchart, diagram,
    prd, sop, architecture, dependency_graph) — aligned with an AI-Based PRD platform scope.

    Internally:
      - Cleans OCR artifacts
      - Splits into ~1000-char chunks with 200-char overlap
      - Builds BM25 index (CPU, milliseconds)
      - Embeds all chunks with Jina retrieval GGUF → upserts to ChromaDB HNSW
    """
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if not payload.document_id or not payload.document_id.strip():
        raise HTTPException(status_code=400, detail="document_id cannot be empty.")

    doc_id = sanitize_rag_id(payload.document_id)

    print(f"\n🚀 /rag/ingest called | document_id='{doc_id}' | text length={len(payload.text):,}")

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            partial(
                replace_project_from_full_text,
                payload.document_id.strip(),
                f"{doc_id}.txt",
                payload.text,
                payload.content_kind,
            ),
        )
        return result
    except Exception as e:
        print(f"❌ Ingest error: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ingest failed: {str(e)}")


@rag_router.post("/ingest-file", response_model=IngestResponse)
async def rag_ingest_file(
    file: UploadFile = File(...),
    document_id: str = Form(...),
    content_kind: Optional[str] = Form(None),
):
    """
    Upload PDF, DOCX, DOC, plain text, or image (Tesseract) — extract text server-side, then same pipeline as /ingest.
    """
    did = (document_id or "").strip()
    if not did:
        raise HTTPException(status_code=400, detail="document_id cannot be empty.")

    max_mb = max(1, settings.RAG_INGEST_FILE_MAX_MB)
    max_b = max_mb * 1024 * 1024
    raw = await file.read()
    if len(raw) > max_b:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {max_mb} MB limit.",
        )

    try:
        text = extract_text_for_rag(file.filename or "upload", raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        _kind = IngestRequest(text=".", document_id=did, content_kind=content_kind).content_kind
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    doc_id = sanitize_rag_id(did)
    logical = f"{doc_id}_{(file.filename or 'file').replace(chr(92), '/')}"

    print(f"\n🚀 /rag/ingest-file | document_id='{doc_id}' | file={file.filename!r} | text length={len(text):,}")

    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            partial(
                replace_project_from_full_text,
                did,
                logical,
                text,
                _kind,
            ),
        )
    except Exception as e:
        print(f"❌ Ingest-file error: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ingest failed: {str(e)}") from e


@rag_router.post("/project/append", response_model=IngestResponse)
async def rag_project_append(payload: ProjectAppendRequest):
    """
    Append text from one logical file (TXT extract, code file, OCR output, etc.) to a **project** corpus,
    then re-chunk and re-embed the **entire** stacked corpus into one vector index (project_id = RAG id).

    Use this to combine PRD PDFs, DOCX, and code ZIP extracts in one retrieval scope.
    """
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty.")
    if not payload.project_id or not payload.project_id.strip():
        raise HTTPException(status_code=400, detail="project_id cannot be empty.")

    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            partial(
                append_project_source_and_reindex,
                payload.project_id.strip(),
                payload.logical_path.strip(),
                payload.text,
                payload.content_kind,
            ),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ project/append error: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@rag_router.post("/project/ingest-batch", response_model=IngestResponse)
async def rag_project_ingest_batch(payload: ProjectBatchIngestRequest):
    """Append many sources in one request (used after ZIP extraction)."""
    if not payload.project_id or not payload.project_id.strip():
        raise HTTPException(status_code=400, detail="project_id cannot be empty.")

    tuples: List[tuple[str, str, Optional[str]]] = [
        (it.logical_path.strip(), it.text, it.content_kind) for it in payload.items
    ]

    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            partial(extend_project_sources_and_reindex, payload.project_id.strip(), tuples),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ project/ingest-batch error: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@rag_router.get("/project/{project_id}/sources")
async def rag_project_sources(project_id: str):
    """How many stacked sources are in memory for this project (resets on server restart)."""
    doc_id = sanitize_rag_id(project_id)
    blocks = _project_source_blocks.get(doc_id) or []
    return {"project_id": doc_id, "sources_stacked": len(blocks), "indexed": doc_id in _bm25_store}


@rag_router.post("/query", response_model=QueryResponse)
async def rag_query(payload: QueryRequest):
    """
    Query the indexed document using the Advanced RAG pipeline.

    LangGraph workflow:
      1. Hybrid retrieval (BM25 ∥ ChromaDB vector with Jina GGUF, merged via RRF)
      2. Document grading (Qwen3 GGUF filters irrelevant chunks, parallel)
      3. Answer generation (Qwen3 GGUF, cites chunk numbers)
      (Iterative rewrite+retry if no relevant chunks found, max 2 retries)

    Make sure to call /rag/ingest first for your document.
    """
    question    = (payload.question or "").strip()
    document_id = (payload.document_id or "").strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if not document_id:
        raise HTTPException(status_code=400, detail="document_id cannot be empty.")

    doc_id = sanitize_rag_id(document_id)

    if doc_id not in _bm25_store:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{doc_id}' not indexed. POST /rag/ingest or /rag/project/append first."
        )

    print(f"\n🤔 /rag/query | doc='{doc_id}' | question='{question[:80]}'")
    t0 = time.time()

    try:
        loop = asyncio.get_event_loop()
        final_state: RAGState = await loop.run_in_executor(
            None,
            run_rag_pipeline,
            question,
            doc_id,
            payload.top_k or FINAL_TOP_K,
        )
    except Exception as e:
        print(f"❌ RAG pipeline error: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")

    elapsed = round(time.time() - t0, 2)

    answer        = final_state.get("answer") or "I cannot find that information in the document."
    corrected_q   = final_state.get("question") or question
    relevant_docs = final_state.get("relevant_docs") or []
    doc_scores    = final_state.get("doc_scores") or []
    retries       = final_state.get("retry_count") or 0

    # Build source labels
    sources = [f"chunk_{i+1}" for i in range(len(relevant_docs))]

    print(f"✅ /rag/query complete in {elapsed}s | retries={retries}")

    return QueryResponse(
        answer=answer,
        corrected_question=corrected_q,
        sources=sources,
        retrieval_scores=doc_scores[:len(sources)] if doc_scores else None,
        retries=retries,
        elapsed_seconds=elapsed,
    )


@rag_router.post("/generate-artifact", response_model=GenerateArtifactResponse)
async def rag_generate_artifact(payload: GenerateArtifactRequest):
    """
    AI-Based PRD Platform: generate a structured artifact from the indexed document using RAG + Qwen GGUF.

    Supported ``artifact_type`` values: ``prd``, ``code_docs``, ``flow_diagram``, ``sop``,
    ``code_review``, ``architecture``, ``dependency_graph`` (aliases: ``cdg``, ``flowchart``).
    """
    document_id = (payload.document_id or "").strip()
    if not document_id:
        raise HTTPException(status_code=400, detail="document_id cannot be empty.")

    doc_id = sanitize_rag_id(document_id)

    if doc_id not in _bm25_store:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{doc_id}' not indexed. Ingest sources or POST /rag/project/append first.",
        )

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            partial(
                run_artifact_generation,
                doc_id,
                payload.artifact_type,
                payload.user_brief,
                payload.top_k,
            ),
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"❌ generate-artifact error: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Artifact generation failed: {str(e)}")


@rag_router.delete("/clear/{document_id}", response_model=ClearResponse)
async def rag_clear(document_id: str):
    """
    Clear the BM25 index and ChromaDB collection for a document.
    Call this before re-indexing a new version of a document.
    """
    doc_id = sanitize_rag_id(document_id)
    clear_project_sources(doc_id)

    # Clear BM25
    removed_bm25 = _bm25_store.pop(doc_id, None) is not None

    # Clear ChromaDB
    removed_chroma = False
    try:
        _chroma_client.delete_collection(doc_id)
        removed_chroma = True
    except Exception:
        pass

    if not removed_bm25 and not removed_chroma:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found in index.")

    print(f"🗑️ /rag/clear | Cleared index for '{doc_id}'")
    return ClearResponse(status="cleared", document_id=doc_id)


@rag_router.get("/status/{document_id}")
async def rag_status(document_id: str):
    """Check if a document has been indexed and how many chunks it has."""
    doc_id = sanitize_rag_id(document_id)
    store  = _bm25_store.get(doc_id)

    if not store:
        return {
            "indexed": False,
            "document_id": doc_id,
            "chunks": 0,
            "sources_stacked": len(_project_source_blocks.get(doc_id) or []),
            "jina_embedding_n_gpu_layers": JINA_EMBED_N_GPU,
            "rag_llm_backend": "llama-cpp-python",
            "rag_llm_model": _rag_llm_model_label(),
            "rag_llm_n_ctx": RAG_LLM_N_CTX,
            "rag_llm_n_gpu_layers": RAG_LLM_N_GPU,
        }

    try:
        collection = _chroma_client.get_collection(doc_id)
        chroma_count = collection.count()
    except Exception:
        chroma_count = 0

    return {
        "indexed":       True,
        "document_id":   doc_id,
        "chunks":        len(store["chunks"]),
        "sources_stacked": len(_project_source_blocks.get(doc_id) or []),
        "bm25_ready":    True,
        "vector_ready":  chroma_count > 0,
        "chroma_chunks": chroma_count,
        "embedding_model": EMBEDDING_MODEL_LABEL,
        "jina_embedding_n_gpu_layers": JINA_EMBED_N_GPU,
        "rag_llm_backend":    "llama-cpp-python",
        "rag_llm_model":      _rag_llm_model_label(),
        "rag_llm_n_ctx":      RAG_LLM_N_CTX,
        "rag_llm_n_gpu_layers": RAG_LLM_N_GPU,
    }
