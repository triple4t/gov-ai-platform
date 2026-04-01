"""FastAPI routes: /api/v1/prd-platform — unauthenticated."""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.core.config import settings
from app.prd_platform.clarifying import CAPABILITIES, get_capability
from app.prd_platform.database import (
    clear_chunks,
    delete_project,
    get_project,
    insert_chunks,
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
