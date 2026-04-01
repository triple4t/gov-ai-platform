"""
Backend entrypoint — PRD Platform + Code Summarizer mode.

Other routers (voice form, OCR, conversational AI, Advanced RAG) are commented out below.
Un-comment imports, `include_router`, and `startup_event` blocks to restore the full unified app.

PRD stack still uses: `settings`, `app.core.local_llm`, PRD DB, `UPLOAD_DIR`.
"""

import asyncio
import logging
import os
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*protected namespace \"model_\".*",
    category=UserWarning,
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- Commented: non-PRD API surface ---
# from app.api.routes import router
from app.core.config import settings

# from app.core.mlflow_util import mlflow_server_reachable
# from app.db import init_db
# import mlflow
# from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# from app.conversational.database import init_conversational_db
# from app.conversational.auth import router as auth_router
# from app.conversational.chat import router as chat_router
# from app.conversational.voice import router as voice_router
# from app.conversational.documents import router as documents_router
# from app.conversational.integrations import router as integrations_router

from app.prd_platform.database import cleanup_expired_projects, init_prd_db
from app.prd_platform.routes import router as prd_platform_router
from app.code_summarizer.routes import router as code_summarizer_router

# from app.services.prompt_registry import setup_prompt_registry
from app.core.llm_routing import llm_provider_is_local
from app.core.local_llm import (
    get_chat_llama,
    get_embed_llama,
    local_chat_gguf_configured,
    local_embedding_gguf_configured,
)

# from app.core.vlm_ocr import get_vlm_ocr_llama, vlm_ocr_chat_handler_mode, vlm_ocr_configured

ADVANCED_RAG_AVAILABLE = False
ADVANCED_RAG_IMPORT_ERROR: str | None = None
advanced_rag_router = None
try:
    from app.advanced_rag import rag_router as advanced_rag_router

    ADVANCED_RAG_AVAILABLE = True
except ImportError as e:
    ADVANCED_RAG_IMPORT_ERROR = str(e)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PRD Platform, Code Summarizer & Hybrid RAG API",
    description="PRD ZIP RAG, code summarization, and optional BM25+Chroma+LangGraph hybrid RAG (/api/v1/rag and /api/v1/prd-platform/rag).",
    version="1.0.0-prd",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Active: PRD + code summarizer + hybrid RAG (from app/advanced_rag; legacy copy: backend/backend/rag_engine.py) ---
app.include_router(prd_platform_router, prefix="/api/v1")
app.include_router(code_summarizer_router, prefix="/api/v1")
if ADVANCED_RAG_AVAILABLE and advanced_rag_router is not None:
    app.include_router(advanced_rag_router, prefix="/api/v1")
    # Same routes under PRD namespace for the PRD UI service switcher
    app.include_router(advanced_rag_router, prefix="/api/v1/prd-platform")
else:
    logger.warning(
        "Hybrid RAG not loaded. pip install rank-bm25 chromadb langgraph langchain-text-splitters. %s",
        ADVANCED_RAG_IMPORT_ERROR or "",
    )

# --- Commented: restore for full unified backend ---
# app.include_router(router)
# app.include_router(auth_router, prefix="/api/v1/auth", tags=["Conversational AI - Auth"])
# app.include_router(chat_router, prefix="/api/v1/chat", tags=["Conversational AI - Chat"])
# app.include_router(voice_router, prefix="/api/v1/voice", tags=["Conversational AI - Voice"])
# app.include_router(documents_router, prefix="/api/v1/documents", tags=["Conversational AI - Documents"])
# app.include_router(integrations_router, prefix="/api/v1/integrations", tags=["Conversational AI - Integrations"])
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    rag = {
        "rag_ingest": "POST /api/v1/prd-platform/rag/ingest",
        "rag_ingest_file": "POST /api/v1/prd-platform/rag/ingest-file (multipart: file, document_id, optional content_kind)",
        "rag_query": "POST /api/v1/prd-platform/rag/query",
        "rag_status": "GET /api/v1/prd-platform/rag/status/{document_id}",
        "rag_generate_artifact": "POST /api/v1/prd-platform/rag/generate-artifact",
    }
    if not ADVANCED_RAG_AVAILABLE:
        rag = {"_note": "Hybrid RAG dependencies missing; install rank-bm25 chromadb langgraph langchain-text-splitters"}
    return {
        "service": "PRD Platform, Code Summarizer & Hybrid RAG (lean mode)",
        "version": "1.0.0-prd",
        "docs": "/docs",
        "endpoints": {
            "prd_capabilities": "GET /api/v1/prd-platform/capabilities",
            "prd_upload": "POST /api/v1/prd-platform/projects/upload",
            "prd_generate": "POST /api/v1/prd-platform/projects/generate",
            "code_summarize_file": "POST /api/v1/code-summarize/file",
            "code_summarize_function": "POST /api/v1/code-summarize/function",
            "code_summarize_module": "POST /api/v1/code-summarize/module",
            "code_summarize_project": "POST /api/v1/code-summarize/project",
            "code_summarize_query": "POST /api/v1/code-summarize/query",
            **rag,
        },
    }


@app.on_event("startup")
async def startup_event():
    # --- PRD + shared filesystem (required) ---
    init_prd_db()
    cleanup_expired_projects()
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    # --- Commented: other subsystems ---
    # init_db()
    # await init_conversational_db()

    # try:
    #     FastAPIInstrumentor.instrument_app(app)
    # except Exception as e:
    #     logger.warning("OpenTelemetry FastAPI instrumentation skipped: %s", e)

    # if not settings.MLFLOW_ENABLED:
    #     logger.info("MLflow disabled (MLFLOW_ENABLED=false).")
    # elif not mlflow_server_reachable(settings.MLFLOW_TRACKING_URI):
    #     logger.warning(...)
    # else:
    #     try:
    #         ...
    #         setup_prompt_registry()
    #     except Exception as e:
    #         logger.error("Failed to initialize MLflow: %s", e)

    logger.info("=" * 60)
    logger.info("  PRD Platform + Code Summarizer + Hybrid RAG (lean backend)")
    logger.info(f"  LLM provider:      {settings.LLM_PROVIDER}")
    if llm_provider_is_local():
        logger.info(
            "  Local chat GGUF:   %s",
            settings.LOCAL_CHAT_GGUF_PATH
            if local_chat_gguf_configured()
            else f"MISSING ({settings.LOCAL_CHAT_GGUF_PATH or 'not set'})",
        )
        logger.info(
            "  Local embed GGUF:  %s",
            settings.LOCAL_EMBEDDING_GGUF_PATH
            if local_embedding_gguf_configured()
            else f"MISSING ({settings.LOCAL_EMBEDDING_GGUF_PATH or 'not set'})",
        )
        logger.info(
            "  llama.cpp GPU:     chat n_gpu_layers=%s, embed=%s, main_gpu=%s",
            settings.LLAMA_N_GPU_LAYERS,
            settings.LLAMA_N_GPU_LAYERS_EMBED,
            settings.LLAMA_MAIN_GPU,
        )
        try:
            from llama_cpp import llama_supports_gpu_offload

            if not llama_supports_gpu_offload():
                logger.warning(
                    "  llama.cpp build has NO GPU offload — local models may run on CPU only."
                )
        except Exception:
            pass

    # logger.info(f"  Deepgram Key:      {'Configured' if settings.DEEPGRAM_API_KEY else 'Missing'}")
    # logger.info(f"  Azure GPT:         ...")
    # logger.info(f"  OpenAI (OCR):      ...")
    # if vlm_ocr_configured(): ...
    if ADVANCED_RAG_AVAILABLE:
        logger.info(
            "  Hybrid RAG:        enabled → /api/v1/rag/* and /api/v1/prd-platform/rag/*"
        )
    else:
        logger.info(
            "  Hybrid RAG:        disabled — %s",
            (ADVANCED_RAG_IMPORT_ERROR or "missing dependencies")[:160],
        )

    logger.info(f"  CORS Origins:      {settings.CORS_ORIGINS}")
    raw_neo4j_enabled = os.getenv("NEO4J_ENABLED", "").strip()
    if raw_neo4j_enabled and (
        "NEO4J_URI" in raw_neo4j_enabled
        or (
            "=" in raw_neo4j_enabled
            and raw_neo4j_enabled.lower() not in ("1", "true", "yes", "on")
        )
    ):
        logger.warning(
            "NEO4J_ENABLED looks malformed (merged lines in .env?). "
            "Use one KEY=value per line. Value preview: %s",
            raw_neo4j_enabled[:120] + ("..." if len(raw_neo4j_enabled) > 120 else ""),
        )
    if settings.NEO4J_ENABLED:
        neo4j_db = settings.NEO4J_DATABASE or "(default)"
        logger.info(
            "  Neo4j CDG sync:    enabled → %s (database=%s)",
            settings.NEO4J_URI,
            neo4j_db,
        )
    else:
        logger.info(
            "  Neo4j CDG sync:    disabled — set NEO4J_ENABLED=true in .env to write graphs on ZIP upload",
        )

    if llm_provider_is_local() and settings.LOCAL_LLM_WARMUP_ON_STARTUP:
        if local_chat_gguf_configured():
            logger.info(
                "Local LLM warmup: loading chat GGUF now (set LOCAL_LLM_WARMUP_ON_STARTUP=false to defer)."
            )
            await asyncio.to_thread(get_chat_llama)
            logger.info("Local LLM warmup: chat model ready.")
        if settings.LOCAL_LLM_WARMUP_EMBED_ON_STARTUP and local_embedding_gguf_configured():
            logger.info("Local LLM warmup: loading embedding GGUF…")
            await asyncio.to_thread(get_embed_llama)
            logger.info("Local LLM warmup: embedding model ready.")
        # if settings.LOCAL_LLM_WARMUP_OCR_VLM_ON_STARTUP and vlm_ocr_configured():
        #     ...

    logger.info("=" * 60)
