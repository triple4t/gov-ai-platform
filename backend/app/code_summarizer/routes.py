"""
FastAPI routes: /api/v1/code-summarize/*

Document summarization remains at POST /api/v1/summarize — do not overlap.
"""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from app.core.config import settings
from app.code_summarizer import rag_service as cs_rag
from app.code_summarizer.summarizer_service import (
    answer_with_rag,
    summarize_file,
    summarize_function_snippet,
    summarize_module,
    summarize_project_from_context,
)
from app.code_summarizer.utils.code_parser import parse_source
from app.code_summarizer.utils.chunker import symbol_chunks

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/code-summarize", tags=["Code Summarizer"])

DetailLevel = Literal["short", "medium", "detailed"]

_ALLOWED_FILE_LANGS = frozenset({"python", "javascript", "typescript", "unknown"})


class SummaryMetadata(BaseModel):
    """Extensible per-response metadata."""

    model_config = {"extra": "allow"}

    language: str | None = None
    file: str | None = None
    files: list[str] | None = None
    functions: list[str] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=list)
    imports: list[str] = Field(default_factory=list)
    truncated: bool = False
    project_id: str | None = None


class CodeSummaryResponse(BaseModel):
    summary: str
    level: str
    metadata: SummaryMetadata


class FileSummarizeRequest(BaseModel):
    content: str = Field(..., description="Raw file source")
    language: str = Field("python", description="python | javascript | typescript | js | ts | ...")
    file_path: str = Field("", description="Optional path for context")
    detail_level: DetailLevel = "medium"

    @field_validator("content")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("content is required")
        return v


class FunctionSummarizeRequest(BaseModel):
    code: str = Field(..., description="Function or class snippet")
    language: str = "python"
    symbol_name: str | None = None
    detail_level: DetailLevel = "medium"

    @field_validator("code")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("code is required")
        return v


class ModuleFilePart(BaseModel):
    path: str
    content: str


class ModuleSummarizeRequest(BaseModel):
    files: list[ModuleFilePart] = Field(..., min_length=1)
    detail_level: DetailLevel = "medium"

    @field_validator("files")
    @classmethod
    def non_empty_content(cls, v: list[ModuleFilePart]) -> list[ModuleFilePart]:
        for f in v:
            if not f.content.strip():
                raise ValueError(f"Empty content for path {f.path!r}")
        return v


class ProjectSummarizeRequest(BaseModel):
    project_id: str = Field(..., min_length=1)
    detail_level: DetailLevel = "medium"


class QuerySummarizeRequest(BaseModel):
    project_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(None, ge=1, le=64)
    detail_level: DetailLevel = "medium"


def _check_body_size(text: str) -> None:
    if len(text) > settings.CODE_SUMMARIZER_HARD_MAX_CHARS:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Payload exceeds {settings.CODE_SUMMARIZER_HARD_MAX_CHARS} characters",
        )


def _normalize_lang(language: str) -> str:
    l = (language or "unknown").strip().lower()
    mapping = {
        "py": "python",
        "js": "javascript",
        "jsx": "javascript",
        "mjs": "javascript",
        "cjs": "javascript",
        "ts": "typescript",
        "tsx": "typescript",
    }
    return mapping.get(l, l)


@router.post("/file", response_model=CodeSummaryResponse)
async def summarize_file_endpoint(body: FileSummarizeRequest):
    _check_body_size(body.content)
    lang = _normalize_lang(body.language)
    if lang not in _ALLOWED_FILE_LANGS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported language; use python or javascript / TypeScript family.",
        )

    parse_lang = lang if lang != "unknown" else body.language
    parsed = parse_source(body.content, parse_lang)
    chunks = symbol_chunks(body.file_path or "file", body.content, parse_lang)
    outline_lines = [f"- {c['kind']} {c['name']} (L{c['start_line']}-L{c['end_line']})" for c in chunks[:40]]
    outline = "\n".join(outline_lines) if outline_lines else "(no symbols extracted)"
    truncated = len(body.content) > settings.CODE_SUMMARIZER_MAX_INPUT_CHARS

    meta = SummaryMetadata(
        language=lang or parsed.language,
        file=body.file_path or None,
        functions=parsed.functions,
        classes=parsed.classes,
        imports=parsed.imports,
        truncated=truncated,
    )
    try:
        result = summarize_file(
            body.content,
            language=lang,
            file_path=body.file_path or "file",
            level=body.detail_level,
            outline=outline,
            metadata=meta.model_dump(exclude_none=True),
        )
    except RuntimeError as e:
        logger.warning("LLM/embeddings configuration error: %s", e)
        raise HTTPException(status_code=503, detail="Summarization service unavailable") from e
    except Exception as e:
        logger.exception("summarize_file: %s", e)
        raise HTTPException(status_code=500, detail="Summarization failed") from e

    return CodeSummaryResponse(
        summary=result["summary"],
        level=result["level"],
        metadata=SummaryMetadata(**{**meta.model_dump(), "truncated": truncated}),
    )


@router.post("/function", response_model=CodeSummaryResponse)
async def summarize_function_endpoint(body: FunctionSummarizeRequest):
    _check_body_size(body.code)
    lang = _normalize_lang(body.language)
    parsed = parse_source(body.code, lang if lang != "unknown" else body.language)
    meta = SummaryMetadata(
        language=lang or parsed.language,
        functions=parsed.functions or ([body.symbol_name] if body.symbol_name else []),
        classes=parsed.classes,
        imports=parsed.imports,
        truncated=len(body.code) > settings.CODE_SUMMARIZER_MAX_INPUT_CHARS,
    )
    try:
        result = summarize_function_snippet(
            body.code,
            language=lang,
            symbol_name=body.symbol_name,
            level=body.detail_level,
            metadata=meta.model_dump(exclude_none=True),
        )
    except RuntimeError as e:
        logger.warning("LLM configuration error: %s", e)
        raise HTTPException(status_code=503, detail="Summarization service unavailable") from e
    except Exception as e:
        logger.exception("summarize_function: %s", e)
        raise HTTPException(status_code=500, detail="Summarization failed") from e

    return CodeSummaryResponse(summary=result["summary"], level=result["level"], metadata=meta)


@router.post("/module", response_model=CodeSummaryResponse)
async def summarize_module_endpoint(body: ModuleSummarizeRequest):
    parts: list[str] = []
    paths: list[str] = []
    total = 0
    for f in body.files:
        _check_body_size(f.content)
        total += len(f.content)
        paths.append(f.path)
        parts.append(f"### {f.path}\n{f.content}\n")
    if total > settings.CODE_SUMMARIZER_HARD_MAX_CHARS:
        raise HTTPException(status_code=413, detail="Combined module sources exceed server limit")

    combined = "\n".join(parts)
    truncated = total > settings.CODE_SUMMARIZER_MAX_INPUT_CHARS
    meta = SummaryMetadata(
        files=paths,
        truncated=truncated,
    )
    try:
        result = summarize_module(
            combined,
            paths=paths,
            level=body.detail_level,
            metadata=meta.model_dump(exclude_none=True),
        )
    except RuntimeError as e:
        logger.warning("LLM configuration error: %s", e)
        raise HTTPException(status_code=503, detail="Summarization service unavailable") from e
    except Exception as e:
        logger.exception("summarize_module: %s", e)
        raise HTTPException(status_code=500, detail="Summarization failed") from e

    return CodeSummaryResponse(summary=result["summary"], level=result["level"], metadata=meta)


@router.post("/project", response_model=CodeSummaryResponse)
async def summarize_project_endpoint(body: ProjectSummarizeRequest):
    try:
        context = cs_rag.retrieve_for_project_summary(body.project_id)
    except ValueError as e:
        msg = str(e)
        if msg == "not_found":
            raise HTTPException(status_code=404, detail="Project not found") from e
        if msg == "not_ready":
            raise HTTPException(
                status_code=400,
                detail="Project exists but is not indexed yet. Upload a ZIP via PRD Platform first.",
            ) from e
        raise HTTPException(status_code=400, detail="Invalid project") from e
    except RuntimeError as e:
        logger.warning("RAG/embeddings error: %s", e)
        raise HTTPException(status_code=503, detail="RAG service unavailable") from e

    if not context.strip():
        raise HTTPException(
            status_code=400,
            detail="No retrievable context for this project (empty index or not ready).",
        )

    meta = SummaryMetadata(
        project_id=body.project_id,
        truncated=len(context) > settings.CODE_SUMMARIZER_MAX_INPUT_CHARS,
    )
    try:
        result = summarize_project_from_context(
            context,
            project_id=body.project_id,
            level=body.detail_level,
            metadata=meta.model_dump(exclude_none=True),
        )
    except RuntimeError as e:
        logger.warning("LLM configuration error: %s", e)
        raise HTTPException(status_code=503, detail="Summarization service unavailable") from e
    except Exception as e:
        logger.exception("summarize_project: %s", e)
        raise HTTPException(status_code=500, detail="Summarization failed") from e

    return CodeSummaryResponse(summary=result["summary"], level=result["level"], metadata=meta)


@router.post("/query", response_model=CodeSummaryResponse)
async def query_codebase_endpoint(body: QuerySummarizeRequest):
    try:
        context = cs_rag.retrieve_for_query(body.project_id, body.question, top_k=body.top_k)
    except ValueError as e:
        msg = str(e)
        if msg == "not_found":
            raise HTTPException(status_code=404, detail="Project not found") from e
        if msg == "not_ready":
            raise HTTPException(
                status_code=400,
                detail="Project not indexed. Upload via PRD Platform first.",
            ) from e
        raise HTTPException(status_code=400, detail="Invalid project") from e
    except RuntimeError as e:
        logger.warning("RAG/embeddings error: %s", e)
        raise HTTPException(status_code=503, detail="RAG service unavailable") from e

    meta = SummaryMetadata(
        project_id=body.project_id,
        truncated=len(context) > settings.CODE_SUMMARIZER_MAX_INPUT_CHARS,
    )
    try:
        result = answer_with_rag(
            body.question,
            context,
            project_id=body.project_id,
            level=body.detail_level,
            metadata=meta.model_dump(exclude_none=True),
        )
    except RuntimeError as e:
        logger.warning("LLM configuration error: %s", e)
        raise HTTPException(status_code=503, detail="Summarization service unavailable") from e
    except Exception as e:
        logger.exception("query: %s", e)
        raise HTTPException(status_code=500, detail="Query failed") from e

    return CodeSummaryResponse(summary=result["summary"], level=result["level"], metadata=meta)
