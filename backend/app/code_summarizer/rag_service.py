"""
Thin RAG adapter: reuse PRD FAISS + chunks for code-summarizer endpoints.
"""

from __future__ import annotations

from app.core.config import settings
from app.prd_platform.database import get_project
from app.prd_platform.rag_service import retrieve_context, retrieve_deep_codebase_context


def assert_project_ready(project_id: str) -> None:
    """Raise ValueError with message if project missing or not indexed."""
    proj = get_project(project_id)
    if not proj:
        raise ValueError("not_found")
    if proj.get("status") != "ready":
        raise ValueError("not_ready")


def retrieve_for_query(project_id: str, question: str, top_k: int | None = None) -> str:
    """Vector search + packed file blocks for a user question."""
    assert_project_ready(project_id)
    k = top_k if top_k is not None else settings.CODE_SUMMARIZER_RAG_TOP_K
    max_c = settings.CODE_SUMMARIZER_RAG_MAX_CHARS
    return retrieve_context(project_id, question, top_k=k, max_chars=max_c)


def retrieve_for_project_summary(project_id: str) -> str:
    """Broad codebase context for high-level project summary."""
    assert_project_ready(project_id)
    return retrieve_deep_codebase_context(
        project_id,
        capability_title="Code summarization",
        answers_block="Produce a high-level project summary: purpose, main modules, entry points, and how pieces fit together.",
    )
