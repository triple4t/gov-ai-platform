"""
Hybrid RAG (BM25 + Chroma + Jina GGUF embeddings + LangGraph + local Qwen GGUF).

Canonical implementation: this package (`rag_engine.py`). The nested folder
`backend/backend/rag_engine.py` is a legacy copy; new work belongs here.
Mounted at ``/api/v1/rag/*`` and aliased at ``/api/v1/prd-platform/rag/*`` for PRD UI.
"""

from app.advanced_rag.rag_engine import rag_router

__all__ = ["rag_router"]
