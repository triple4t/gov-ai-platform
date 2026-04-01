"""Embeddings for PRD platform RAG: Azure OpenAI or local GGUF (llama-cpp-python)."""

import logging
from typing import Sequence

from openai import AzureOpenAI

from app.core.config import settings
from app.core.llm_routing import llm_provider_is_local

logger = logging.getLogger(__name__)

_client: AzureOpenAI | None = None


def prd_embedding_configuration_error() -> str | None:
    """
    If PRD ZIP indexing cannot run, return a user-facing message; otherwise None.
    Call before ingest/embed to avoid opaque 500s when .env is incomplete.
    """
    if llm_provider_is_local():
        from app.core.local_llm import local_embedding_gguf_configured

        if not local_embedding_gguf_configured():
            return (
                "LLM_PROVIDER=local requires LOCAL_EMBEDDING_GGUF_PATH to an existing embedding .gguf file "
                "(text / retrieval model). Add it to .env and restart. "
                "Alternatively set LLM_PROVIDER=azure and configure AZURE_OPENAI_EMBEDDING_API_KEY, "
                "AZURE_OPENAI_EMBEDDING_ENDPOINT, and AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME."
            )
        return None
    if (
        not settings.AZURE_OPENAI_EMBEDDING_API_KEY
        or not settings.AZURE_OPENAI_EMBEDDING_ENDPOINT
        or not settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    ):
        return (
            "When LLM_PROVIDER is not local, PRD indexing needs Azure embeddings: set "
            "AZURE_OPENAI_EMBEDDING_API_KEY, AZURE_OPENAI_EMBEDDING_ENDPOINT, and "
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME."
        )
    return None


def _use_local_embeddings() -> bool:
    return llm_provider_is_local()


def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        if not settings.AZURE_OPENAI_EMBEDDING_API_KEY or not settings.AZURE_OPENAI_EMBEDDING_ENDPOINT:
            raise RuntimeError("Azure embedding credentials not configured (AZURE_OPENAI_EMBEDDING_*)")
        if not settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT:
            raise RuntimeError("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME not set")
        _client = AzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_EMBEDDING_ENDPOINT,
            api_key=settings.AZURE_OPENAI_EMBEDDING_API_KEY,
            api_version=settings.AZURE_OPENAI_EMBEDDING_API_VERSION,
        )
    return _client


def embed_texts(texts: Sequence[str], batch_size: int = 16) -> tuple[list[list[float]], int]:
    """
    Returns (list of embedding vectors, dimension).
    """
    if _use_local_embeddings():
        from app.core.local_llm import local_embed_texts

        # Index chunks: Passage: prefix for Jina text-retrieval GGUF (matches Advanced RAG).
        return local_embed_texts(texts, batch_size=batch_size, for_query=False)

    client = _get_client()
    deployment = settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    all_vecs: list[list[float]] = []
    dim = settings.PRD_EMBEDDING_DIM_DEFAULT

    batch: list[str] = []
    for t in texts:
        batch.append(t[:8000] if t else " ")
        if len(batch) >= batch_size:
            resp = client.embeddings.create(model=deployment, input=batch)
            for item in sorted(resp.data, key=lambda x: x.index):
                v = list(item.embedding)
                all_vecs.append(v)
                dim = len(v)
            batch = []
    if batch:
        resp = client.embeddings.create(model=deployment, input=batch)
        for item in sorted(resp.data, key=lambda x: x.index):
            v = list(item.embedding)
            all_vecs.append(v)
            dim = len(v)

    if len(all_vecs) != len(texts):
        logger.warning("Embedding count mismatch: %s vs %s", len(all_vecs), len(texts))
    return all_vecs, dim


def embed_query(text: str) -> tuple[list[float], int]:
    if _use_local_embeddings():
        from app.core.local_llm import local_embed_texts

        vecs, dim = local_embed_texts([text], batch_size=1, for_query=True)
        return vecs[0], dim
    vecs, dim = embed_texts([text], batch_size=1)
    return vecs[0], dim
