"""Whether LLM calls should use local GGUF only (no Azure / OpenAI for chat or embeddings)."""

from app.core.config import settings


def llm_provider_is_local() -> bool:
    return settings.LLM_PROVIDER.strip().lower() == "local"
