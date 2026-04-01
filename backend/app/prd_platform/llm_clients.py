"""Azure OpenAI or local GGUF for the PRD platform: chat completions vs Codex via Responses API."""

import json
import re
from typing import Any

from openai import AzureOpenAI, OpenAI

from app.core.config import settings
from app.core.llm_routing import llm_provider_is_local
from app.core.local_llm import local_chat_complete as _local_chat_complete

_chat_client: AzureOpenAI | None = None
_codex_responses_client: OpenAI | None = None


def _chat() -> AzureOpenAI:
    global _chat_client
    if _chat_client is None:
        if not settings.AZURE_GPT_KEY or not settings.AZURE_GPT_ENDPOINT:
            raise RuntimeError("AZURE_GPT_KEY / AZURE_GPT_ENDPOINT not configured")
        _chat_client = AzureOpenAI(
            azure_endpoint=settings.AZURE_GPT_ENDPOINT.rstrip("/"),
            api_key=settings.AZURE_GPT_KEY,
            api_version=settings.AZURE_GPT_API_VERSION,
        )
    return _chat_client


def _codex_responses_openai() -> OpenAI:
    """OpenAI-compatible client pointed at Azure /openai/v1/ for Responses API (Codex)."""
    global _codex_responses_client
    if _codex_responses_client is None:
        if not settings.AZURE_GPT_51_KEY or not settings.AZURE_GPT_51_ENDPOINT:
            raise RuntimeError(
                "AZURE_GPT_51_KEY / AZURE_GPT_51_ENDPOINT required for code generation (Codex)"
            )
        if not settings.AZURE_GPT_51_DEPLOYMENT:
            raise RuntimeError("AZURE_GPT_51_DEPLOYMENT not set")
        base_url = f"{settings.AZURE_GPT_51_ENDPOINT.rstrip('/')}/openai/v1/"
        kwargs: dict[str, Any] = {
            "api_key": settings.AZURE_GPT_51_KEY,
            "base_url": base_url,
        }
        ver = (settings.AZURE_GPT_51_RESPONSES_API_VERSION or "").strip()
        if ver:
            kwargs["default_query"] = {"api-version": ver}
        _codex_responses_client = OpenAI(**kwargs)
    return _codex_responses_client


def _response_output_text(resp: Any) -> str:
    """Best-effort text from Responses API result."""
    text = getattr(resp, "output_text", None)
    if text and str(text).strip():
        return str(text).strip()
    parts: list[str] = []
    for item in getattr(resp, "output", None) or []:
        if getattr(item, "type", None) == "message":
            for c in getattr(item, "content", None) or []:
                if getattr(c, "type", None) in ("output_text", "text"):
                    t = getattr(c, "text", None)
                    if t:
                        parts.append(str(t))
    return "\n".join(parts).strip()


def _codex_responses_complete(system: str, user: str, max_tokens: int) -> str:
    client = _codex_responses_openai()
    deployment = settings.AZURE_GPT_51_DEPLOYMENT
    resp = client.responses.create(
        model=deployment,
        instructions=system,
        input=user,
        max_output_tokens=max_tokens,
        store=False,
    )
    return _response_output_text(resp)


def chat_complete(use_codex: bool, system: str, user: str, max_tokens: int = 8192) -> str:
    """
    LLM_PROVIDER=local: Qwen GGUF via llama-cpp-python (both code and non-code paths).
    LLM_PROVIDER=azure: Azure chat completions; use_codex=True uses Responses API (Codex).
    """
    if llm_provider_is_local():
        return _local_chat_complete(system, user, max_tokens=max_tokens)

    if use_codex:
        return _codex_responses_complete(system, user, max_tokens)

    client = _chat()
    deployment = settings.AZURE_GPT_DEPLOYMENT
    if not deployment:
        raise RuntimeError("Azure deployment name not configured (AZURE_GPT_DEPLOYMENT)")

    resp = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_completion_tokens=max_tokens,
    )
    choice = resp.choices[0].message.content
    return (choice or "").strip()


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}\s*$", text)
    if not m:
        m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None
