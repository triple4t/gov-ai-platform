"""
LLM-backed code summarization with short / medium / detailed levels.
Uses Codex path for code-heavy prompts (same stack as PRD platform).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from typing import Any, Literal

from app.core.config import settings
from app.core.llm_routing import llm_provider_is_local
from app.prd_platform.llm_clients import chat_complete

logger = logging.getLogger(__name__)

DetailLevel = Literal["short", "medium", "detailed"]
USE_CODEX = True

_LEVEL_INSTRUCTIONS: dict[str, str] = {
    "short": (
        "Length: 2–4 bullet lines or about 80–120 words total. "
        "No preamble; state only facts about the code."
    ),
    "medium": (
        "Length: several short sections (Purpose, Key behavior, Dependencies). "
        "About 200–350 words. Use plain text; optional simple bullets."
    ),
    "detailed": (
        "Length: thorough but still scannable—structure, responsibilities, "
        "important branches, error handling, extension points, and notable dependencies. "
        "Up to ~600 words unless the snippet is tiny."
    ),
}

_SYSTEM_BASE = """You are a senior software engineer explaining code to another engineer.
Structure the answer with markdown: use ## SectionName for each section requested in the task (exact heading titles).
Use bullet lists under sections when helpful. Do not wrap the whole answer in a fenced code block.
Be accurate; if something is unclear from the snippet or excerpts, say so briefly."""


_cache_lock = threading.Lock()
_cache_store: dict[str, tuple[float, dict[str, Any]]] = {}


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + "\n\n[…truncated for model context…]", True


def _cache_get(key: str) -> dict[str, Any] | None:
    if not settings.CODE_SUMMARY_CACHE_ENABLED:
        return None
    now = time.time()
    with _cache_lock:
        item = _cache_store.get(key)
        if not item:
            return None
        exp, payload = item
        if exp < now:
            del _cache_store[key]
            return None
        return payload


def _cache_set(key: str, payload: dict[str, Any]) -> None:
    if not settings.CODE_SUMMARY_CACHE_ENABLED:
        return
    ttl = max(30, settings.CODE_SUMMARY_CACHE_TTL_SECONDS)
    with _cache_lock:
        _cache_store[key] = (time.time() + ttl, payload)
        # bound size
        if len(_cache_store) > 512:
            oldest = sorted(_cache_store.items(), key=lambda x: x[1][0])[:128]
            for k, _ in oldest:
                _cache_store.pop(k, None)


def _cache_key(endpoint: str, payload: dict[str, Any]) -> str:
    """Stable key; large `body` fields are hashed instead of embedded verbatim."""
    p = dict(payload)
    body = p.get("body")
    if isinstance(body, str) and len(body) > 2048:
        p["body"] = hashlib.sha256(body.encode("utf-8", errors="replace")).hexdigest()
    raw = json.dumps({"e": endpoint, "p": p}, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def summarize_with_llm(
    *,
    endpoint: str,
    user_task: str,
    code_or_context: str,
    level: DetailLevel,
    extra_cache_payload: dict[str, Any] | None = None,
) -> str:
    """Run one LLM completion; optional in-process cache."""
    max_in = settings.CODE_SUMMARIZER_MAX_INPUT_CHARS
    if llm_provider_is_local():
        # llama-cpp rejects when the chat-formatted prompt exceeds LLAMA_N_CTX (before max_tokens is applied).
        reserve = max(512, settings.CODE_SUMMARIZER_LOCAL_CTX_RESERVE_TOKENS)
        avail_tok = max(512, settings.LLAMA_N_CTX - reserve)
        cap = int(avail_tok * settings.CODE_SUMMARIZER_LOCAL_CHARS_PER_TOKEN)
        max_in = min(max_in, max(4096, cap))
    body, truncated = _truncate(code_or_context.strip(), max_in)
    cache_payload = {"level": level, "task": user_task, "body": body, **(extra_cache_payload or {})}
    key = _cache_key(endpoint, cache_payload)
    hit = _cache_get(key)
    if hit and "summary" in hit:
        return str(hit["summary"])

    sys_p = f"{_SYSTEM_BASE}\n\n{_LEVEL_INSTRUCTIONS.get(level, _LEVEL_INSTRUCTIONS['medium'])}"
    user_p = f"{user_task}\n\n---\nCode / context:\n{body}\n"
    if truncated:
        user_p += "\n(Note: input was truncated at server limit.)\n"

    try:
        max_out = 4096
        if llm_provider_is_local():
            max_out = min(4096, max(512, settings.LLAMA_N_CTX // 2))
        text = chat_complete(USE_CODEX, sys_p, user_p, max_tokens=max_out)
    except Exception as e:
        logger.exception("Code summarizer LLM call failed: %s", e)
        raise

    out = (text or "").strip()
    _cache_set(key, {"summary": out})
    return out


def summarize_file(
    content: str,
    *,
    language: str,
    file_path: str,
    level: DetailLevel,
    outline: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    task = (
        f"Summarize this {language} source file ({file_path or 'unknown path'}).\n"
        f"Structural outline (may be incomplete):\n{outline}\n\n"
        "Use these sections in order (markdown ## headings):\n"
        "## Purpose\n"
        "## Key functions and classes\n"
        "## Logic and behavior\n"
    )
    summary = summarize_with_llm(
        endpoint="file",
        user_task=task,
        code_or_context=content,
        level=level,
        extra_cache_payload={"language": language, "file": file_path},
    )
    return {"summary": summary, "level": level, "metadata": metadata}


def summarize_function_snippet(
    snippet: str,
    *,
    language: str,
    symbol_name: str | None,
    level: DetailLevel,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    name = symbol_name or "(anonymous or unknown)"
    task = (
        f"Explain this {language} function or class snippet. Symbol hint: {name}.\n\n"
        "Use these sections in order (markdown ## headings):\n"
        "## What it does\n"
        "## Inputs and outputs\n"
        "## Notes\n"
    )
    summary = summarize_with_llm(
        endpoint="function",
        user_task=task,
        code_or_context=snippet,
        level=level,
        extra_cache_payload={"language": language, "symbol": name},
    )
    return {"summary": summary, "level": level, "metadata": metadata}


def summarize_module(
    combined: str,
    *,
    paths: list[str],
    level: DetailLevel,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    task = (
        "Summarize this module (related source files). "
        f"Files: {', '.join(paths) if paths else 'unknown'}.\n\n"
        "Use these sections in order (markdown ## headings):\n"
        "## Module responsibility\n"
        "## How files interact\n"
        "## Notable dependencies or risks\n"
    )
    summary = summarize_with_llm(
        endpoint="module",
        user_task=task,
        code_or_context=combined,
        level=level,
        extra_cache_payload={"paths": paths},
    )
    return {"summary": summary, "level": level, "metadata": metadata}


def summarize_project_from_context(
    context: str,
    *,
    project_id: str,
    level: DetailLevel,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    task = (
        f"Using ONLY the repository excerpts below, write a project-level summary for id `{project_id}`. "
        "Do not invent files or frameworks not evidenced in the excerpts.\n\n"
        "Use these sections in order (markdown ## headings). If the excerpts lack evidence for a section, state that briefly.\n"
        "## Purpose\n"
        "## Tech stack\n"
        "## Architecture\n"
        "## Key modules\n"
        "## Main flows\n"
    )
    summary = summarize_with_llm(
        endpoint="project",
        user_task=task,
        code_or_context=context,
        level=level,
        extra_cache_payload={"project_id": project_id},
    )
    return {"summary": summary, "level": level, "metadata": metadata}


def answer_with_rag(
    question: str,
    context: str,
    *,
    project_id: str,
    level: DetailLevel,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    task = (
        f"Answer the user's question using ONLY the repository excerpts below. "
        f"If the excerpts do not contain enough information, say so and answer what you can.\n"
        f"Question: {question}\n"
        f"Project id: {project_id}\n\n"
        "Structure (markdown ## headings):\n"
        "## Answer\n"
        "(Direct, concise response first.)\n"
        "## Details\n"
        "(Optional: bullet points with file paths from the excerpts as citations.)\n"
    )
    summary = summarize_with_llm(
        endpoint="query",
        user_task=task,
        code_or_context=context or "(no retrieved context)",
        level=level,
        extra_cache_payload={"project_id": project_id, "q": question[:500]},
    )
    return {"summary": summary, "level": level, "metadata": metadata}
