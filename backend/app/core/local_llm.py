"""
Local GGUF inference via llama-cpp-python (chat + embeddings).
Used when settings.LLM_PROVIDER == \"local\".
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Sequence

from app.core.config import settings

logger = logging.getLogger(__name__)

_chat_llama: Any | None = None
_embed_llama: Any | None = None
_embed_lock = threading.Lock()
# Serialize all llama.cpp chat completions (PRD, RAG, etc.) — Llama instances are not thread-safe.
_chat_completion_lock = threading.Lock()

# Git LFS pointer files are tiny text; real GGUF for multi-B models are much larger.
_MIN_GGUF_BYTES = 8 * 1024 * 1024


def _validate_gguf_file(path: str, *, kind: str) -> None:
    norm = os.path.normcase(path)
    if f"{os.sep}.no_exist{os.sep}" in norm or "/.no_exist/" in path.replace("\\", "/"):
        logger.warning(
            "GGUF path is under Hugging Face '.no_exist' (often incomplete download). "
            "Re-download: hf download <repo> <file.gguf> (venv: .\\venv\\Scripts\\hf.exe) — point LOCAL_*_GGUF_PATH at snapshots/…"
        )
    try:
        size = os.path.getsize(path)
    except OSError as e:
        raise RuntimeError(f"Cannot read GGUF file {path!r}: {e}") from e
    if size < _MIN_GGUF_BYTES:
        raise RuntimeError(
            f"{kind} GGUF at {path!r} is only {size} bytes — likely a failed LFS download or pointer file. "
            "Remove the bad cache folder and run: hf download <org>/<repo> <name>.gguf"
        )
    try:
        with open(path, "rb") as f:
            head = f.read(120)
    except OSError as e:
        raise RuntimeError(f"Cannot read start of GGUF {path!r}: {e}") from e
    if head.startswith(b"version https://git-lfs.github.com/spec"):
        raise RuntimeError(
            f"{path!r} is a Git LFS pointer text file, not model weights. "
            "Run: hf download … <file>.gguf and use the real .gguf path under snapshots/."
        )


def _load_llama(**kw: Any) -> Any:
    from llama_cpp import Llama

    try:
        return Llama(**kw)
    except (ValueError, OSError) as e:
        msg = str(e)
        hint = (
            "If the file is valid on disk: (1) upgrade llama-cpp-python / llama.cpp build (Qwen3.x needs a recent release); "
            "(2) try LLAMA_N_GPU_LAYERS=0 for CPU-only load; (3) avoid paths under hub '.no_exist'. "
            "Set LLAMA_VERBOSE=true for loader details."
        )
        raise RuntimeError(f"Failed to load GGUF: {msg}. {hint}") from e


def _llama_perf_kw() -> dict[str, int]:
    """n_threads and n_batch for chat/embed Llama instances."""
    out: dict[str, int] = {}
    nthr = settings.LLAMA_THREADS
    if nthr > 0:
        out["n_threads"] = nthr
    else:
        cpu = os.cpu_count() or 8
        out["n_threads"] = max(1, min(32, cpu))
    nb = settings.LLAMA_N_BATCH
    if nb > 0:
        out["n_batch"] = nb
    return out


def local_chat_gguf_configured() -> bool:
    p = settings.LOCAL_CHAT_GGUF_PATH
    return bool(p) and os.path.isfile(p)


def local_embedding_gguf_configured() -> bool:
    p = settings.LOCAL_EMBEDDING_GGUF_PATH
    return bool(p) and os.path.isfile(p)


def _local_embed_jina_prefixes_enabled() -> bool:
    """Match Advanced RAG Jina retrieval convention (see app.advanced_rag.rag_engine)."""
    mode = (settings.LOCAL_EMBEDDING_JINA_PREFIXES or "auto").strip().lower()
    if mode in ("1", "true", "yes", "on"):
        return True
    if mode in ("0", "false", "no", "off"):
        return False
    p = (settings.LOCAL_EMBEDDING_GGUF_PATH or "").replace("\\", "/").lower()
    return "jina" in p and "retrieval" in p


def _prepare_local_embed_input(text: str, *, for_query: bool) -> str:
    """Prefix + optional normalize flag pairing for Jina text-retrieval GGUF."""
    if not _local_embed_jina_prefixes_enabled():
        return text if (text or "").strip() else " "
    t = (text or "").strip()
    if not t:
        return "Query: " if for_query else "Passage: "
    if for_query:
        if t.lower().startswith("query:"):
            return t
        return f"Query: {t}"
    if t.lower().startswith("passage:"):
        return t
    return f"Passage: {t}"


def llama_chat_completion_lock():
    """Use around `create_chat_completion` so concurrent HTTP handlers do not share one Llama unsafely."""
    return _chat_completion_lock


def get_chat_llama():
    global _chat_llama
    if _chat_llama is None:
        from llama_cpp import Llama

        path = settings.LOCAL_CHAT_GGUF_PATH
        if not path or not os.path.isfile(path):
            raise RuntimeError(
                "LOCAL_CHAT_GGUF_PATH must point to an existing .gguf file "
                f"(configured value: {path!r})"
            )
        _validate_gguf_file(path, kind="Chat")
        kw: dict[str, Any] = {
            "model_path": path,
            "n_ctx": settings.LLAMA_N_CTX,
            "n_gpu_layers": settings.LLAMA_N_GPU_LAYERS,
            "main_gpu": settings.LLAMA_MAIN_GPU,
            "verbose": settings.LLAMA_VERBOSE,
        }
        kw.update(_llama_perf_kw())
        if settings.LLAMA_CHAT_FORMAT:
            kw["chat_format"] = settings.LLAMA_CHAT_FORMAT
        logger.info(
            "Loading local chat GGUF: %s (n_threads=%s, n_batch=%s)",
            path,
            kw.get("n_threads"),
            kw.get("n_batch"),
        )
        logger.warning(
            "Llama.cpp may sit with no new logs for several minutes while memory-mapping this file "
            "(multi-GB models are normal: 2–10+ min on HDD, less on NVMe). Do not kill the process."
        )
        t0 = time.perf_counter()
        _chat_llama = _load_llama(**kw)
        logger.info("Chat GGUF loaded in %.1f s", time.perf_counter() - t0)
    return _chat_llama


def _token_len(llm: Any, text: str) -> int:
    return len(llm.tokenize(text.encode("utf-8"), add_bos=False, special=True))


def _truncate_user_to_context(llm: Any, system: str, user: str, template_overhead: int) -> str:
    """
    Shrink user text so system + user + chat-template overhead stays under n_ctx.
    llama-cpp raises if the formatted prompt length >= n_ctx before max_tokens is applied.
    """
    n_ctx = llm.n_ctx()
    ns = _token_len(llm, system)
    margin = 64
    if ns + template_overhead + margin >= n_ctx:
        raise ValueError(
            f"System prompt (~{ns} tokens) leaves no room in n_ctx={n_ctx}; "
            "increase LLAMA_N_CTX or shorten the system message."
        )
    budget = n_ctx - ns - template_overhead - margin
    nu = _token_len(llm, user)
    if nu <= budget:
        return user

    lo, hi = 0, len(user)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        nmid = _token_len(llm, user[:mid])
        if nmid <= budget:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    suffix = "\n\n[…truncated to fit LLAMA_N_CTX; increase LLAMA_N_CTX for longer inputs]"
    out = (user[:best].rstrip() + suffix) if best < len(user) else user[:best]
    while best > 0 and _token_len(llm, out) > budget:
        best = max(0, best - 500)
        out = user[:best].rstrip() + (suffix if best < len(user) else "")
    logger.warning(
        "local_chat_complete: truncated user text to ~%s chars (~%s tokens) for n_ctx=%s",
        len(out),
        _token_len(llm, out),
        n_ctx,
    )
    return out


def _embed_gpu_layer_chain() -> list[int]:
    """Try requested offload first, then partial GPU, then CPU — fits T4-class VRAM with chat+VLM resident."""
    req = int(settings.LLAMA_N_GPU_LAYERS_EMBED)
    partial = int(settings.LLAMA_N_GPU_LAYERS_EMBED_PARTIAL)
    chain: list[int] = [req]
    if req == 0:
        return chain
    if partial > 0 and partial != req:
        chain.append(partial)
    chain.append(0)
    seen: set[int] = set()
    out: list[int] = []
    for ngl in chain:
        if ngl in seen:
            continue
        seen.add(ngl)
        out.append(ngl)
    return out


def _embed_load_recoverable(err: str) -> bool:
    e = err.lower()
    return "llama_context" in e or "failed to create" in e or "cuda" in e or "vulkan" in e


def get_embed_llama():
    global _embed_llama
    if _embed_llama is None:
        path = settings.LOCAL_EMBEDDING_GGUF_PATH
        if not path or not os.path.isfile(path):
            raise RuntimeError(
                "LOCAL_EMBEDDING_GGUF_PATH must point to an existing .gguf file "
                f"(configured value: {path!r})"
            )
        _validate_gguf_file(path, kind="Embedding")
        kw: dict[str, Any] = {
            "model_path": path,
            "n_ctx": settings.LLAMA_EMBED_N_CTX,
            "n_gpu_layers": settings.LLAMA_N_GPU_LAYERS_EMBED,
            "main_gpu": settings.LLAMA_MAIN_GPU,
            "embedding": True,
            "verbose": settings.LLAMA_VERBOSE,
        }
        kw.update(_llama_perf_kw())
        logger.info("Loading local embedding GGUF: %s", path)
        logger.warning(
            "Embedding model load can also take minutes the first time; wait for 'Embedding GGUF loaded'."
        )
        t0 = time.perf_counter()
        chain = _embed_gpu_layer_chain()
        requested = chain[0]
        last_exc: BaseException | None = None
        for i, ngl in enumerate(chain):
            kw["n_gpu_layers"] = ngl
            try:
                _embed_llama = _load_llama(**kw)
                if i > 0:
                    logger.warning(
                        "Embedding GGUF loaded with n_gpu_layers=%s (fallback chain after %s failed). "
                        "Set LLAMA_N_GPU_LAYERS_EMBED=%s to avoid retries, or LLAMA_N_GPU_LAYERS_EMBED_PARTIAL=0 "
                        "to go straight to CPU after the first failure.",
                        ngl,
                        requested,
                        ngl,
                    )
                break
            except RuntimeError as e:
                last_exc = e
                err_s = str(e)
                last_idx = i == len(chain) - 1
                if last_idx or not _embed_load_recoverable(err_s):
                    raise
                nxt = chain[i + 1] if i + 1 < len(chain) else None
                logger.warning(
                    "Embedding GGUF failed with n_gpu_layers=%s (%s); retrying with n_gpu_layers=%s…",
                    ngl,
                    err_s[:200] + ("…" if len(err_s) > 200 else ""),
                    nxt,
                )
        else:
            if last_exc:
                raise last_exc
            raise RuntimeError("Embedding GGUF: empty fallback chain")
        logger.info("Embedding GGUF loaded in %.1f s", time.perf_counter() - t0)
    return _embed_llama


def local_chat_complete(system: str, user: str, max_tokens: int = 8192) -> str:
    """Single-turn chat for PRD / code prompts (system + user)."""
    llm = get_chat_llama()
    n_ctx = llm.n_ctx()
    template_overhead = min(1024, max(384, n_ctx // 6))
    user = _truncate_user_to_context(llm, system, user, template_overhead)
    ns = _token_len(llm, system)
    nu = _token_len(llm, user)
    want = max(1, int(max_tokens))
    safe_max = max(1, min(want, n_ctx - ns - nu - template_overhead - 16))
    if safe_max < want:
        logger.info(
            "local_chat_complete: max_tokens %s -> %s (n_ctx=%s, prompt~%s tokens)",
            want,
            safe_max,
            n_ctx,
            ns + nu + template_overhead,
        )
    with _chat_completion_lock:
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=safe_max,
            temperature=0.2,
        )
    choice = resp["choices"][0].get("message") or {}
    content = choice.get("content") or ""
    return str(content).strip()


def local_chat_complete_messages(messages: list[dict[str, str]], max_tokens: int | None = None) -> str:
    """Multi-turn OpenAI-style messages (role / content)."""
    llm = get_chat_llama()
    lim = (
        max_tokens if max_tokens is not None else settings.LOCAL_CHAT_MAX_TOKENS
    )
    want = max(1, int(lim))
    n_ctx = llm.n_ctx()
    template_overhead = min(1024, max(384, n_ctx // 6))
    approx_prompt = sum(
        _token_len(llm, str(m.get("content") or "")) for m in messages
    ) + template_overhead
    safe_max = max(1, min(want, n_ctx - approx_prompt - 16))
    if safe_max < want:
        logger.info(
            "local_chat_complete_messages: max_tokens %s -> %s (n_ctx=%s, approx_prompt~%s)",
            want,
            safe_max,
            n_ctx,
            approx_prompt,
        )
    with _chat_completion_lock:
        resp = llm.create_chat_completion(
            messages=messages,
            max_tokens=safe_max,
            temperature=0.2,
        )
    choice = resp["choices"][0].get("message") or {}
    content = choice.get("content") or ""
    return str(content).strip()


def _normalize_embedding(vec: Any) -> list[float]:
    if vec is None:
        return []
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    if not vec:
        return []
    first = vec[0]
    if isinstance(first, (list, tuple)):
        vec = first
    return [float(x) for x in vec]


def local_embed_texts(
    texts: Sequence[str],
    batch_size: int = 16,
    *,
    for_query: bool = False,
) -> tuple[list[list[float]], int]:
    """
    Embed texts one-by-one (llama-cpp-python embed is per-input).
    Returns (vectors, dimension). Truncates each string like Azure path (~8k chars).

    When LOCAL_EMBEDDING_JINA_PREFIXES is auto/on for Jina *text-retrieval* GGUF paths,
    uses ``Query:`` for search queries (for_query=True) and ``Passage:`` for index chunks,
    with ``normalize=True`` — same as ``app.advanced_rag.rag_engine``.
    """
    llm = get_embed_llama()
    use_jina = _local_embed_jina_prefixes_enabled()
    all_vecs: list[list[float]] = []
    dim = settings.PRD_EMBEDDING_DIM_DEFAULT
    total_n = len(texts)
    if total_n > 0:
        logger.info(
            "local_embed_texts: embedding %d chunk(s) sequentially (llama.cpp); large repos can take many minutes — progress logged every 50.",
            total_n,
        )
    done = 0
    batch: list[str] = []
    for t in texts:
        raw = (t[:8000] if t else " ")
        batch.append(_prepare_local_embed_input(raw, for_query=for_query))
        if len(batch) >= batch_size:
            for item in batch:
                with _embed_lock:
                    v = _normalize_embedding(llm.embed(item, normalize=use_jina))
                all_vecs.append(v)
                dim = len(v)
                done += 1
                if done % 50 == 0 or done == total_n:
                    logger.info("local_embed_texts: %d / %d", done, total_n)
            batch = []
    for item in batch:
        with _embed_lock:
            v = _normalize_embedding(llm.embed(item, normalize=use_jina))
        all_vecs.append(v)
        dim = len(v)
        done += 1
        if done % 50 == 0 or done == total_n:
            logger.info("local_embed_texts: %d / %d", done, total_n)
    if len(all_vecs) != len(texts):
        logger.warning("Embedding count mismatch: %s vs %s", len(all_vecs), len(texts))
    return all_vecs, dim
