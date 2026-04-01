"""
Local multimodal document OCR via llama-cpp-python + vision chat handler.

Handlers:
- **qwen25_vl** — `Qwen25VLChatHandler` (e.g. noctrex/LightOnOCR-2-1B-GGUF + mmproj).
- **glm_ocr** — GLM-OCR GGUF ([gMASK] / <|user|> / image placeholders per zai-org/GLM-OCR chat template).

Set `LOCAL_OCR_VLM_GGUF_PATH`, `LOCAL_OCR_VLM_MMPROJ_PATH`, and optionally
`LOCAL_OCR_VLM_CHAT_HANDLER=auto|qwen25_vl|glm_ocr` (`auto` picks glm when the main path contains `glm-ocr`).
"""

from __future__ import annotations

import base64
import io
import logging
from collections import Counter
import mimetypes
import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Generator

from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)

_vlm_stderr_fd_lock = threading.Lock()
_vlm_llama: Any | None = None
_VLM_IMPORT_ERROR: str | None = None
GlmOcrChatHandler: Any = None

try:
    from llama_cpp.llama_chat_format import Llava15ChatHandler, Qwen25VLChatHandler

    class GlmOcrChatHandler(Llava15ChatHandler):
        """GLM-OCR / glm4v-style prompt for ggml-org/GLM-OCR-GGUF (see zai-org/GLM-OCR chat_template.jinja)."""

        DEFAULT_SYSTEM_MESSAGE = None

        CHAT_FORMAT = (
            "[gMASK] "
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|system|>\n"
            "{% if message['content'] is string %}{{ message['content'] }}{% endif %}"
            "{% elif message['role'] == 'user' %}"
            "<|user|>\n"
            "{% if message['content'] is string %}"
            "{{ message['content'] }}/nothink"
            "{% else %}"
            "{% for content in message['content'] %}"
            "{% if content['type'] == 'image_url' %}"
            "{% if content.image_url is string %}"
            "{{ content.image_url }}"
            "{% else %}"
            "{{ content.image_url.url }}"
            "{% endif %}"
            "{% elif content['type'] == 'text' %}"
            "{{ content['text'] }}"
            "{% endif %}"
            "{% endfor %}"
            "/nothink"
            "{% endif %}"
            "{% elif message['role'] == 'assistant' and message.content is not none %}"
            "<|assistant|>\n{{ message['content'] }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "<|assistant|>\n \n"
            "{% endif %}"
        )

except ImportError as e:  # pragma: no cover
    Llava15ChatHandler = None  # type: ignore[misc, assignment]
    Qwen25VLChatHandler = None  # type: ignore[misc, assignment]
    GlmOcrChatHandler = None
    _VLM_IMPORT_ERROR = str(e)


def vlm_ocr_chat_handler_mode() -> str:
    """Return ``glm_ocr`` or ``qwen25_vl`` (for logging / diagnostics)."""
    explicit = (settings.LOCAL_OCR_VLM_CHAT_HANDLER or "auto").strip().lower()
    main_l = (settings.LOCAL_OCR_VLM_GGUF_PATH or "").lower()
    if explicit == "glm_ocr":
        return "glm_ocr"
    if explicit == "qwen25_vl":
        return "qwen25_vl"
    if explicit in ("", "auto"):
        return "glm_ocr" if "glm-ocr" in main_l else "qwen25_vl"
    logger.warning("Unknown LOCAL_OCR_VLM_CHAT_HANDLER=%r — using qwen25_vl", explicit)
    return "qwen25_vl"


def vlm_ocr_configured() -> bool:
    if _VLM_IMPORT_ERROR or Llava15ChatHandler is None:
        return False
    mode = vlm_ocr_chat_handler_mode()
    if mode == "glm_ocr" and GlmOcrChatHandler is None:
        return False
    if mode == "qwen25_vl" and Qwen25VLChatHandler is None:
        return False
    main = (settings.LOCAL_OCR_VLM_GGUF_PATH or "").strip()
    mm = (settings.LOCAL_OCR_VLM_MMPROJ_PATH or "").strip()
    if not main or not mm:
        return False
    return os.path.isfile(main) and os.path.isfile(mm)


def _ocr_system_prompt() -> str:
    s = (settings.LOCAL_OCR_VLM_SYSTEM_PROMPT or "").strip()
    return s or "You are a helpful OCR assistant."


def _ocr_user_prompt() -> str:
    """Default text matches backend/backend app.py `lighton_ocr_image` (PRD mini-backend)."""
    s = (settings.LOCAL_OCR_VLM_USER_PROMPT or "").strip()
    if s:
        return s
    return (
        "Extract all readable text from this image. "
        "If this is a flowchart, diagram, wireframe, or architecture drawing, "
        "transcribe labels and describe structure as plain text: list steps/boxes, "
        "arrows as lines like 'A -> B', and group related elements. "
        "Preserve reading order. Do not refuse; output text only."
    )


def _fit_max_side(im: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return im
    w, h = im.size
    if w <= max_side and h <= max_side:
        return im
    scale = max_side / max(w, h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return im.resize((nw, nh), Image.Resampling.LANCZOS)


def _pil_to_jpeg_data_uri(im: Image.Image, *, quality: int = 90) -> str:
    rgb = im.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _image_to_vlm_data_uri(im: Image.Image, *, max_side: int | None = None) -> str:
    """Resize (if configured) and JPEG-encode for stable VLM token counts and size."""
    ms = int(settings.LOCAL_OCR_VLM_MAX_IMAGE_SIDE) if max_side is None else int(max_side)
    fitted = _fit_max_side(im, ms)
    if fitted.size != im.size:
        logger.debug("VLM OCR: resized %s → %s (max_side=%s)", im.size, fitted.size, ms)
    return _pil_to_jpeg_data_uri(fitted)


def image_file_to_data_uri(file_path: str, max_image_side: int | None = None) -> str:
    """Build data URI for VLM: native file + MIME when within max side (like lighton_ocr_image), else JPEG resize."""
    mode = (settings.LOCAL_OCR_VLM_DATA_URI_MODE or "native_if_fits").strip().lower()
    ms = int(settings.LOCAL_OCR_VLM_MAX_IMAGE_SIDE) if max_image_side is None else int(max_image_side)
    if mode == "jpeg_fitted":
        with Image.open(file_path) as im:
            return _image_to_vlm_data_uri(im, max_side=ms)
    with Image.open(file_path) as im:
        w, h = im.size
    if ms > 0 and max(w, h) > ms:
        with Image.open(file_path) as im:
            return _image_to_vlm_data_uri(im, max_side=ms)
    abs_path = os.path.abspath(file_path)
    mime_type, _ = mimetypes.guess_type(abs_path)
    mime_type = mime_type or "image/jpeg"
    with open(abs_path, "rb") as f:
        b64_data = base64.standard_b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64_data}"


def _vlm_thread_kw() -> dict[str, int]:
    if settings.LLAMA_THREADS > 0:
        return {"n_threads": settings.LLAMA_THREADS}
    return {}


@contextmanager
def _suppress_llama_cpp_stderr() -> Generator[None, None, None]:
    """Hide llama.cpp clip_model_loader / add_text spam unless LOCAL_OCR_VLM_VERBOSE is on."""
    if settings.LOCAL_OCR_VLM_VERBOSE:
        yield
        return
    prev_ggml = os.environ.get("GGML_LOG_LEVEL")
    os.environ["GGML_LOG_LEVEL"] = "error"
    devnull = open(os.devnull, "w", encoding="utf-8")
    old_err = sys.stderr
    saved_fd2: int | None = None
    fd_lock_held = False
    try:
        sys.stderr = devnull
        if settings.LOCAL_OCR_VLM_SILENCE_STDERR_FD:
            _vlm_stderr_fd_lock.acquire()
            fd_lock_held = True
            try:
                saved_fd2 = os.dup(2)
                os.dup2(devnull.fileno(), 2)
            except OSError:
                if saved_fd2 is not None:
                    try:
                        os.close(saved_fd2)
                    except OSError:
                        pass
                    saved_fd2 = None
        yield
    finally:
        if saved_fd2 is not None:
            try:
                os.dup2(saved_fd2, 2)
            except OSError:
                pass
            try:
                os.close(saved_fd2)
            except OSError:
                pass
        if fd_lock_held:
            _vlm_stderr_fd_lock.release()
        sys.stderr = old_err
        devnull.close()
        if prev_ggml is None:
            os.environ.pop("GGML_LOG_LEVEL", None)
        else:
            os.environ["GGML_LOG_LEVEL"] = prev_ggml


def get_vlm_ocr_llama() -> Any:
    global _vlm_llama
    if _vlm_llama is None:
        if not vlm_ocr_configured():
            raise RuntimeError(
                "VLM OCR not configured: set LOCAL_OCR_VLM_GGUF_PATH and LOCAL_OCR_VLM_MMPROJ_PATH"
            )
        from llama_cpp import Llama

        main = os.path.normpath(settings.LOCAL_OCR_VLM_GGUF_PATH.strip())
        mm = os.path.normpath(settings.LOCAL_OCR_VLM_MMPROJ_PATH.strip())
        mode = vlm_ocr_chat_handler_mode()
        if mode == "glm_ocr":
            if GlmOcrChatHandler is None:
                raise RuntimeError("GlmOcrChatHandler unavailable (llama_cpp import failed)")
            handler_cls: type = GlmOcrChatHandler
            logger.info("VLM OCR chat handler: glm_ocr (GLM-OCR template)")
        else:
            if Qwen25VLChatHandler is None:
                raise RuntimeError("Qwen25VLChatHandler unavailable (llama_cpp import failed)")
            handler_cls = Qwen25VLChatHandler
            logger.info("VLM OCR chat handler: qwen25_vl")
        chat_handler = handler_cls(clip_model_path=mm, verbose=settings.LOCAL_OCR_VLM_VERBOSE)
        kw: dict[str, Any] = {
            "model_path": main,
            "n_ctx": settings.LOCAL_OCR_VLM_N_CTX,
            "n_gpu_layers": settings.LOCAL_OCR_VLM_N_GPU_LAYERS,
            "main_gpu": settings.LLAMA_MAIN_GPU,
            "chat_handler": chat_handler,
            "verbose": settings.LOCAL_OCR_VLM_VERBOSE,
        }
        kw.update(_vlm_thread_kw())
        logger.info("Loading VLM OCR GGUF: %s (mmproj: %s)", main, mm)
        t0 = time.perf_counter()
        try:
            with _suppress_llama_cpp_stderr():
                _vlm_llama = Llama(**kw)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load VLM OCR model: {e}. "
                "Check paths, llama-cpp-python build, and GPU/CPU settings (LOCAL_OCR_VLM_N_GPU_LAYERS)."
            ) from e
        logger.info("VLM OCR model loaded in %.1f s", time.perf_counter() - t0)
    return _vlm_llama


def _vlm_ocr_messages(
    image_data_uri: str, *, force_user_only: bool = False
) -> list[dict[str, Any]]:
    """Same shape as backend/backend `lighton_ocr_image`: single user turn unless INCLUDE_SYSTEM is set.

    When force_user_only=True we omit the app system message; Qwen/llama.cpp may still apply a chat-template
    system block in traces — that is not controlled here.
    """
    user_content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": image_data_uri}},
        {"type": "text", "text": _ocr_user_prompt()},
    ]
    include_system = bool(settings.LOCAL_OCR_VLM_INCLUDE_SYSTEM) and not force_user_only
    if include_system:
        return [
            {"role": "system", "content": _ocr_system_prompt()},
            {"role": "user", "content": user_content},
        ]
    return [{"role": "user", "content": user_content}]


def _vlm_output_log_preview(text: str, *, limit: int = 200) -> str:
    """Short log line: avoid megabytes of repeated '!' interleaving with other log output on Windows consoles."""
    if not text:
        return ""
    s = text.strip()
    non_ws = [c for c in s if not c.isspace()]
    if len(non_ws) >= 48:
        ch, n = Counter(non_ws).most_common(1)[0]
        if n / len(non_ws) >= 0.72:
            return f"<{len(s)} chars, ~{100 * n // len(non_ws)}% {ch!r}>"
    one = s.replace("\r", " ").replace("\n", "\\n")
    if len(one) <= limit:
        return one
    return one[: limit - 1] + "…"


def _run_vlm_on_image_uri(
    image_data_uri: str,
    *,
    max_tokens: int | None = None,
    repeat_penalty: float | None = None,
    force_user_only: bool = False,
) -> str:
    llm = get_vlm_ocr_llama()
    messages = _vlm_ocr_messages(image_data_uri, force_user_only=force_user_only)
    mt = int(settings.LOCAL_OCR_VLM_MAX_TOKENS) if max_tokens is None else int(max_tokens)
    mt = max(32, mt)
    rp = float(settings.LOCAL_OCR_VLM_REPEAT_PENALTY) if repeat_penalty is None else float(repeat_penalty)
    rp = max(1.0, rp)
    with _suppress_llama_cpp_stderr():
        resp = llm.create_chat_completion(
            messages=messages,
            max_tokens=mt,
            temperature=float(settings.LOCAL_OCR_VLM_TEMPERATURE),
            repeat_penalty=rp,
        )
    choice = resp["choices"][0].get("message") or {}
    text = str(choice.get("content") or "").strip()
    preview = _vlm_output_log_preview(text)
    with _suppress_llama_cpp_stderr():
        logger.info("VLM OCR text: chars=%d preview=%s", len(text), preview)
    return text


def extract_text_from_image_vlm(
    image_path: str,
    max_image_side: int | None = None,
    *,
    force_user_only: bool = False,
    max_tokens: int | None = None,
    repeat_penalty: float | None = None,
) -> str:
    if not vlm_ocr_configured():
        return ""
    try:
        uri = image_file_to_data_uri(image_path, max_image_side=max_image_side)
        return _run_vlm_on_image_uri(
            uri,
            max_tokens=max_tokens,
            repeat_penalty=repeat_penalty,
            force_user_only=force_user_only,
        )
    except Exception as e:
        logger.warning("VLM OCR failed for %s: %s", image_path, e)
        return ""


def extract_text_from_png_bytes_vlm(
    png_bytes: bytes,
    max_image_side: int | None = None,
    *,
    force_user_only: bool = False,
    max_tokens: int | None = None,
    repeat_penalty: float | None = None,
) -> str:
    """OCR a single page rendered as PNG bytes (e.g. from PyMuPDF)."""
    if not vlm_ocr_configured():
        return ""
    try:
        im = Image.open(io.BytesIO(png_bytes))
        ms = int(settings.LOCAL_OCR_VLM_MAX_IMAGE_SIDE) if max_image_side is None else int(max_image_side)
        uri = _image_to_vlm_data_uri(im, max_side=ms)
        return _run_vlm_on_image_uri(
            uri,
            max_tokens=max_tokens,
            repeat_penalty=repeat_penalty,
            force_user_only=force_user_only,
        )
    except Exception as e:
        logger.warning("VLM OCR failed on PNG raster: %s", e)
        return ""
