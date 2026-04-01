"""Extract plain text from uploaded files for Hybrid RAG ingest (no full OCR pipeline import)."""

from __future__ import annotations

import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text_for_rag(filename: str, data: bytes) -> str:
    """
    Best-effort text extraction. Raises ValueError with a short message if unsupported or empty.
    """
    if not data:
        raise ValueError("Empty file.")
    name = (filename or "upload").strip()
    ext = Path(name).suffix.lower()

    if ext == ".pdf":
        import fitz

        doc = fitz.open(stream=data, filetype="pdf")
        try:
            parts: list[str] = []
            for page in doc:
                parts.append(page.get_text() or "")
            text = "\n".join(parts).strip()
        finally:
            doc.close()
        if not text:
            raise ValueError("No extractable text in PDF (may be scanned images only — use an image upload with Tesseract/VLM configured).")
        return text

    if ext == ".docx":
        try:
            from docx import Document
        except ImportError as e:
            raise ValueError("python-docx not installed.") from e
        d = Document(io.BytesIO(data))
        text = "\n".join(p.text for p in d.paragraphs if p.text).strip()
        if not text:
            raise ValueError("No text found in DOCX.")
        return text

    if ext == ".doc":
        try:
            import mammoth
        except ImportError as e:
            raise ValueError("mammoth not installed for .doc files.") from e
        result = mammoth.extract_raw_text(io.BytesIO(data))
        text = (result.value or "").strip()
        if not text:
            raise ValueError("No text extracted from .doc.")
        return text

    if ext in (".txt", ".md", ".csv", ".json", ".log", ".xml", ".html", ".htm", ".rtf"):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="replace")
        text = text.strip()
        if not text:
            raise ValueError("Text file is empty.")
        return text

    if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"):
        try:
            import pytesseract
            from PIL import Image
        except ImportError as e:
            raise ValueError("Image OCR needs Pillow and pytesseract (and Tesseract binary on PATH).") from e
        try:
            img = Image.open(io.BytesIO(data))
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            text = (pytesseract.image_to_string(img) or "").strip()
        except Exception as e:
            raise ValueError(f"Image OCR failed: {e}") from e
        if not text:
            raise ValueError("No text read from image (try a clearer scan or paste text manually).")
        return text

    raise ValueError(
        f"Unsupported type '{ext or '(no extension)'}'. "
        "Use .pdf, .docx, .doc, .txt/.md, or image (.png, .jpg, …)."
    )
