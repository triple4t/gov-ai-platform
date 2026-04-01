import logging
import os
import shutil
from collections import Counter
from typing import Any

import cv2
import fitz
import numpy as np
import pytesseract
from PIL import Image

from app.core.config import settings
from app.core.vlm_ocr import (
    extract_text_from_image_vlm,
    extract_text_from_png_bytes_vlm,
    vlm_ocr_configured,
)

logger = logging.getLogger(__name__)


def _raster_ocr_mode() -> str:
    m = (settings.OCR_RASTER_BACKEND or "auto").strip().lower()
    if m not in ("auto", "vlm", "tesseract"):
        return "auto"
    return m


def _vlm_text_looks_garbage(text: str) -> bool:
    """Heuristic: VLM returned mostly filler ('?', '!', replacement chars) — prefer Tesseract."""
    if not text:
        return False
    s = text.strip()
    if len(s) < 8:
        return False
    if len(set(s)) <= 2:
        return True
    # Stuck decoding one token (common when vision grid/context is overloaded)
    non_ws = [c for c in s if not c.isspace()]
    if non_ws:
        top = Counter(non_ws).most_common(1)[0][1]
        if top / len(non_ws) >= 0.82:
            return True
    for ch in ("?", "!", "\ufffd"):
        if s.count(ch) / len(s) >= 0.45:
            return True
    bad = sum(1 for c in s if c == "\ufffd" or ("\uE000" <= c <= "\uF8FF"))
    if bad / len(s) >= 0.3:
        return True
    return False


def _vlm_garbage_retry_sides() -> list[int]:
    raw = (settings.LOCAL_OCR_VLM_GARBAGE_RETRY_SIDES or "").strip()
    out: list[int] = []
    if raw and raw.lower() not in ("none", "off", "-"):
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                v = int(part)
            except ValueError:
                continue
            if v > 0:
                out.append(v)
    if not out:
        ms = int(settings.LOCAL_OCR_VLM_GARBAGE_RETRY_MAX_SIDE or 0)
        if ms > 0:
            out.append(ms)
    return out


def _vlm_garbage_retry_decode_kw() -> dict[str, Any]:
    mt = int(settings.LOCAL_OCR_VLM_GARBAGE_RETRY_MAX_TOKENS or 0)
    mt_arg: int | None = mt if mt >= 32 else None
    rp = float(settings.LOCAL_OCR_VLM_GARBAGE_RETRY_REPEAT_PENALTY)
    if rp <= 1.0:
        rp = 1.35
    return {
        "force_user_only": bool(settings.LOCAL_OCR_VLM_GARBAGE_RETRY_USER_ONLY),
        "max_tokens": mt_arg,
        "repeat_penalty": rp,
    }


# Optional: Word documents (.docx, .doc)
try:
    from docx import Document as DocxDocument
    _DOCX_AVAILABLE = True
except ImportError:
    _DOCX_AVAILABLE = False
try:
    import mammoth
    _MAMMOTH_AVAILABLE = True
except ImportError:
    _MAMMOTH_AVAILABLE = False

def _resolve_tesseract_executable() -> str | None:
    """Find tesseract: TESSERACT_CMD if valid, then PATH, then common Windows install paths."""
    candidates: list[str] = []
    configured = (settings.TESSERACT_CMD or "").strip()
    if configured:
        candidates.append(os.path.expandvars(os.path.expanduser(configured)))
    which = shutil.which("tesseract")
    if which:
        candidates.append(which)
    if os.name == "nt":
        candidates.extend(
            (
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            )
        )
    seen: set[str] = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        if os.path.isfile(path):
            return path
    return None


_tesseract = _resolve_tesseract_executable()
TESSERACT_AVAILABLE = _tesseract is not None
if _tesseract:
    pytesseract.pytesseract.tesseract_cmd = _tesseract
    logger.info("Tesseract OCR: %s", _tesseract)
else:
    if vlm_ocr_configured():
        logger.info(
            "Tesseract not found — image/scanned-PDF fallback OCR disabled; primary path is local VLM."
        )
    else:
        logger.warning(
            "Tesseract OCR executable not found — set LOCAL_OCR_VLM_GGUF_PATH + LOCAL_OCR_VLM_MMPROJ_PATH "
            "for VLM OCR, or install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki "
            "and set TESSERACT_CMD if needed."
        )

def enhance_image_for_ocr(image_array):
    """Applies advanced OpenCV transformations to handle noisy/complex background ID cards."""
    if image_array is None:
        return None
    
    # 1. Convert to grayscale
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_array

    # 2. Rescale (often helps with small text on IDs)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 3. Denoise - helps remove the texture in Aadhaar backgrounds
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # 4. Otsu's thresholding
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. Morphological operations to clean up small specks
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return processed


def _tesseract_from_pixmap(pix) -> str:
    if not TESSERACT_AVAILABLE:
        return ""
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    enhanced_image = enhance_image_for_ocr(img)
    return pytesseract.image_to_string(enhanced_image, config=r"--oem 3 --psm 6")


def _extract_text_from_image_tesseract(image_path: str) -> str:
    if not TESSERACT_AVAILABLE:
        return ""
    try:
        image_array = cv2.imread(image_path)
        enhanced_image = enhance_image_for_ocr(image_array)
        if enhanced_image is None:
            enhanced_image = Image.open(image_path)
        return pytesseract.image_to_string(enhanced_image, config=r"--oem 3 --psm 6")
    except Exception as e:
        logger.warning("Tesseract OCR error on image %s: %s", image_path, e)
        return ""


def _ocr_pdf_page_raster(pix, page_index: int) -> str:
    mode = _raster_ocr_mode()
    png_bytes = pix.tobytes("png")
    if mode == "tesseract":
        return _tesseract_from_pixmap(pix)
    if mode == "vlm":
        return extract_text_from_png_bytes_vlm(png_bytes) if vlm_ocr_configured() else ""
    if vlm_ocr_configured():
        vlm_text = extract_text_from_png_bytes_vlm(png_bytes) or ""
        if vlm_text.strip() and not _vlm_text_looks_garbage(vlm_text):
            return vlm_text
        if TESSERACT_AVAILABLE:
            logger.info("VLM OCR empty or low-quality on PDF page %s; trying Tesseract", page_index + 1)
            tess = (_tesseract_from_pixmap(pix) or "").strip()
            if tess:
                return tess
            if _vlm_text_looks_garbage(vlm_text):
                logger.warning(
                    "VLM OCR unusable and Tesseract returned empty on PDF page %s — "
                    "install Tesseract + language packs or lower LOCAL_OCR_VLM_MAX_IMAGE_SIDE.",
                    page_index + 1,
                )
                return ""
            return vlm_text.strip()
        if _vlm_text_looks_garbage(vlm_text):
            dec_kw = _vlm_garbage_retry_decode_kw()
            for side in _vlm_garbage_retry_sides():
                logger.info(
                    "VLM OCR garbage on PDF page %s; retry max_image_side=%s max_tokens=%s force_user_only=%s "
                    "(drops app system if any; no Tesseract)",
                    page_index + 1,
                    side,
                    dec_kw.get("max_tokens") or "(default)",
                    dec_kw.get("force_user_only"),
                )
                vlm2 = (
                    extract_text_from_png_bytes_vlm(
                        png_bytes, max_image_side=side, **dec_kw
                    )
                    or ""
                )
                if vlm2.strip() and not _vlm_text_looks_garbage(vlm2):
                    return vlm2
            logger.warning(
                "VLM OCR unusable (garbage after retries) on PDF page %s; Tesseract not installed — "
                "install Tesseract (TESSERACT_CMD) or tune LOCAL_OCR_VLM_*. "
                "Note: llama.cpp may keep a large vision token grid (image_tokens->nx) even when max_image_side shrinks.",
                page_index + 1,
            )
            return ""
        if not vlm_text.strip():
            logger.info("VLM OCR empty on PDF page %s; Tesseract not installed", page_index + 1)
        return vlm_text.strip()
    return _tesseract_from_pixmap(pix)


def extract_text_from_image(image_path: str) -> str:
    """Raster OCR: VLM (if configured) and/or Tesseract per OCR_RASTER_BACKEND."""
    mode = _raster_ocr_mode()
    text = ""
    if mode == "tesseract":
        text = _extract_text_from_image_tesseract(image_path)
    elif mode == "vlm":
        text = extract_text_from_image_vlm(image_path)
    else:
        if vlm_ocr_configured():
            vlm_text = extract_text_from_image_vlm(image_path) or ""
            if vlm_text.strip() and not _vlm_text_looks_garbage(vlm_text):
                text = vlm_text
            elif TESSERACT_AVAILABLE:
                logger.info(
                    "VLM OCR empty or low-quality; trying Tesseract for %s",
                    os.path.basename(image_path),
                )
                tess = (_extract_text_from_image_tesseract(image_path) or "").strip()
                if tess:
                    text = tess
                elif _vlm_text_looks_garbage(vlm_text):
                    text = ""
                    logger.warning(
                        "VLM OCR unusable and Tesseract returned empty for %s — "
                        "install Tesseract + language packs (e.g. hin+eng) or lower LOCAL_OCR_VLM_MAX_IMAGE_SIDE.",
                        os.path.basename(image_path),
                    )
                else:
                    text = vlm_text
            else:
                if _vlm_text_looks_garbage(vlm_text):
                    dec_kw = _vlm_garbage_retry_decode_kw()
                    text = ""
                    for side in _vlm_garbage_retry_sides():
                        logger.info(
                            "VLM OCR garbage; retry max_image_side=%s max_tokens=%s force_user_only=%s "
                            "(drops app system if any; no Tesseract): %s",
                            side,
                            dec_kw.get("max_tokens") or "(default)",
                            dec_kw.get("force_user_only"),
                            os.path.basename(image_path),
                        )
                        vlm2 = (
                            extract_text_from_image_vlm(
                                image_path, max_image_side=side, **dec_kw
                            )
                            or ""
                        )
                        if vlm2.strip() and not _vlm_text_looks_garbage(vlm2):
                            text = vlm2
                            break
                    if not (text or "").strip():
                        logger.warning(
                            "VLM OCR unusable (garbage after retries); Tesseract not installed — "
                            "install Tesseract (TESSERACT_CMD) or tune LOCAL_OCR_VLM_* (%s). "
                            "If stderr is noisy, set LOCAL_OCR_VLM_SILENCE_STDERR_FD=true (single-worker dev).",
                            os.path.basename(image_path),
                        )
                else:
                    text = vlm_text
                    if not (text or "").strip():
                        logger.info(
                            "VLM OCR returned empty; Tesseract not available for %s",
                            os.path.basename(image_path),
                        )
        else:
            text = _extract_text_from_image_tesseract(image_path)

    if not (text or "").strip() and not vlm_ocr_configured() and not TESSERACT_AVAILABLE:
        logger.warning(
            "No OCR engine for image %s — set LOCAL_OCR_VLM_GGUF_PATH + LOCAL_OCR_VLM_MMPROJ_PATH "
            "or install Tesseract and TESSERACT_CMD.",
            os.path.basename(image_path),
        )
    return text

def _extract_text_from_pdf_direct(doc) -> str:
    """Extract text directly from PDF pages (no Tesseract). Works for digital PDFs with selectable text."""
    full_text = ""
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        full_text += page.get_text()
        if page_number < len(doc) - 1:
            full_text += "\n\n"
    return full_text.strip()


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF. Uses direct text extraction for digital PDFs; falls back to Tesseract OCR for scanned/image-only PDFs."""
    try:
        doc = fitz.open(pdf_path)

        # 1. Try direct text extraction first (no Tesseract needed) — works for most documents like offer letters
        direct_text = _extract_text_from_pdf_direct(doc)
        if len(direct_text.strip()) > 50:
            doc.close()
            return direct_text

        # 2. Fallback: raster OCR (local VLM and/or Tesseract) for scanned or image-only PDFs
        try:
            full_text = ""
            for page_number in range(len(doc)):
                page = doc.load_page(page_number)
                pix = page.get_pixmap()
                text = _ocr_pdf_page_raster(pix, page_number)
                full_text += f"\n--- Page {page_number + 1} ---\n{text}"
            doc.close()
            return full_text.strip() or direct_text
        except Exception as ocr_err:
            doc.close()
            # If Tesseract isn't installed, return whatever we got from direct extraction
            logger.warning("OCR fallback failed on PDF %s: %s", pdf_path, ocr_err)
            return direct_text

    except Exception as e:
        logger.warning("OCR error on PDF %s: %s", pdf_path, e)
        return ""


def extract_text_from_docx(docx_path: str) -> str:
    """Extract text from a .docx file."""
    if not _DOCX_AVAILABLE:
        raise ValueError("python-docx is not installed. Install with: pip install python-docx")
    try:
        doc = DocxDocument(docx_path)
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        logger.warning("Error reading DOCX %s: %s", docx_path, e)
        return ""


def extract_text_from_doc(doc_path: str) -> str:
    """Extract text from a .doc (binary Word) file using mammoth."""
    if not _MAMMOTH_AVAILABLE:
        raise ValueError("mammoth is not installed. Install with: pip install mammoth")
    try:
        with open(doc_path, "rb") as f:
            result = mammoth.extract_raw_text(f)
        return result.value or ""
    except Exception as e:
        logger.warning("Error reading DOC %s: %s", doc_path, e)
        return ""


def extract_text_from_file(file_path: str) -> str:
    """Determines file type and extracts text. Supports PDF, images, .docx, .doc."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".tif"]:
        return extract_text_from_image(file_path)
    elif ext == ".docx" and _DOCX_AVAILABLE:
        return extract_text_from_docx(file_path)
    elif ext == ".doc" and _MAMMOTH_AVAILABLE:
        return extract_text_from_doc(file_path)
    elif ext in [".docx", ".doc"]:
        raise ValueError(
            "Word support requires python-docx (.docx) and mammoth (.doc). "
            "Install with: pip install python-docx mammoth"
        )
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
