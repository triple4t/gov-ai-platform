import os
import pytesseract
import cv2
import numpy as np
from PIL import Image
import fitz
from app.core.config import settings

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

# Set up the tesseract executable path from settings
if settings.TESSERACT_CMD and os.path.exists(settings.TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD

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

def extract_text_from_image(image_path: str) -> str:
    """Extracts text from a single image file."""
    try:
        # Read using OpenCV first for preprocessing
        image_array = cv2.imread(image_path)
        enhanced_image = enhance_image_for_ocr(image_array)
        
        # fallback to standard PIL if OpenCV fails to read for some reason
        if enhanced_image is None:
            image = Image.open(image_path)
            enhanced_image = image
            
        custom_config = r'--oem 3 --psm 6' 
        text = pytesseract.image_to_string(enhanced_image, config=custom_config)
        return text
    except Exception as e:
        print(f"OCR Error on image {image_path}: {e}")
        return ""

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

        # 2. Fallback: OCR via Tesseract for scanned or image-only PDFs
        try:
            full_text = ""
            for page_number in range(len(doc)):
                page = doc.load_page(page_number)
                pix = page.get_pixmap()
                img = np.frombuffer(pix.samples, dtype=np.uint8)
                img = img.reshape(pix.height, pix.width, pix.n)
                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                enhanced_image = enhance_image_for_ocr(img)
                custom_config = r'--oem 3 --psm 6'
                text = pytesseract.image_to_string(enhanced_image, config=custom_config)
                full_text += f"\n--- Page {page_number + 1} ---\n{text}"
            doc.close()
            return full_text.strip() or direct_text
        except Exception as ocr_err:
            doc.close()
            # If Tesseract isn't installed, return whatever we got from direct extraction
            print(f"OCR fallback failed on PDF {pdf_path}: {ocr_err}")
            return direct_text

    except Exception as e:
        print(f"OCR Error on PDF {pdf_path}: {e}")
        return ""


def extract_text_from_docx(docx_path: str) -> str:
    """Extract text from a .docx file."""
    if not _DOCX_AVAILABLE:
        raise ValueError("python-docx is not installed. Install with: pip install python-docx")
    try:
        doc = DocxDocument(docx_path)
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        print(f"Error reading DOCX {docx_path}: {e}")
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
        print(f"Error reading DOC {doc_path}: {e}")
        return ""


def extract_text_from_file(file_path: str) -> str:
    """Determines file type and extracts text. Supports PDF, images, .docx, .doc."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in [".png", ".jpg", ".jpeg", ".tiff"]:
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
