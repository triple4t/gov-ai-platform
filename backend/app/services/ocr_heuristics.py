"""
Lightweight regex pre-fill for Indian document OCR when LLM JSON is empty or flat fields are missing.
Does not replace structured extraction; only fills gaps from raw_text patterns.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def _digits_only(s: str) -> str:
    return re.sub(r"\D", "", s)


def heuristic_fill_indian_ids(raw_text: str) -> dict[str, Any]:
    """
    Extract Aadhaar (12 digits), PAN, mobile (+91 / 10-digit), and a simple DOB from OCR text.
    Returns a dict with only keys that matched (subset of: id_number, phone_number, date_of_birth).
    """
    if not raw_text or not str(raw_text).strip():
        return {}

    text = str(raw_text)
    upper = text.upper()
    out: dict[str, Any] = {}

    # Aadhaar: XXXX XXXX XXXX or 12 consecutive digits
    for pattern in (r"\b(\d{4}\s+\d{4}\s+\d{4})\b", r"\b(\d{12})\b"):
        m = re.search(pattern, text)
        if m:
            d = _digits_only(m.group(1))
            if len(d) == 12:
                out["id_number"] = f"{d[:4]} {d[4:8]} {d[8:]}"
                break

    # PAN (only if id_number still empty — avoid overwriting Aadhaar)
    if "id_number" not in out:
        m = re.search(r"\b([A-Z]{5}\d{4}[A-Z])\b", upper)
        if m:
            out["id_number"] = m.group(1)

    # Mobile: +91 optional, then 10 digits starting 6–9
    m = re.search(r"(?:\+91|91)[\s\-]*([6-9]\d{9})\b", text.replace(" ", ""))
    if not m:
        m = re.search(r"\b([6-9]\d{9})\b", re.sub(r"[^\d]", " ", text))
    if m:
        out["phone_number"] = f"+91 {m.group(1)}"

    # DOB common on IDs: DD/MM/YYYY or DD-MM-YYYY
    m = re.search(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", text)
    if m:
        out["date_of_birth"] = m.group(1)

    if out:
        logger.info("OCR heuristic pre-fill from raw_text: keys=%s", list(out.keys()))
    return out


def flat_form_fields_empty(d: dict[str, Any]) -> bool:
    """True if all five UI form scalars are missing or blank."""
    keys = ("name", "id_number", "date_of_birth", "address", "phone_number")
    for k in keys:
        v = d.get(k)
        if v is not None and str(v).strip():
            return False
    return True


def merge_heuristic_into_structured(structured: dict[str, Any], raw_text: str) -> dict[str, Any]:
    """
    If flat fields are empty, merge heuristic keys into structured dict (preserving fields/other_fields).
    """
    if not flat_form_fields_empty(structured):
        return structured
    hints = heuristic_fill_indian_ids(raw_text)
    if not hints:
        return structured
    merged = dict(structured)
    for k, v in hints.items():
        if v and not (merged.get(k) or "").strip():
            merged[k] = v
    merged.setdefault("fields", merged.get("fields") or {})
    merged.setdefault("other_fields", merged.get("other_fields") or {})
    if not isinstance(merged["fields"], dict):
        merged["fields"] = {}
    if not isinstance(merged["other_fields"], dict):
        merged["other_fields"] = {}
    return merged
