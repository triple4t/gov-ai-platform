"""
Document Summarization Service.
Uses the same OpenAI/Azure LLM as doc_extraction_service to generate
concise summaries and structured insights from document text.
Supports vision fallback for images when Tesseract OCR is not available.
"""

import base64
import json
import logging
import re
from openai import OpenAI
from app.core.config import settings
from app.core.llm_routing import llm_provider_is_local

logger = logging.getLogger(__name__)

# Max characters to send to the model (leave room for response)
MAX_INPUT_CHARS = 120_000


def _cloud_openai_client() -> OpenAI:
    return OpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
    )


def extract_text_from_image_with_vision(image_path: str) -> str:
    """
    Fallback when raster OCR returns little/no text: cloud vision (gpt-4o) or local VLM OCR
    (LOCAL_OCR_VLM_* + Qwen25VLChatHandler). Returns empty string on failure.
    """
    if llm_provider_is_local():
        from app.core.vlm_ocr import extract_text_from_image_vlm, vlm_ocr_configured

        if vlm_ocr_configured():
            text = extract_text_from_image_vlm(image_path)
            if text.strip():
                logger.info("Summarize: extracted text via local VLM OCR.")
            return text
        logger.info(
            "Summarize vision fallback: set LOCAL_OCR_VLM_GGUF_PATH and LOCAL_OCR_VLM_MMPROJ_PATH "
            "(e.g. LightOnOCR GGUF + mmproj) or use OPENAI_API_KEY for cloud vision."
        )
        return ""
    if not settings.OPENAI_API_KEY:
        return ""
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        b64 = base64.standard_b64encode(data).decode("utf-8")
        ext = image_path.lower().split(".")[-1] if "." in image_path else "png"
        mime = "image/png" if ext == "png" else "image/jpeg" if ext in ("jpg", "jpeg") else "image/tiff"
        response = _cloud_openai_client().chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all text from this image exactly as written. Preserve structure (paragraphs, bullets) where possible. Return only the extracted text, no commentary.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                    ],
                }
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        if text:
            logger.info("Vision fallback extracted text from image.")
        return text
    except Exception as e:
        logger.warning(f"Vision image extraction failed: {e}")
        return ""


def summarize_document(raw_text: str) -> dict:
    """
    Generate a concise summary and key points from document text.
    Returns dict with keys: summary (str), key_points (list of str), success (bool), message (str).
    """
    if not raw_text or not raw_text.strip():
        return {
            "success": False,
            "message": "No text to summarize.",
            "summary": "",
            "key_points": [],
        }

    text = raw_text.strip()
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS] + "\n\n[Document truncated for length.]"
        logger.info(f"Document truncated to {MAX_INPUT_CHARS} chars for summarization.")

    prompt = f"""You are an expert at summarizing long documents for government and policy use.

TASK:
1. Read the following document text.
2. Write a concise executive summary (2–4 short paragraphs) that captures the main purpose, decisions, and outcomes.
3. Extract 5–15 key points as a bullet list (short phrases or sentences).

DOCUMENT TEXT:
{text}

OUTPUT FORMAT:
Return ONLY valid JSON in this exact structure (no markdown, no code fence):
{{
  "summary": "Your executive summary here as a single string. Use \\n for paragraph breaks.",
  "key_points": [
    "First key point",
    "Second key point",
    "Third key point"
  ]
}}

RULES:
- Return ONLY the JSON object. No other text before or after.
- summary: clear, neutral, professional tone; suitable for cabinet notes and policy reports.
- key_points: actionable or informative bullets; no duplicates.
"""

    try:
        if llm_provider_is_local():
            from app.core.local_llm import local_chat_complete, local_chat_gguf_configured

            if not local_chat_gguf_configured():
                return {
                    "success": False,
                    "message": "Local chat GGUF not configured (LOCAL_CHAT_GGUF_PATH).",
                    "summary": "",
                    "key_points": [],
                }
            raw_output = local_chat_complete(
                system="You return only valid JSON as instructed. No markdown fences, no extra text.",
                user=prompt,
                max_tokens=8192,
            ).strip()
        else:
            if not settings.OPENAI_API_KEY:
                return {
                    "success": False,
                    "message": "OpenAI API key not configured (set OPENAI_API_KEY or use LLM_PROVIDER=local).",
                    "summary": "",
                    "key_points": [],
                }
            response = _cloud_openai_client().chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_output = (response.choices[0].message.content or "").strip()
        if not raw_output:
            return {
                "success": False,
                "message": "Empty response from summarization model.",
                "summary": "",
                "key_points": [],
            }

        # Parse JSON (allow wrapped in markdown code block)
        json_str = raw_output
        m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw_output, re.DOTALL)
        if m:
            json_str = m.group(1)
        else:
            match = re.search(r"(\{.*\})", raw_output, re.DOTALL)
            if match:
                json_str = match.group(1)

        data = json.loads(json_str)
        summary = data.get("summary") or ""
        key_points = data.get("key_points") or []
        if not isinstance(key_points, list):
            key_points = [str(k) for k in key_points] if key_points else []

        return {
            "success": True,
            "message": "Summary generated successfully.",
            "summary": summary,
            "key_points": key_points,
        }
    except json.JSONDecodeError as e:
        logger.warning(f"Summarization JSON parse error: {e}")
        return {
            "success": False,
            "message": "Could not parse summary response.",
            "summary": "",
            "key_points": [],
        }
    except Exception as e:
        logger.error(f"Summarization error: {e}", exc_info=True)
        return {
            "success": False,
            "message": str(e),
            "summary": "",
            "key_points": [],
        }
