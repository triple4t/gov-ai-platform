"""
API Router for the Voice-Based Form Filling Service.
Exposes the core endpoints:
  - POST /api/v1/extract-voice-form : Accept audio, return extracted form entities
  - GET  /api/v1/health             : Health check with Ollama model status
"""

import logging
import asyncio
import websockets
import json
import os
import uuid
import aiofiles
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect, status, BackgroundTasks
from app.services.stt_service import stt_service
from app.services.extraction_service import extraction_service
from app.services.ocr_service import extract_text_from_file
from app.services.doc_extraction_service import extract_structured_data_with_llm
from app.services.ocr_heuristics import merge_heuristic_into_structured
from app.services.barcode_service import scan_file_for_barcodes
from app.services.summary_service import summarize_document, extract_text_from_image_with_vision
from app.db import insert_document, get_all_documents
from app.services.eval_service import evaluate_trace_realtime_sync, evaluate_trace_single_field_sync
from app.models.schemas import (
    VoiceFormResponse,
    HealthResponse,
    DocumentOCRResponse,
    ExtractedData,
    BarcodeData,
    SummarizeResponse,
    FaceVerificationResponse,
)
from app.services.face_service import face_service
from app.core.config import settings
from app.core.llm_routing import llm_provider_is_local
from app.core.local_llm import local_chat_gguf_configured, local_embedding_gguf_configured
from app.core.vlm_ocr import vlm_ocr_configured

logger = logging.getLogger(__name__)

# Document types supported for summarization (reuses OCR + docx extraction)
SUMMARIZE_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".png", ".jpg", ".jpeg", ".tiff"}

router = APIRouter(prefix="/api/v1", tags=["Voice Form Filling"])


@router.post("/extract-voice-form", response_model=VoiceFormResponse)
async def extract_voice_form(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(..., description="Audio file (wav, mp3, webm, ogg, m4a)"),
    language: str = Form("hi-IN", description="BCP-47 language code: hi-IN, mr-IN, en-IN"),
):
    """
    Main endpoint: Accept an audio file from the citizen, process the full pipeline:
      1. STT: Transcribe audio to native text
      2. Extract & Translate: Pull out the 6 form entities and translate perfectly using Llama3.2
      4. Validate: Run Pydantic regex checks on Aadhaar, PAN, Phone
      5. Return: Clean JSON payload with entities and field count
    """
    try:
        # --- Validate file format ---
        file_ext = audio.filename.split(".")[-1].lower() if audio.filename else "webm"
        if file_ext not in settings.SUPPORTED_AUDIO_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format: .{file_ext}. Supported: {settings.SUPPORTED_AUDIO_FORMATS}"
            )

        # --- Read audio bytes ---
        audio_bytes = await audio.read()
        if len(audio_bytes) > settings.MAX_AUDIO_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"Audio file exceeds maximum size of {settings.MAX_AUDIO_SIZE_MB}MB"
            )

        logger.info(f"Received audio: {audio.filename} ({len(audio_bytes)} bytes, lang={language})")

        # --- Phase 1: Speech-to-Text ---
        transcript = await stt_service.transcribe_audio(audio_bytes, file_ext, language)
        logger.info(f"Phase 1 (STT) complete: '{transcript}'")

        # --- Phase 2: Entity Extraction & Translation ---
        entities = await extraction_service.extract_entities(transcript)
        fields_filled = extraction_service.count_filled_fields(entities)
        logger.info(f"Phase 2 (Extraction) complete: {fields_filled}/6 fields filled")
        
        # Trigger background real-time evaluation
        background_tasks.add_task(evaluate_trace_realtime_sync, transcript, entities.model_dump_json())

        return VoiceFormResponse(
            success=True,
            transcript=transcript,
            translated_text=transcript, # Deprecated field but keeping for compatibility
            entities=entities,
            fields_filled=fields_filled,
        )

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        return VoiceFormResponse(success=False, error=str(e))
    except ConnectionError as e:
        logger.error(f"Connection error: {e}")
        return VoiceFormResponse(success=False, error=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in extract-voice-form: {e}", exc_info=True)
        return VoiceFormResponse(success=False, error=f"Processing failed: {str(e)}")


@router.post("/extract-text-form", response_model=VoiceFormResponse)
async def extract_text_form(
    text: str = Form(..., description="Text input (Hindi, Marathi, or English)"),
    language: str = Form("hi", description="Source language code: hi, mr, en"),
):
    """
    Alternative endpoint: Accept raw text instead of audio.
    Useful for testing the translation + extraction pipeline without audio.
    Skips Phase 1 (STT) and goes directly to Phase 2 (Translation).
    """
    try:
        logger.info(f"Received text input ({language}): '{text[:80]}...'")

        # --- Phase 1: Entity Extraction & Translation ---
        entities = await extraction_service.extract_entities(text)
        fields_filled = extraction_service.count_filled_fields(entities)

        return VoiceFormResponse(
            success=True,
            transcript=text,
            translated_text=text,
            entities=entities,
            fields_filled=fields_filled,
        )

    except Exception as e:
        logger.error(f"Error in extract-text-form: {e}", exc_info=True)
        return VoiceFormResponse(success=False, error=f"Processing failed: {str(e)}")


@router.post("/extract-single-field")
async def extract_single_field(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(..., description="Audio file"),
    language: str = Form("hi-IN", description="BCP-47 language code"),
    field_name: str = Form(..., description="Field key, e.g. 'full_name'"),
    field_description: str = Form(..., description="Human label, e.g. 'Full Name'"),
):
    """
    Guided mode endpoint: Extract a single specific field from voice input.
    Used when the UI walks the user through fields one by one.
    """
    try:
        file_ext = audio.filename.split(".")[-1].lower() if audio.filename else "webm"
        audio_bytes = await audio.read()

        if len(audio_bytes) < 1000:
            return {"success": False, "error": "Recording too short", "value": ""}

        # Phase 1: STT
        transcript = await stt_service.transcribe_audio(audio_bytes, file_ext, language)
        logger.info(f"Single-field STT ({field_name}): '{transcript}'")

        # Phase 2: Targeted single-field extraction & translation
        value = await extraction_service.extract_single_field(transcript, field_name, field_description)
        logger.info(f"Single-field Result ({field_name}): '{value}'")
        
        background_tasks.add_task(evaluate_trace_single_field_sync, transcript, field_name, value)

        return {
            "success": True,
            "field_name": field_name,
            "value": value,
            "transcript": transcript,
            "translated_text": transcript, # Keep deprecated field for frontend compat
        }

    except Exception as e:
        logger.error(f"Single-field error: {e}", exc_info=True)
        # Fallback: try to return the transcript as the value
        return {"success": False, "error": str(e), "value": ""}

@router.post("/extract-single-text-field")
async def extract_single_text_field(
    background_tasks: BackgroundTasks,
    text: str = Form(..., description="The STT transcript"),
    language: str = Form("hi-IN", description="BCP-47 language code"),
    field_name: str = Form(..., description="Field key, e.g. 'full_name'"),
    field_description: str = Form(..., description="Human label, e.g. 'Full Name'"),
):
    """
    Called after the WebSocket STT stream finishes.
    Takes the final text and extracts the single form entity.
    """
    try:
        if not text or len(text.strip()) == 0:
             return {"success": False, "error": "No text provided", "value": ""}

        logger.info(f"Extract-text-field ({field_name}): '{text}'")
        value = await extraction_service.extract_single_field(text, field_name, field_description)
        logger.info(f"Extract-text-field Result ({field_name}): '{value}'")
        
        background_tasks.add_task(evaluate_trace_single_field_sync, text, field_name, value)

        return {
            "success": True,
            "field_name": field_name,
            "value": value,
            "transcript": text,
            "translated_text": text,
        }
    except Exception as e:
        logger.error(f"Text-field error: {e}", exc_info=True)
        return {"success": False, "error": str(e), "value": ""}

@router.websocket("/stream-stt")
async def stream_stt(websocket: WebSocket, language: str = "hi"):
    """
    Real-time WebSocket connection that proxies raw audio buffers directly to Deepgram.
    Returns partial and final transcripts instantly.
    """
    await websocket.accept()

    # Map frontend language codes
    dg_language = "hi"
    if "hi" in language.lower():
        dg_language = "hi"
    elif "mr" in language.lower():
        dg_language = "mr"
    elif "en" in language.lower():
        dg_language = "en-IN"

    if not settings.DEEPGRAM_API_KEY:
        await websocket.close(code=1011, reason="Deepgram API Key not set")
        return

    # Deepgram Streaming URL (smart_format=true, interim_results=true)
    dg_url = f"wss://api.deepgram.com/v1/listen?language={dg_language}&model=nova-3&smart_format=true&interim_results=true&endpointing=300"
    
    try:
        async with websockets.connect(
            dg_url, 
            additional_headers={"Authorization": f"Token {settings.DEEPGRAM_API_KEY}"}
        ) as dg_ws:
            
            async def forward_to_deepgram():
                try:
                    while True:
                        data = await websocket.receive_bytes()
                        await dg_ws.send(data)
                except WebSocketDisconnect:
                    logger.info("Frontend WebSocket disconnected.")
                    # Send a closing message to Deepgram if needed, though closing the socket is fine
                    await dg_ws.send(json.dumps({"type": "CloseStream"}))
                except Exception as e:
                    logger.debug(f"Frontend WS error: {e}")
            
            async def forward_to_frontend():
                try:
                    while True:
                        msg = await dg_ws.recv()
                        # Pass Deepgram's native JSON straight to the frontend
                        await websocket.send_text(msg)
                except websockets.exceptions.ConnectionClosed:
                    logger.info("Deepgram WebSocket closed.")
                except Exception as e:
                    logger.debug(f"Deepgram WS error: {e}")
                    
            await asyncio.gather(forward_to_deepgram(), forward_to_frontend())

    except Exception as e:
        logger.error(f"WebSocket proxy error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint — Deepgram + LLM backend (local GGUF or Azure)."""
    try:
        dg_status = "configured" if settings.DEEPGRAM_API_KEY else "missing_key"

        if llm_provider_is_local():
            c = "ok" if local_chat_gguf_configured() else "missing"
            e = "ok" if local_embedding_gguf_configured() else "missing"
            vo = "vlm_ocr=ok" if vlm_ocr_configured() else "vlm_ocr=off"
            azure_status = f"local_llm chat={c} embed={e} {vo}"
            llm_ok = local_chat_gguf_configured()
            overall_status = "healthy" if dg_status == "configured" and llm_ok else "degraded"
        else:
            azure_status = (
                f"configured ({settings.AZURE_GPT_DEPLOYMENT})"
                if settings.AZURE_GPT_KEY and settings.AZURE_GPT_ENDPOINT
                else "missing_credentials"
            )
            overall_status = (
                "healthy"
                if dg_status == "configured" and "configured" in azure_status
                else "degraded"
            )

        return HealthResponse(
            status=overall_status,
            service="Voice Form Filling Backend",
            deepgram_status=dg_status,
            azure_status=azure_status,
        )
    except Exception as e:
        return HealthResponse(
            status="error",
            service="Voice Form Filling Backend",
            deepgram_status="unknown",
            azure_status=f"error: {str(e)}",
        )


# --- Document OCR Endpoints ---

@router.post("/document-ocr", response_model=DocumentOCRResponse)
async def process_document(file: UploadFile = File(...)):
    """
    Accepts a document (PDF or image), extracts text via OCR, 
    scans for barcodes, and parses structured data using LLM.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    allowed_exts = {".png", ".jpg", ".jpeg", ".pdf", ".tiff", ".tif"}
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format: {file_ext}. Allowed: {allowed_exts}"
        )

    file_id = str(uuid.uuid4())
    upload_path = os.path.join(settings.UPLOAD_DIR, f"{file_id}{file_ext}")

    try:
        # 1. Save file asynchronously
        async with aiofiles.open(upload_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)

        # 2. Barcode Scanning
        barcodes = []
        if file_ext in {".png", ".jpg", ".jpeg", ".tiff", ".tif"}:
            # Sync call to OpenCV-based scan
            barcodes = scan_file_for_barcodes(upload_path)

        # 3. OCR Extraction
        raw_text = extract_text_from_file(upload_path)
        _rt = raw_text or ""
        logger.info(
            "Document OCR extracted text: chars=%d preview=%r",
            len(_rt),
            _rt[:1200] if len(_rt) <= 1200 else _rt[:1200] + "…",
        )

        if not raw_text or not raw_text.strip():
            if barcodes:
                return DocumentOCRResponse(
                    success=True,
                    message=f"No text extracted, but found {len(barcodes)} barcode(s).",
                    raw_text="",
                    barcodes=barcodes
                )
            return DocumentOCRResponse(success=False, message="Failed to extract text or document is empty.")

        # 4. LLM Structured Extraction (+ regex pre-fill when flat fields are empty)
        structured_dict = extract_structured_data_with_llm(raw_text)
        structured_dict = merge_heuristic_into_structured(structured_dict, raw_text)
        extracted_data = ExtractedData(**structured_dict)

        # 5. Persist to SQLite
        doc_id = insert_document(file.filename, raw_text, structured_dict)

        return DocumentOCRResponse(
            success=True,
            message=f"Document processed (ID: {doc_id})",
            raw_text=raw_text,
            extracted_data=extracted_data,
            barcodes=barcodes
        )

    except Exception as e:
        logger.error(f"OCR Processing error: {e}", exc_info=True)
        return DocumentOCRResponse(success=False, message=f"Processing failed: {str(e)}")

    finally:
        if os.path.exists(upload_path):
            try:
                os.remove(upload_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup {upload_path}: {e}")


@router.get("/documents")
async def list_documents():
    """Retrieve all previously processed OCR documents from the database."""
    try:
        documents = get_all_documents()

        for doc in documents:
            if doc.get("other_fields"):
                try:
                    doc["other_fields"] = json.loads(doc["other_fields"])
                except Exception:
                    pass

        return {"success": True, "documents": documents}

    except Exception as e:
        logger.error(f"Error fetching documents: {e}")
        return {"success": False, "message": str(e)}


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_document_upload(file: UploadFile = File(..., description="Document to summarize (PDF, DOC, DOCX, or image)")):
    """
    Document Summarization: accept any supported document, extract text (OCR for scans/images/PDFs, direct for DOCX/DOC),
    then generate a concise summary and key points using the same LLM as OCR extraction.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in SUMMARIZE_ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format: {file_ext}. Allowed: {', '.join(sorted(SUMMARIZE_ALLOWED_EXTENSIONS))}",
        )

    file_id = str(uuid.uuid4())
    upload_path = os.path.join(settings.UPLOAD_DIR, f"summarize_{file_id}{file_ext}")

    try:
        async with aiofiles.open(upload_path, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)

        raw_text = extract_text_from_file(upload_path)

        # When image OCR fails (e.g. Tesseract not installed), try vision API fallback
        image_exts = {".png", ".jpg", ".jpeg", ".tiff"}
        if (not raw_text or not raw_text.strip()) and file_ext in image_exts:
            raw_text = extract_text_from_image_with_vision(upload_path)

        if not raw_text or not raw_text.strip():
            msg = "Could not extract any text from the document. It may be empty or in an unsupported format."
            if file_ext in image_exts:
                msg += (
                    " For images, configure local VLM OCR (LOCAL_OCR_VLM_GGUF_PATH + LOCAL_OCR_VLM_MMPROJ_PATH), "
                    "install Tesseract, set OPENAI_API_KEY for cloud vision, or use a PDF/Word file."
                )
            elif file_ext == ".doc":
                msg += " For Word documents, please use .docx or export as PDF."
            return SummarizeResponse(success=False, message=msg, raw_text_preview=None)

        result = summarize_document(raw_text)

        preview = raw_text.strip()[:500] if raw_text else None

        if not result.get("success"):
            return SummarizeResponse(
                success=False,
                message=result.get("message", "Summarization failed."),
                raw_text_preview=preview,
            )

        return SummarizeResponse(
            success=True,
            message=result.get("message", "Summary generated."),
            summary=result.get("summary"),
            key_points=result.get("key_points") or [],
            raw_text_preview=preview,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Summarize error: {e}", exc_info=True)
        return SummarizeResponse(success=False, message=str(e))
    finally:
        if os.path.exists(upload_path):
            try:
                os.remove(upload_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup {upload_path}: {e}")


# --- Face Detection Endpoints ---

@router.post("/face-verify", response_model=FaceVerificationResponse)
async def verify_face(
    doc_image: UploadFile = File(..., description="Reference document photo (Aadhaar/PAN)"),
    capture_image: UploadFile = File(..., description="Captured live selfie")
):
    """
    Exhaustive parity port of Face Verification & Liveness logic.
    Compares a reference ID photo with a captured selfie using MSE and Eye Detection.
    """
    doc_id = str(uuid.uuid4())
    cap_id = str(uuid.uuid4())
    
    doc_ext = os.path.splitext(doc_image.filename)[1] if doc_image.filename else ".jpg"
    cap_ext = os.path.splitext(capture_image.filename)[1] if capture_image.filename else ".jpg"
    
    doc_path = os.path.join(settings.UPLOAD_DIR, f"face_ref_{doc_id}{doc_ext}")
    cap_path = os.path.join(settings.UPLOAD_DIR, f"face_cap_{cap_id}{cap_ext}")
    
    try:
        # Save reference image asynchronously
        async with aiofiles.open(doc_path, 'wb') as out_file:
            content = await doc_image.read()
            await out_file.write(content)
            
        # Save capture image asynchronously
        async with aiofiles.open(cap_path, 'wb') as out_file:
            content = await capture_image.read()
            await out_file.write(content)
            
        # 1. Face Matching (MSE based)
        match_result = face_service.verify_faces(doc_path, cap_path)
        
        # 2. Liveness Detection (Eye detection based)
        liveness_result = face_service.detect_liveness(cap_path)
        
        # Final Decision (must pass both match and liveness)
        verified = match_result.get("verified", False) and liveness_result.get("is_live", False)
        
        return FaceVerificationResponse(
            success=True,
            status="success",
            verification_result="Verified" if verified else "Not Verified",
            match_score=round(match_result.get("similarity_score", 0) * 100, 2),
            liveness_status="Live" if liveness_result.get("is_live", False) else "Spoof/Not Detected",
            liveness_details=liveness_result,
            match_details=match_result
        )
        
    except Exception as e:
        logger.error(f"Face Verification execution error: {e}", exc_info=True)
        return FaceVerificationResponse(success=False, error=str(e))
    finally:
        # Exhaustive Cleanup to prevent disk bloat
        for path in [doc_path, cap_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as cleanup_err:
                    logger.warning(f"Cleanup failed for {path}: {cleanup_err}")
