"""
Voice-Based Form Filling & Navigation - Backend Service
========================================================
A standalone FastAPI microservice (Backend-as-a-Service) that:
  1. Accepts citizen voice input (audio files)
  2. Transcribes audio to text (STT)
  3. Translates Hindi/Marathi to English (translategemma:4b via Ollama)
  4. Extracts 6 structured form entities (Name, Phone, PAN, Age, Aadhaar, Address)
  5. Validates and returns clean JSON

Designed for seamless integration into a larger main project backend.
"""

import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.core.config import settings
from app.db import init_db
from app.conversational.database import init_conversational_db
from app.conversational.auth import router as auth_router
from app.conversational.chat import router as chat_router
from app.conversational.voice import router as voice_router
from app.conversational.documents import router as documents_router
from app.conversational.integrations import router as integrations_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Create the FastAPI Application ---
app = FastAPI(
    title="Voice Form Filling Service",
    description=(
        "A standalone Backend-as-a-Service for extracting structured form data "
        "from citizen voice input in Hindi, Marathi, and English. "
        "Supports Aadhaar, PAN Card, and other Indian government document formats."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Register API Routers ---
app.include_router(router)
app.include_router(auth_router, prefix="/api/v1/auth", tags=["Conversational AI - Auth"])
app.include_router(chat_router, prefix="/api/v1/chat", tags=["Conversational AI - Chat"])
app.include_router(voice_router, prefix="/api/v1/voice", tags=["Conversational AI - Voice"])
app.include_router(documents_router, prefix="/api/v1/documents", tags=["Conversational AI - Documents"])
app.include_router(integrations_router, prefix="/api/v1/integrations", tags=["Conversational AI - Integrations"])


@app.get("/")
async def root():
    """Root endpoint - service information."""
    return {
        "service": "Unified Voice & Document OCR Backend",
        "version": "1.1.0",
        "docs": "/docs",
        "endpoints": {
            "extract_voice_form": "POST /api/v1/extract-voice-form",
            "document_ocr": "POST /api/v1/document-ocr",
            "summarize": "POST /api/v1/summarize",
            "list_documents": "GET /api/v1/documents",
            "health": "GET /api/v1/health",
            "auth_register": "POST /api/v1/auth/register",
            "auth_login": "POST /api/v1/auth/login",
            "chat_send": "POST /api/v1/chat/send",
            "voice_stt": "POST /api/v1/voice/speech-to-text",
            "voice_tts": "POST /api/v1/voice/text-to-speech",
            "documents_upload": "POST /api/v1/documents/upload",
            "integrations": "GET /api/v1/integrations/services/list",
        },
    }


@app.on_event("startup")
async def startup_event():
    """Log service startup information and initialize components."""
    init_db()
    await init_conversational_db()
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    logger.info("=" * 60)
    logger.info("  Unified Voice & OCR Backend Service Starting...")
    logger.info(f"  Deepgram Key:      {'Configured' if settings.DEEPGRAM_API_KEY else 'Missing'}")
    logger.info(f"  Azure GPT:         {settings.AZURE_GPT_DEPLOYMENT} ({'Configured' if settings.AZURE_GPT_KEY else 'Missing'})")
    logger.info(f"  OpenAI (OCR):      {settings.OPENAI_MODEL} ({'Configured' if settings.OPENAI_API_KEY else 'Missing'})")
    logger.info(f"  Conversational AI: Auth, Chat, Voice, Documents, Integrations")
    logger.info(f"  CORS Origins:      {settings.CORS_ORIGINS}")
    logger.info("=" * 60)
