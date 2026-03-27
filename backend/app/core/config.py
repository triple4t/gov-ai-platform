"""
Configuration module for the Voice Form Filling & Conversational AI Backend.
Loads environment variables and exposes them as typed settings.
Single .env for all features; names aligned across services.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _split_list(value: str, default: list[str] | None = None) -> list[str]:
    if not value:
        return default or []
    return [x.strip() for x in value.split(",") if x.strip()]


class Settings:
    """Application settings loaded from environment variables."""

    # --- Voice Form Filling & OCR (existing) ---
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_EXTRACT_MODEL: str = os.getenv("OLLAMA_EXTRACT_MODEL", "llama3.2:1b")
    OLLAMA_TRANSLATE_MODEL: str = os.getenv("OLLAMA_TRANSLATE_MODEL", "translategemma:4b")

    AZURE_GPT_ENDPOINT: str = os.getenv("AZURE_GPT_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_GPT_KEY: str = os.getenv("AZURE_GPT_KEY") or os.getenv("AZURE_OPENAI_KEY", "")
    AZURE_GPT_DEPLOYMENT: str = os.getenv("AZURE_GPT_DEPLOYMENT") or os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
    AZURE_GPT_API_VERSION: str = os.getenv("AZURE_GPT_API_VERSION", os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"))

    CORS_ORIGINS: list[str] = _split_list(os.getenv("CORS_ORIGINS"), ["http://localhost:5173", "http://localhost:3000"])

    DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "")

    SUPPORTED_AUDIO_FORMATS: list[str] = ["wav", "mp3", "webm", "ogg", "m4a"]
    MAX_AUDIO_SIZE_MB: int = 25

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")

    OCR_DB_FILE: str = os.getenv("OCR_DB_FILE", "ocr_documents.db")
    TESSERACT_CMD: str = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")

    # --- MLflow Integrations ---
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "Government AI Backend")

    # --- Conversational AI (integrated) ---
    CONVERSATIONAL_DATABASE_URL: str = os.getenv("CONVERSATIONAL_DATABASE_URL", "sqlite+aiosqlite:///./citizen_assistant.db")

    SECRET_KEY: str = os.getenv("SECRET_KEY", "gov-ai-secret-key-change-in-production")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

    SUPPORTED_LANGUAGES: list[str] = _split_list(
        os.getenv("SUPPORTED_LANGUAGES"),
        ["hi", "en", "bn", "te", "mr", "ta", "ur", "gu", "kn", "or", "pa", "as", "ml"],
    )

    SARVAM_API_KEY: str = os.getenv("SARVAM_API_KEY", "")

    UMANG_API_KEY: str = os.getenv("UMANG_API_KEY", "")
    DIGILOCKER_API_KEY: str = os.getenv("DIGILOCKER_API_KEY", "")
    DIGILOCKER_CLIENT_ID: str = os.getenv("DIGILOCKER_CLIENT_ID", "")

    MAX_UPLOAD_SIZE: int = int(os.getenv("MAX_UPLOAD_SIZE", str(10 * 1024 * 1024)))
    ALLOWED_DOC_EXTENSIONS: list[str] = _split_list(
        os.getenv("ALLOWED_EXTENSIONS"),
        [".pdf", ".docx", ".txt", ".xlsx"],
    )


settings = Settings()
