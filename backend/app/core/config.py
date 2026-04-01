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

    # --- Local GGUF (llama-cpp-python): chat + embeddings ---
    # LLM_PROVIDER=local uses LOCAL_*_GGUF_PATH for PRD/chat/code LLM and embeddings.
    # LLM_PROVIDER=azure uses Azure OpenAI for chat + Azure embeddings (previous behavior).
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "local").strip().lower()
    LOCAL_CHAT_GGUF_PATH: str = os.getenv("LOCAL_CHAT_GGUF_PATH", "").strip()
    LOCAL_EMBEDDING_GGUF_PATH: str = os.getenv("LOCAL_EMBEDDING_GGUF_PATH", "").strip()
    # Jina v4 text-retrieval GGUF: Query:/Passage: + normalize (same as Advanced RAG). auto = on if path has jina+retrieval.
    LOCAL_EMBEDDING_JINA_PREFIXES: str = os.getenv("LOCAL_EMBEDDING_JINA_PREFIXES", "auto").strip().lower()
    LLAMA_N_CTX: int = int(os.getenv("LLAMA_N_CTX", "8192"))
    LLAMA_EMBED_N_CTX: int = int(os.getenv("LLAMA_EMBED_N_CTX", "8192"))
    LLAMA_N_GPU_LAYERS: int = int(os.getenv("LLAMA_N_GPU_LAYERS", "-1"))
    # Default 0: embedding load shares VRAM with chat + VLM OCR; full GPU (-1) often hits "Failed to create llama_context".
    LLAMA_N_GPU_LAYERS_EMBED: int = int(os.getenv("LLAMA_N_GPU_LAYERS_EMBED", "0"))
    # If embed load fails at LLAMA_N_GPU_LAYERS_EMBED (e.g. -1 OOM), try this many GPU layers before CPU (0 = skip partial step).
    LLAMA_N_GPU_LAYERS_EMBED_PARTIAL: int = int(os.getenv("LLAMA_N_GPU_LAYERS_EMBED_PARTIAL", "32"))
    LLAMA_MAIN_GPU: int = int(os.getenv("LLAMA_MAIN_GPU", "0"))
    # 0 = use most CPU cores (capped) for llama.cpp; set e.g. 8 to fix thread count.
    LLAMA_THREADS: int = int(os.getenv("LLAMA_THREADS", "0"))
    # Prompt processing batch size (larger often helps throughput on CPU; try 512–1024).
    LLAMA_N_BATCH: int = int(os.getenv("LLAMA_N_BATCH", "512"))
    # e.g. qwen, chatml — leave empty for llama.cpp auto-detect from GGUF metadata
    LLAMA_CHAT_FORMAT: str = os.getenv("LLAMA_CHAT_FORMAT", "").strip()
    # Lower default keeps conversational UI responsive on local CPU; raise for longer answers.
    LOCAL_CHAT_MAX_TOKENS: int = int(os.getenv("LOCAL_CHAT_MAX_TOKENS", "384"))
    # When LLM_PROVIDER=local only: cap DB history length (fewer prompt tokens = faster on CPU).
    LOCAL_CHAT_CPU_HISTORY_MESSAGES: int = int(os.getenv("LOCAL_CHAT_CPU_HISTORY_MESSAGES", "8"))
    # When LLM_PROVIDER=local: truncate each prior user/assistant message to this many chars (0 = off).
    LOCAL_CHAT_CPU_MSG_CHAR_CAP: int = int(os.getenv("LOCAL_CHAT_CPU_MSG_CHAR_CAP", "2000"))
    # Cap for PRD / long markdown generations when LLM_PROVIDER=local (8192 is very slow on CPU).
    # Set to 0 to disable capping (use full requested max_tokens from callers).
    LOCAL_DOC_GENERATION_MAX_TOKENS: int = int(os.getenv("LOCAL_DOC_GENERATION_MAX_TOKENS", "4096"))
    # Set LLAMA_VERBOSE=1 to print llama.cpp load diagnostics when a GGUF fails to load.
    LLAMA_VERBOSE: bool = os.getenv("LLAMA_VERBOSE", "").strip().lower() in ("1", "true", "yes", "on")
    # Preload chat GGUF during API startup so the first HTTP request is not blocked for many minutes.
    LOCAL_LLM_WARMUP_ON_STARTUP: bool = os.getenv(
        "LOCAL_LLM_WARMUP_ON_STARTUP", "true"
    ).strip().lower() in ("1", "true", "yes", "on")
    # Also preload embedding GGUF at startup (uses extra RAM; disable if you only need chat on boot).
    LOCAL_LLM_WARMUP_EMBED_ON_STARTUP: bool = os.getenv(
        "LOCAL_LLM_WARMUP_EMBED_ON_STARTUP", "false"
    ).strip().lower() in ("1", "true", "yes", "on")
    # Preload VLM OCR GGUF + mmproj at startup (extra RAM; first OCR request faster).
    LOCAL_LLM_WARMUP_OCR_VLM_ON_STARTUP: bool = os.getenv(
        "LOCAL_LLM_WARMUP_OCR_VLM_ON_STARTUP", "false"
    ).strip().lower() in ("1", "true", "yes", "on")

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
    # Raster OCR for images / scanned PDF pages: auto = VLM if LOCAL_OCR_VLM_* set, else Tesseract; vlm | tesseract to force.
    OCR_RASTER_BACKEND: str = os.getenv("OCR_RASTER_BACKEND", "auto").strip().lower()
    # Local VLM OCR (llama-cpp + mmproj): LightOnOCR / Qwen2.5-VL use qwen25_vl; ggml-org/GLM-OCR-GGUF uses glm_ocr.
    LOCAL_OCR_VLM_GGUF_PATH: str = os.getenv("LOCAL_OCR_VLM_GGUF_PATH", "").strip()
    LOCAL_OCR_VLM_MMPROJ_PATH: str = os.getenv("LOCAL_OCR_VLM_MMPROJ_PATH", "").strip()
    # auto = glm_ocr if main GGUF path contains "glm-ocr", else qwen25_vl (Qwen25VLChatHandler).
    LOCAL_OCR_VLM_CHAT_HANDLER: str = os.getenv("LOCAL_OCR_VLM_CHAT_HANDLER", "auto").strip().lower()
    # Vision eats context: 8192 leaves room after large image token grids (see backend/backend lighton_ocr_image).
    LOCAL_OCR_VLM_N_CTX: int = int(os.getenv("LOCAL_OCR_VLM_N_CTX", "8192"))
    LOCAL_OCR_VLM_MAX_TOKENS: int = int(os.getenv("LOCAL_OCR_VLM_MAX_TOKENS", "4096"))
    LOCAL_OCR_VLM_N_GPU_LAYERS: int = int(os.getenv("LOCAL_OCR_VLM_N_GPU_LAYERS", "-1"))
    # Longest side in pixels before VLM encode (cuts token count / CPU time; 0 = no limit).
    # Default 1024 balances token count vs legibility for local VLMs (was 1536; huge sides → many image tokens).
    LOCAL_OCR_VLM_MAX_IMAGE_SIDE: int = int(os.getenv("LOCAL_OCR_VLM_MAX_IMAGE_SIDE", "768"))
    # When VLM output is garbage and Tesseract is missing, retry once with this max side (0 = off). Smaller → fewer image tokens.
    LOCAL_OCR_VLM_GARBAGE_RETRY_MAX_SIDE: int = int(os.getenv("LOCAL_OCR_VLM_GARBAGE_RETRY_MAX_SIDE", "512"))
    # Comma-separated max sides to try (default 512,384,256). Vision grid may stay large in llama.cpp; empty/"off" = no side retries.
    LOCAL_OCR_VLM_GARBAGE_RETRY_SIDES: str = os.getenv("LOCAL_OCR_VLM_GARBAGE_RETRY_SIDES", "512,384,256").strip()
    # Garbage retries: cap decode length (reduces "!" spam to max_tokens); 0 = use LOCAL_OCR_VLM_MAX_TOKENS.
    LOCAL_OCR_VLM_GARBAGE_RETRY_MAX_TOKENS: int = int(os.getenv("LOCAL_OCR_VLM_GARBAGE_RETRY_MAX_TOKENS", "640"))
    LOCAL_OCR_VLM_GARBAGE_RETRY_REPEAT_PENALTY: float = float(
        os.getenv("LOCAL_OCR_VLM_GARBAGE_RETRY_REPEAT_PENALTY", "1.35")
    )
    LOCAL_OCR_VLM_GARBAGE_RETRY_USER_ONLY: bool = os.getenv(
        "LOCAL_OCR_VLM_GARBAGE_RETRY_USER_ONLY", "true"
    ).strip().lower() in ("1", "true", "yes", "on")
    # jpeg_fitted: always PIL→JPEG resize (stable; avoids huge native grids). native_if_fits: raw file when within max side.
    LOCAL_OCR_VLM_DATA_URI_MODE: str = os.getenv("LOCAL_OCR_VLM_DATA_URI_MODE", "jpeg_fitted").strip().lower()
    # Match reference OCR: user-only message unless explicitly enabled.
    LOCAL_OCR_VLM_INCLUDE_SYSTEM: bool = os.getenv("LOCAL_OCR_VLM_INCLUDE_SYSTEM", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    LOCAL_OCR_VLM_TEMPERATURE: float = float(os.getenv("LOCAL_OCR_VLM_TEMPERATURE", "0"))
    LOCAL_OCR_VLM_REPEAT_PENALTY: float = float(os.getenv("LOCAL_OCR_VLM_REPEAT_PENALTY", "1.15"))
    # Llama.cpp tensor dump on load/inference — off by default (independent of LLAMA_VERBOSE).
    LOCAL_OCR_VLM_VERBOSE: bool = os.getenv("LOCAL_OCR_VLM_VERBOSE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    # Redirect OS stderr (fd 2) during VLM load/inference so native llama.cpp add_text/clip logs are hidden.
    # Opt-in: serializes VLM calls with a lock while fd 2 is redirected (safe for single-worker dev).
    LOCAL_OCR_VLM_SILENCE_STDERR_FD: bool = os.getenv(
        "LOCAL_OCR_VLM_SILENCE_STDERR_FD", ""
    ).strip().lower() in ("1", "true", "yes", "on")
    LOCAL_OCR_VLM_SYSTEM_PROMPT: str = os.getenv("LOCAL_OCR_VLM_SYSTEM_PROMPT", "").strip()
    LOCAL_OCR_VLM_USER_PROMPT: str = os.getenv("LOCAL_OCR_VLM_USER_PROMPT", "").strip()

    # --- MLflow Integrations ---
    # Set MLFLOW_ENABLED=false to skip all MLflow/OpenTelemetry autolog startup (no tracking server needed).
    MLFLOW_ENABLED: bool = os.getenv("MLFLOW_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
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

    # --- PRD Platform (RAG + FAISS + SQLite) ---
    AZURE_OPENAI_EMBEDDING_API_KEY: str = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY", "")
    AZURE_OPENAI_EMBEDDING_ENDPOINT: str = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT", "").rstrip("/")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "")
    AZURE_OPENAI_EMBEDDING_API_VERSION: str = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION", "2024-02-01")

    AZURE_GPT_51_KEY: str = os.getenv("AZURE_GPT_51_KEY", "")
    AZURE_GPT_51_ENDPOINT: str = (os.getenv("AZURE_GPT_51_ENDPOINT") or "").rstrip("/")
    AZURE_GPT_51_DEPLOYMENT: str = os.getenv("AZURE_GPT_51_DEPLOYMENT", "")
    AZURE_GPT_51_API_VERSION: str = os.getenv("AZURE_GPT_51_API_VERSION", "2025-04-01-preview")
    # Codex (gpt-5.x-codex) uses the Responses API on Azure, not chat.completions.
    AZURE_GPT_51_RESPONSES_API_VERSION: str = os.getenv("AZURE_GPT_51_RESPONSES_API_VERSION", "preview")

    PRD_PLATFORM_DB_FILE: str = os.getenv("PRD_PLATFORM_DB_FILE", "prd_platform.db")
    PRD_PLATFORM_MAX_ZIP_MB: int = int(os.getenv("PRD_PLATFORM_MAX_ZIP_MB", "50"))
    PRD_PLATFORM_MAX_FILES: int = int(os.getenv("PRD_PLATFORM_MAX_FILES", "500"))
    PRD_PLATFORM_MAX_FILE_BYTES: int = int(os.getenv("PRD_PLATFORM_MAX_FILE_BYTES", str(256 * 1024)))
    PRD_PLATFORM_CHUNK_CHARS: int = int(os.getenv("PRD_PLATFORM_CHUNK_CHARS", "1500"))
    PRD_PLATFORM_CHUNK_OVERLAP: int = int(os.getenv("PRD_PLATFORM_CHUNK_OVERLAP", "200"))
    PRD_PLATFORM_RAG_TOP_K: int = int(os.getenv("PRD_PLATFORM_RAG_TOP_K", "18"))
    PRD_PLATFORM_RAG_MAX_CHARS: int = int(os.getenv("PRD_PLATFORM_RAG_MAX_CHARS", str(24_000)))
    # Deep RAG for tech_docs / architecture / code_review (multi-query + per-file cap)
    PRD_TECH_DOCS_RAG_TOP_K_PER_QUERY: int = int(os.getenv("PRD_TECH_DOCS_RAG_TOP_K_PER_QUERY", "12"))
    PRD_TECH_DOCS_RAG_MAX_CHARS: int = int(os.getenv("PRD_TECH_DOCS_RAG_MAX_CHARS", str(32_000)))
    PRD_TECH_DOCS_MAX_CHUNKS_PER_FILE: int = int(os.getenv("PRD_TECH_DOCS_MAX_CHUNKS_PER_FILE", "3"))
    PRD_TECH_DOCS_PRIORITY_MAX_CHUNKS: int = int(os.getenv("PRD_TECH_DOCS_PRIORITY_MAX_CHUNKS", "24"))
    # Max characters for README/manifest bootstrap before vector context fills the rest.
    PRD_TECH_DOCS_PRIORITY_MAX_CHARS: int = int(os.getenv("PRD_TECH_DOCS_PRIORITY_MAX_CHARS", str(12_000)))
    PRD_EMBEDDING_DIM_DEFAULT: int = int(os.getenv("PRD_EMBEDDING_DIM", "1536"))
    PRD_PLATFORM_RETENTION_DAYS: int = int(os.getenv("PRD_PLATFORM_RETENTION_DAYS", "7"))

    # --- Neo4j (optional: import graph from PRD ZIP / CDG sync) ---
    NEO4J_ENABLED: bool = os.getenv("NEO4J_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")
    NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "").strip()
    NEO4J_SYNC_FAIL_OPEN: bool = os.getenv("NEO4J_SYNC_FAIL_OPEN", "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    NEO4J_MAX_GRAPH_NODES: int = int(os.getenv("NEO4J_MAX_GRAPH_NODES", "0"))
    NEO4J_MAX_GRAPH_EDGES: int = int(os.getenv("NEO4J_MAX_GRAPH_EDGES", "0"))

    # --- Code Summarizer (RAG reuses PRD index + embeddings) ---
    CODE_SUMMARIZER_MAX_INPUT_CHARS: int = int(os.getenv("CODE_SUMMARIZER_MAX_INPUT_CHARS", str(120_000)))
    # Reject bodies larger than this (bytes/chars) before processing.
    CODE_SUMMARIZER_HARD_MAX_CHARS: int = int(os.getenv("CODE_SUMMARIZER_HARD_MAX_CHARS", str(500_000)))
    CODE_SUMMARIZER_RAG_TOP_K: int = int(os.getenv("CODE_SUMMARIZER_RAG_TOP_K", "24"))
    CODE_SUMMARIZER_RAG_MAX_CHARS: int = int(os.getenv("CODE_SUMMARIZER_RAG_MAX_CHARS", str(28_000)))
    CODE_SUMMARY_CACHE_ENABLED: bool = os.getenv("CODE_SUMMARY_CACHE_ENABLED", "false").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    CODE_SUMMARY_CACHE_TTL_SECONDS: int = int(os.getenv("CODE_SUMMARY_CACHE_TTL_SECONDS", "600"))
    # Local GGUF context: reserve tokens for chat template + generation (body is capped to fit LLAMA_N_CTX).
    CODE_SUMMARIZER_LOCAL_CTX_RESERVE_TOKENS: int = int(
        os.getenv("CODE_SUMMARIZER_LOCAL_CTX_RESERVE_TOKENS", "2800")
    )
    CODE_SUMMARIZER_LOCAL_CHARS_PER_TOKEN: float = float(
        os.getenv("CODE_SUMMARIZER_LOCAL_CHARS_PER_TOKEN", "3.2")
    )
    # Optional Redis for distributed cache (not required for v1).
    REDIS_URL: str = os.getenv("REDIS_URL", "").strip()


settings = Settings()
