# Government AI — Backend

## Dependencies

**Policy:** Any new Python package must be added to [`requirements.txt`](requirements.txt) with a pinned version, then installed into the project virtual environment (never rely on global `pip` only).

## Setup (virtual environment)

### Windows (PowerShell)

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run the API server

With the venv **activated**, use **no `--reload`** by default (stable; PRD ZIP extract writes under `uploads/`, which can otherwise restart the server mid-request and skip Neo4j sync):

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Windows (PowerShell)

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Optional: auto-reload while editing Python

Only if you need reload on code changes:

- **bash:**  
  `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --reload-exclude uploads`
- **PowerShell:** use **single quotes** so `*` is not expanded — never use `"uploads/**/*"` in double quotes:  
  `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --reload-exclude 'uploads'`

Copy `.env.example` to `.env` and fill in API keys before running features that call external services.

## Local GGUF models (Qwen chat + Jina embeddings)

By default `LLM_PROVIDER=local`: conversational chat (`/api/v1/chat/*`), PRD RAG embeddings, PRD generation, and code summarizer LLM calls use [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) with two `.gguf` files.

1. Set **absolute paths** in `.env`:
   - `LOCAL_CHAT_GGUF_PATH` — e.g. Qwen3.5 chat GGUF  
   - `LOCAL_EMBEDDING_GGUF_PATH` — e.g. Jina embeddings v4 retrieval GGUF  
   Hugging Face cache (Windows): `%USERPROFILE%\.cache\huggingface\hub\...\<snapshot>\*.gguf`. List files with:  
   `Get-ChildItem -Path "$env:USERPROFILE\.cache\huggingface\hub" -Recurse -Filter "*.gguf"`  
   Download a file into that cache with the Hub CLI from your venv: **`.\venv\Scripts\hf.exe download org/repo file.gguf`** (the old `huggingface-cli` name is not used by current `huggingface_hub`).

2. **VRAM:** Loading both models can exceed GPU memory. Use `LLAMA_N_GPU_LAYERS` / `LLAMA_N_GPU_LAYERS_EMBED` (e.g. split layers between GPU and CPU, or `0` for CPU offload). See `.env.example`.

3. **GPU for all local models:** Use `-1` for full layer offload on each stack (defaults in code are already `-1`). Set explicitly in `.env` so nothing is left implicit:
   - `LLAMA_N_GPU_LAYERS` / `LLAMA_N_GPU_LAYERS_EMBED` / `LLAMA_MAIN_GPU` — PRD chat + Jina embeddings + conversational local LLM  
   - `LOCAL_OCR_VLM_N_GPU_LAYERS` — document / image VLM OCR  
   - `RAG_LLM_N_GPU_LAYERS` / `JINA_EMBED_N_GPU_LAYERS` — Advanced RAG (`/api/v1/rag/*`) second in-process handles  
   Offload only works if **`llama-cpp-python` is built or installed with CUDA**; a CPU-only wheel will still run on CPU (you may see `CLIP using CPU backend` in logs).

4. **Re-index PRD projects:** FAISS indexes are tied to embedding dimension. If you previously used Azure `text-embedding-3-*`, **delete old projects or re-upload ZIPs** after switching to Jina (dimension is detected on index build).

5. **CUDA on Windows:** Pre-built **Linux** CUDA wheels exist for recent `llama-cpp-python` versions; **Windows** often gets a **CPU-only** wheel from PyPI, so `n_gpu_layers=-1` still runs on CPU. With **CUDA Toolkit** + **VS 2022 C++ build tools** installed, rebuild GPU bindings from the backend folder:

   ```powershell
   cd backend
   .\scripts\install_llama_cuda_windows.ps1
   ```

   Then restart the API and confirm startup does **not** warn that GPU offload is unavailable; `python -c "from llama_cpp import llama_supports_gpu_offload; print(llama_supports_gpu_offload())"` should print `True`.

   **Linux** can use the same version with a CUDA wheel, for example:  
   `pip install llama-cpp-python==0.3.19 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124`  
   (pick `cu121`–`cu125` closest to your toolkit; driver 12.6 is backward compatible with cu124-style runtimes.)

6. **First load looks “stuck”:** A 7B+ GGUF can take **several minutes** to mmap with little or no log output after `llama_context: …`. Leave it running, or watch for `Chat GGUF loaded in … s`. By default **`LOCAL_LLM_WARMUP_ON_STARTUP=true`** loads the chat model during API startup (you’ll wait before “Application startup complete”). Set `LOCAL_LLM_WARMUP_ON_STARTUP=false` for a fast boot and accept a long first request instead.

7. **Conversational chat on CPU:** If the startup log warns that **GPU offload is not available**, a **9B** model can take **many minutes** per reply (normal for CPU). Options: install a **CUDA** build of `llama-cpp-python`, use a **smaller / more quantized** GGUF for `LOCAL_CHAT_GGUF_PATH`, configure **Azure** (`LLM_PROVIDER=azure`), or tune **`LOCAL_CHAT_MAX_TOKENS`**, **`LOCAL_CHAT_CPU_HISTORY_MESSAGES`**, **`LOCAL_CHAT_CPU_MSG_CHAR_CAP`**, **`LLAMA_N_BATCH`**, and **`LLAMA_THREADS`** (see `.env.example`).

To use Azure for chat + embeddings again, set `LLM_PROVIDER=azure` and configure the Azure variables in `.env`.

## Advanced RAG (`/api/v1/rag/*` and `/api/v1/prd-platform/rag/*`)

The **canonical** implementation is [`app/advanced_rag/rag_engine.py`](app/advanced_rag/rag_engine.py) (with fallbacks to `LOCAL_CHAT_GGUF_PATH` / `LOCAL_EMBEDDING_GGUF_PATH`). The nested file [`backend/backend/rag_engine.py`](backend/backend/rag_engine.py) is a legacy copy only.

The lean `app/main.py` mounts the same router **twice**: under `/api/v1/rag` and under `/api/v1/prd-platform/rag` so the PRD UI “Hybrid RAG” tab can call the latter. Install deps if missing: `pip install rank-bm25 chromadb langgraph langchain-text-splitters`.

- **Ingest:** `POST /api/v1/rag/ingest` (or `.../prd-platform/rag/ingest`) with OCR- or PDF-extracted text (after `POST /api/v1/document-ocr` you can send `raw_text` here).
- **Query:** `POST /api/v1/rag/query` — LangGraph flow: BM25 + Chroma (Jina `Query:`/`Passage:` embeddings) + RRF, chunk grading, answer generation.
- **Artifacts:** `POST /api/v1/rag/generate-artifact` — PRD, diagrams, SOP, etc.
- **Projects:** `POST /api/v1/rag/project/append`, `POST /api/v1/rag/project/ingest-batch` — stack many sources under one `project_id`.

By default **`RAG_LLM_GGUF_PATH`** and **`JINA_EMBED_LOCAL_PATH`** fall back to **`LOCAL_CHAT_GGUF_PATH`** and **`LOCAL_EMBEDDING_GGUF_PATH`**. You can still override with RAG-specific env vars (see `.env.example`).

**Note:** This path uses a **second** in-process Qwen GGUF handle for RAG (separate from `get_chat_llama()`), so RAM usage is higher than conversational chat alone.

### OCR (root backend)

Document OCR uses [`app/services/ocr_service.py`](app/services/ocr_service.py): **local VLM** (`LOCAL_OCR_VLM_*` + `Qwen25VLChatHandler`) when configured, else **Tesseract** — aligned with the LightOnOCR-style flow in `backend/backend/run_ocr.py`, without running the nested `app.py` server.
