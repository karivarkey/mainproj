# Copilot instructions — mainproj

Purpose
- Help AI coding agents become productive quickly in this repo (Flask + LLM translation + RAG).

Big picture (what to know first)
- Entry: [server.py](server.py) creates the app via `create_app()` in [app/main.py](app/main.py).
- Routes: API endpoints live in [app/api/routes.py](app/api/routes.py) (single blueprint `api`).
- Services: core logic lives in [app/services/*](app/services/) with clear responsibilities:
  - `llm_service.py`: downloads/loads .gguf LLMs (uses `llama_cpp.Llama`), exposes `llm_generate`, stream helpers and `SERVER_URL`.
  - `translator_service.py`: NLLB-based translation helpers; caches models under `models/translators` and exposes `translate()` used by endpoints.
  - `rag_service.py`: FAISS-backed retrieval using `sentence-transformers/paraphrase-mpnet-base-v2`.
  - `cache_service.py`: central in-memory + disk `model_cache` used to track downloaded models.

Key config and directories
- [app/config.py](app/config.py) defines `DEVICE`, `LLM_DIR`, `TRANS_DIR`, RAG files and `LANG_MAP` (short language codes → model codes). `ensure_dirs()` is called during app creation.
- Models and state live under `models/`:
  - `models/llms/` — `.gguf` model binaries
  - `models/translators/` — saved translator model folders
  - `models/rag/` — `rag.index` and `metadata.json`

Data flow and common patterns
- Primary inference flow (/infer): input -> `translate()` to English -> `rag_retrieve()` -> build prompt -> `llm_generate` or `llm_generate_stream` -> translate sentences back. See [app/api/routes.py](app/api/routes.py) for sentence-by-sentence SSE streaming logic (event types: `meta`, `sentence`, `done`, `error`).
- Streaming: SSE endpoint yields JSON lines; keep translation work per-sentence to avoid large batches.
- Error handling: look for two recurring classes of runtime issues in code:
  - Translator initialization conflicts mentioning `torchvision`/`nms` → suggest `pip install --upgrade transformers torch torchvision`.
  - LLM failures with `llama_decode returned -1` → usually means context/memory limits; change `n_ctx` when loading or reduce `max_new_tokens`.

Developer workflows and commands
- Run the server locally: `python server.py` (serves Flask on port 5005 by default).
- Install deps: `pip install -r requirements.txt` or create via `environment.yml` (conda). Heavy deps: `torch`, `transformers`, `sentence-transformers`, `faiss`, `llama-cpp-python`.
- Download LLMs:
  - Use the API: POST `/download_llm` with `{ "url":..., "name":... }` (handled by `llm_service.download_gguf`).
  - Or drop a `.gguf` into `models/llms/` (name must match `llm_service.local_gguf_path` logic).
- Translators: `translator_service.download_translator()` caches tokens and model weights into `models/translators/` (tokenizers still fetched from HF by default). Quantization mode controlled by env var `TRANSLATOR_QUANTIZE` (8bit, 4bit, or none).
- Quick translation harness: [test.py](test.py) demonstrates batch translation and model initialization for offline experiments — useful when testing translators outside Flask.
- Resource benchmarks: POST `/benchmark/resource` with `llm_name`, `prompts`, and optional `rag_data` to measure RAM/CPU/VRAM usage at each stage (baseline, LLM load, translator load, RAG load, per-prompt inference).

Project-specific conventions
- Single shared `model_cache` (in `cache_service`) records available `llms` and `translators`. Always update via `save_cache(model_cache)` when adding entries.
- Translators and LLMS use filesystem-safe names by replacing `/` with `__` (see `local_translator_path` and `local_gguf_path`).
- GPU detection: `app/config.py` and `translator_service.py` choose CUDA when available — tests should account for both CPU and GPU runs.
- Prompting: final prompts are assembled in [app/api/routes.py](app/api/routes.py). Modify there for system-level prompt changes rather than in service helpers.

Integration points and external dependencies
- Hugging Face models: `transformers` (NLLB) and `sentence-transformers` for embeddings.
- FAISS for vector search; embeddings are normalized and stored in `models/rag/rag.index`.
- `llama_cpp` (llama-cpp-python) is used to load local GGUF models.
- Network: routes assume local model binaries or downloads; `SERVER_URL` in `llm_service` is a compatibility constant.

Where to look to extend behavior
- Change prompt templates: [app/api/routes.py](app/api/routes.py) (search `final_prompt`).
- Add/modify translations: [app/services/translator_service.py](app/services/translator_service.py) (see `NLLB_MODEL` and `NLLB_LANG_MAP`).
- RAG behavior and thresholds: [app/services/rag_service.py](app/services/rag_service.py) (see `similarity_threshold` and DIM handling).
- LLM loading options and context sizes: [app/services/llm_service.py](app/services/llm_service.py) (functions `load_llm_from_gguf` / `load_llm`).

Notes for agents
- Be conservative with heavy local operations: downloading models and loading GPU models are slow and environment-dependent; prefer to change code paths or prompts first, not to re-download models during quick edits.
- When modifying endpoints, run `python server.py` and exercise via `curl` or a small test client; SSE responses are text/event-stream — use `curl -N` to view streaming output.
- Preserve existing semantic checks: endpoints return explicit 400/503 with suggestions for common failures (keep these messages intact unless improving diagnostics).

If anything here is unclear or missing, tell me what area you want expanded (e.g., more examples for SSE clients, dependency pins, or a local dev checklist).
