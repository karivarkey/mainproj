import os
from flask import Flask
from flask_cors import CORS
from app.config import ensure_dirs
# Changed from app.api.routes to just app.api
from app.api import bp as api_bp 


def _should_run_startup_preload() -> bool:
    enabled = os.environ.get("STARTUP_PRELOAD_ENABLED", "true").strip().lower() in ("1", "true", "yes")
    if not enabled:
        return False

    # Avoid loading models in the Werkzeug reloader parent process.
    run_main = os.environ.get("WERKZEUG_RUN_MAIN")
    if run_main is None:
        return True
    return run_main.lower() == "true"


def _pick_startup_llm_name() -> str | None:
    from app.services.llm_service import list_all_llms

    available = list_all_llms()
    if not available:
        return None

    requested = (os.environ.get("STARTUP_LLM_NAME") or "qwen2.5").strip()
    requested_lower = requested.lower()

    for candidate in available:
        if candidate.lower() == requested_lower:
            return candidate

    for candidate in available:
        if requested_lower and requested_lower in candidate.lower():
            return candidate

    for candidate in available:
        if "qwen2.5" in candidate.lower():
            return candidate

    return available[0]


def _run_startup_preload() -> None:
    strict = os.environ.get("STARTUP_PRELOAD_STRICT", "true").strip().lower() in ("1", "true", "yes")
    errors = []

    onnx_family = (os.environ.get("STARTUP_ONNX_FAMILY") or "nllb").strip().lower()
    if onnx_family not in ("m2m", "nllb"):
        onnx_family = "nllb"

    preload_onnx = os.environ.get("STARTUP_PRELOAD_ONNX", "false").strip().lower() in ("1", "true", "yes")
    preload_nllb = os.environ.get("STARTUP_PRELOAD_NLLB", "true").strip().lower() in ("1", "true", "yes")
    preload_rag = os.environ.get("STARTUP_PRELOAD_RAG", "true").strip().lower() in ("1", "true", "yes")
    preload_llm = os.environ.get("STARTUP_PRELOAD_LLM", "true").strip().lower() in ("1", "true", "yes")

    if preload_onnx:
        try:
            from app.services.onnx_model_download_service import ensure_default_onnx_models, ensure_onnx_tokenizer
            from app.services.onnx_translator_service import preload_onnx_translator

            ensure_default_onnx_models(force_download=False, family=onnx_family)
            ensure_onnx_tokenizer(force_download=False, family=onnx_family)
            preload_onnx_translator(onnx_family=onnx_family)
            print(f"[STARTUP] ONNX translator preloaded for family='{onnx_family}'")
        except Exception as exc:
            errors.append(f"onnx:{exc}")
            print(f"[STARTUP] ONNX preload failed: {exc}")

    if preload_nllb:
        try:
            from app.services.translator_service import preload_translator

            preload_translator()
            print("[STARTUP] NLLB translator preloaded")
        except Exception as exc:
            errors.append(f"nllb:{exc}")
            print(f"[STARTUP] NLLB preload failed: {exc}")

    if preload_rag:
        try:
            from app.services.rag_backend import ensure_active_backend_loaded
            from app.services.rag_service import get_embed_model

            ensure_active_backend_loaded()
            get_embed_model()
            print("[STARTUP] RAG backend and embedding model preloaded")
        except Exception as exc:
            errors.append(f"rag:{exc}")
            print(f"[STARTUP] RAG preload failed: {exc}")

    if preload_llm:
        try:
            from app.services.llm_service import load_llm_from_gguf

            llm_name = _pick_startup_llm_name()
            if not llm_name:
                raise RuntimeError("No local GGUF models found in models/llms")

            n_ctx = int(os.environ.get("STARTUP_LLM_N_CTX", "4096"))
            n_gpu_layers = int(os.environ.get("STARTUP_LLM_N_GPU_LAYERS", "-1"))
            load_llm_from_gguf(llm_name, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
            print(f"[STARTUP] LLM preloaded: {llm_name} (n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers})")
        except Exception as exc:
            errors.append(f"llm:{exc}")
            print(f"[STARTUP] LLM preload failed: {exc}")

    if errors and strict:
        raise RuntimeError("Startup preload failed: " + " | ".join(errors))

def create_app():
    ensure_dirs()
    app = Flask(__name__)
    CORS(app, resources={r"*": {"origins": "*"}})
    
    # This registers all the routes linked in app/api/__init__.py
    app.register_blueprint(api_bp) 

    if _should_run_startup_preload():
        _run_startup_preload()
    
    return app