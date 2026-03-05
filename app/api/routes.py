from app.services.rag_backend import available_backends, load_backend, get_active_backend_name
from flask import Blueprint, request, jsonify, Response, stream_with_context
import json
import re
import time
import os
import tempfile
from pathlib import Path
from werkzeug.utils import secure_filename
import psutil
import app.config as app_config
from app.config import LANG_MAP, LANG_ALIASES, NLLB_LANG_MAP, USE_ONNX_TRANSLATOR, ONNX_LANG_MAP, ONNX_MODEL_FAMILY, ONNX_BACKEND_NAME
from app.services.cache_service import model_cache
from app.services.llm_service import download_gguf, load_llm_from_gguf, llm_generate, llm_generate_stream, unload_llm, get_current_name, SERVER_URL
from app.services.translator_service import translate, detect_supported_language, unload_translator, preload_translator, local_translator_path
from app.services.rag_service import rag_add, rag_remove, rag_retrieve, rag_list, rag_clear, add_pdf_to_rag, get_embed_model
from app.services.benchmark_service import benchmark_pipeline, benchmark_resource_usage, benchmark_llm_metrics, benchmark_translator_metrics, benchmark_rag_metrics
from app.services.benchmark_cache_service import benchmark_query_cache
from app.services.onnx_translator_service import translate_onnx, get_onnx_status, unload_onnx_translator, preload_onnx_translator, ensure_onnx_models
from app.services.onnx_model_download_service import get_onnx_catalog, download_onnx_models, ensure_onnx_tokenizer, list_downloaded_onnx_model_files, ensure_default_onnx_models
from app.services.query_cache_service import QueryCache


def _clean_generation(text: str) -> str:
    """Sanitize model output by removing noise and context echoes."""
    cleaned = text
    # drop fenced blocks and stray backticks
    cleaned = re.sub(r"```.*?```", " ", cleaned, flags=re.S)
    cleaned = re.sub(r"`+", "", cleaned)
    # remove common prefixes
    cleaned = re.sub(r"^\s*Answer\s*:\s*", "", cleaned, flags=re.I)
    # strip out any residual context echoes
    cleaned = re.sub(r"CONTEXT:.*", "", cleaned, flags=re.S|re.I)
    cleaned = re.sub(r"Document \d+:.*", "", cleaned, flags=re.S|re.I)
    # collapse whitespace and trim
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _clean_rag_context(docs: list) -> str:
    """
    Clean RAG documents: filter tables, remove incomplete lines, extract main text.
    Returns cleaned, readable context for LLM prompt.
    """
    if not docs:
        return ""
    
    cleaned_docs = []
    for doc in docs:
        text = doc.get("text") if isinstance(doc, dict) else str(doc)
        if not text:
            continue
        
        # Remove table-like formatting (lines with many |, -, ✓, ✗, etc.)
        lines = text.split('\n')
        filtered_lines = []
        for line in lines:
            # Skip table separators and headers
            if re.match(r'^[\s\|:=\-\+]+$', line):
                continue
            # Skip lines that are mostly table formatting
            if line.count('|') > 3 or line.count('—') > 2:
                continue
            # Skip numbered/bullet list continuations that look incomplete
            if re.match(r'^\s*\d+\s+[A-Z]\s*$', line):
                continue
            # Keep meaningful content
            line_clean = line.strip()
            if line_clean and len(line_clean) > 10:  # Practical minimum
                filtered_lines.append(line_clean)
        
        # Join cleaned lines and remove very long incomplete sequences
        cleaned_text = " ".join(filtered_lines)
        # Truncate to reasonable length and ensure sentence ends
        if len(cleaned_text) > 500:
            cleaned_text = cleaned_text[:500]
            # Find last sentence boundary
            for punct in '.!?।':
                idx = cleaned_text.rfind(punct)
                if idx > 200:  # Ensure we keep enough content
                    cleaned_text = cleaned_text[:idx+1]
                    break
        
        if cleaned_text and len(cleaned_text) > 20:
            cleaned_docs.append(cleaned_text)
    
    return "\n".join(cleaned_docs)


def _ensure_complete_sentence(text: str) -> str:
    """
    Ensure text ends with a sentence terminator. If cut mid-word, backtrack to last space.
    """
    if not text:
        return text
    
    text = text.strip()
    if not text:
        return text
    
    # Already ends with punctuation
    if text[-1] in '.!?।':
        return text
    
    # Check if ends mid-word (last char is alphanumeric)
    if text[-1].isalnum():
        # Find last space or punctuation
        last_space = text.rfind(' ')
        if last_space > len(text) / 2:  # Keep at least half the text
            text = text[:last_space]
            # Add period if not already ending with punct
            if text and text[-1] not in '.!?।':
                text = text.rstrip() + '.'
            return text
    
    # Append period
    return text + '.'

bp = Blueprint("api", __name__)
ACTIVE_TRANSLATOR = "onnx" if USE_ONNX_TRANSLATOR else "nllb"

QUERY_CACHE_FILE = Path(getattr(app_config, "QUERY_CACHE_FILE", Path("models") / "query_cache.json"))
QUERY_CACHE_SIMILARITY_THRESHOLD = float(getattr(app_config, "QUERY_CACHE_SIMILARITY_THRESHOLD", 0.80))
QUERY_CACHE_MAX_ENTRIES = int(getattr(app_config, "QUERY_CACHE_MAX_ENTRIES", 1000))
QUERY_CACHE_ENABLED = bool(getattr(app_config, "QUERY_CACHE_ENABLED", True))

# Initialize query cache (lazy-loaded on first access)
query_cache = None


def get_query_cache():
    """Lazy-load query cache on first access."""
    global query_cache
    if query_cache is None and QUERY_CACHE_ENABLED:
        query_cache = QueryCache(
            cache_file=QUERY_CACHE_FILE,
            similarity_threshold=QUERY_CACHE_SIMILARITY_THRESHOLD,
            max_entries=QUERY_CACHE_MAX_ENTRIES,
        )
    return query_cache


def _get_effective_active_translator() -> str:
    """Return runtime active translator, falling back to NLLB if ONNX isn't available."""
    global ACTIVE_TRANSLATOR
    if ACTIVE_TRANSLATOR == "onnx":
        try:
            if get_onnx_status().get("available"):
                return "onnx"
        except Exception:
            pass
        return "nllb"
    return "nllb"


def _public_translator_name(translator_key: str) -> str:
    return ONNX_BACKEND_NAME if translator_key == "onnx" else "nllb"

@bp.post("/download_llm")
def ep_download_llm():
    body = request.get_json() or {}
    url = body.get("url")
    name = body.get("name")
    if not url:
        return jsonify({"error": "url required"}), 400
    if not name:
        return jsonify({"error": "name required"}), 400
    path = download_gguf(url, name)
    return jsonify({"ok": True, "path": str(path)})

@bp.get("/list_llms")
def ep_list_llms():
    from app.services.llm_service import list_all_llms

    return jsonify({
        "downloaded_llms": list_all_llms(),
        "loaded_llm": get_current_name(),
        "server_url": SERVER_URL,
    })

@bp.post("/load_llm")
def ep_load_llm():
    body = request.get_json() or {}
    name = body.get("name")
    port = int(body.get("port", 8080))  # Ignored for llama-cpp, kept for compatibility
    ctx_size = int(body.get("ctx_size", 4096))
    n_gpu_layers = int(body.get("n_gpu_layers", -1))
    if not name:
        return jsonify({"error": "name required"}), 400
    # load_llm_from_gguf accepts n_ctx and n_gpu_layers, not port or ctx_size
    load_llm_from_gguf(name, n_ctx=ctx_size, n_gpu_layers=n_gpu_layers)
    return jsonify({"ok": True, "loaded": name, "server_url": SERVER_URL})

@bp.get("/current_llm")
def ep_current_llm():
    """Get the currently loaded LLM name."""
    return jsonify({
        "ok": True,
        "loaded_llm": get_current_name(),
        "server_url": SERVER_URL
    })

@bp.get("/translator_status")
def ep_translator_status():
    """Get status of available translators (NLLB and ONNX)."""
    onnx_status = get_onnx_status()
    active_key = _get_effective_active_translator()
    active = _public_translator_name(active_key)

    nllb_path = local_translator_path(app_config.NLLB_MODEL)
    nllb_downloaded = nllb_path.exists() and any(nllb_path.iterdir())
    onnx_downloaded_models = list_downloaded_onnx_model_files(family=ONNX_MODEL_FAMILY)
    nllb_downloaded_models = [app_config.NLLB_MODEL] if nllb_downloaded else []

    return jsonify({
        "active_translator": active,
        "active_translator_key": active_key,
        "onnx_backend_name": ONNX_BACKEND_NAME,
        "onnx": onnx_status,
        "nllb": {
            "available": True,
            "model": app_config.NLLB_MODEL,
            "downloaded": nllb_downloaded,
            "local_path": str(nllb_path),
        },
        "downloaded_models": {
            "onnx": onnx_downloaded_models,
            ONNX_BACKEND_NAME: onnx_downloaded_models,
            "nllb": nllb_downloaded_models,
            "all": [
                *[f"{ONNX_BACKEND_NAME}:{name}" for name in onnx_downloaded_models],
                *[f"nllb:{name}" for name in nllb_downloaded_models],
            ],
        },
    })


@bp.get("/onnx_models/catalog")
def ep_onnx_models_catalog():
    """List ONNX model files available in the configured Google Drive folder."""
    refresh = str(request.args.get("refresh", "false")).lower() in ("1", "true", "yes")
    family = (request.args.get("family") or ONNX_MODEL_FAMILY).strip().lower()
    try:
        catalog = get_onnx_catalog(force_refresh=refresh, family=family)
        tokenizer_ready = False
        tokenizer_error = None
        try:
            from app.services.onnx_model_download_service import is_onnx_tokenizer_ready
            tokenizer_ready = bool(is_onnx_tokenizer_ready(family=family))
        except Exception as tokenizer_exc:
            tokenizer_error = str(tokenizer_exc)
        downloaded_files = list_downloaded_onnx_model_files(family=family)
        downloaded_set = set(downloaded_files)
        downloaded_name_set = {item.split("/")[-1] for item in downloaded_files}
        default_files = catalog.get("default_files") or []
        all_default_downloaded = all(
            (
                file_name in downloaded_set
                or f"encoder/{file_name}" in downloaded_set
                or f"decoder/{file_name}" in downloaded_set
                or f"lm_head/{file_name}" in downloaded_set
            )
            for file_name in default_files
        )

        family_files_status = []
        for file_item in (catalog.get("files") or []):
            file_name = str(file_item.get("name") or "")
            if not file_name:
                continue
            is_downloaded = (
                file_name in downloaded_set
                or f"encoder/{file_name}" in downloaded_set
                or f"decoder/{file_name}" in downloaded_set
                or f"lm_head/{file_name}" in downloaded_set
                or file_name in downloaded_name_set
            )
            family_files_status.append({
                "name": file_name,
                "downloaded": bool(is_downloaded),
            })

        total_family_files = len(family_files_status)
        downloaded_family_files = sum(1 for item in family_files_status if item["downloaded"])
        all_family_downloaded = total_family_files > 0 and downloaded_family_files == total_family_files

        return jsonify({
            "ok": True,
            **catalog,
            "downloaded_files": downloaded_files,
            "all_default_downloaded": all_default_downloaded,
            "tokenizer_ready": tokenizer_ready,
            "tokenizer_error": tokenizer_error,
            "family_files_status": family_files_status,
            "total_family_files": total_family_files,
            "downloaded_family_files": downloaded_family_files,
            "all_family_downloaded": all_family_downloaded,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@bp.get("/api/onnx_models/catalog")
def ep_onnx_models_catalog_api_alias():
    """Alias for clients expecting /api-prefixed ONNX catalog route."""
    return ep_onnx_models_catalog()


@bp.post("/onnx_models/download")
def ep_onnx_models_download():
    """Download default or selected ONNX model files from Drive into local models directory."""
    body = request.get_json() or {}
    files = body.get("files")
    family = (body.get("family") or ONNX_MODEL_FAMILY).strip().lower()
    include_tokenizer = body.get("include_tokenizer", True)
    if files is not None and not isinstance(files, list):
        return jsonify({"error": "files must be an array of file names"}), 400
    if not isinstance(include_tokenizer, bool):
        return jsonify({"error": "include_tokenizer must be a boolean"}), 400

    try:
        details = download_onnx_models(selected_files=files, include_tokenizer=include_tokenizer, family=family)
        return jsonify({"ok": True, **details})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@bp.post("/api/onnx_models/download")
def ep_onnx_models_download_api_alias():
    """Alias for clients expecting /api-prefixed ONNX download route."""
    return ep_onnx_models_download()


@bp.post("/onnx_tokenizer/ensure")
def ep_onnx_tokenizer_ensure():
    """Ensure ONNX tokenizer assets exist locally, downloading if required."""
    body = request.get_json(silent=True) or {}
    force_download = body.get("force_download", False)
    family = (body.get("family") or ONNX_MODEL_FAMILY).strip().lower()
    if not isinstance(force_download, bool):
        return jsonify({"error": "force_download must be a boolean"}), 400

    try:
        details = ensure_onnx_tokenizer(force_download=force_download, family=family)
        return jsonify({"ok": True, **details})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@bp.post("/api/onnx_tokenizer/ensure")
def ep_onnx_tokenizer_ensure_api_alias():
    """Alias for clients expecting /api-prefixed ONNX tokenizer ensure route."""
    return ep_onnx_tokenizer_ensure()

@bp.post("/toggle_translator")
def ep_toggle_translator():
    """Toggle between ONNX and NLLB translator."""
    global ACTIVE_TRANSLATOR
    body = request.get_json() or {}
    use_onnx = body.get("use_onnx", _get_effective_active_translator() == "onnx")
    
    onnx_status = get_onnx_status()
    if use_onnx and not onnx_status["available"]:
        return jsonify({
            "error": "ONNX models not available",
            "details": onnx_status
        }), 503
    
    if use_onnx:
        print("[API] Switching to ONNX translator")
        unload_translator()  # Clean up NLLB if loaded
        ACTIVE_TRANSLATOR = "onnx"
    else:
        print("[API] Switching to NLLB translator")
        unload_onnx_translator()  # Clean up ONNX if loaded
        ACTIVE_TRANSLATOR = "nllb"
    
    return jsonify({
        "ok": True,
        "active_translator": _public_translator_name("onnx" if use_onnx else "nllb"),
        "active_translator_key": "onnx" if use_onnx else "nllb",
        "onnx_backend_name": ONNX_BACKEND_NAME,
    })


@bp.post("/translator_preload")
def ep_translator_preload():
    """Preload translator models without running a translation."""
    global ACTIVE_TRANSLATOR
    body = request.get_json() or {}
    use_onnx = body.get("use_onnx", _get_effective_active_translator() == "onnx")

    if use_onnx:
        try:
            details = preload_onnx_translator()
            ACTIVE_TRANSLATOR = "onnx"
        except Exception as e:
            return jsonify({
                "error": str(e),
                "details": get_onnx_status(),
            }), 503
    else:
        details = preload_translator()
        ACTIVE_TRANSLATOR = "nllb"

    return jsonify({
        "ok": True,
        "active_translator": _public_translator_name("onnx" if use_onnx else "nllb"),
        "active_translator_key": "onnx" if use_onnx else "nllb",
        "onnx_backend_name": ONNX_BACKEND_NAME,
        "details": details,
    })

@bp.post("/translate")
def ep_translate():
    body = request.get_json() or {}
    text = body.get("text")
    target = (body.get("target") or "en").lower()
    stream = bool(body.get("stream", True))
    max_tokens = int(body.get("max_new_tokens", 256))
    use_onnx = body.get("use_onnx", USE_ONNX_TRANSLATOR)

    if not text:
        return jsonify({"error": "text required"}), 400

    # Auto-detect source language (must be in LANG_CONF)
    src_lang_key = detect_supported_language(text)
    if not src_lang_key:
        return jsonify({"error": "could not auto-detect a supported language"}), 400

    # Determine which language map and translation function to use
    if use_onnx:
        target_map = ONNX_LANG_MAP
        translate_fn = translate_onnx
        backend_key = "onnx"
        backend = ONNX_BACKEND_NAME
        # For ONNX, use short codes (e.g., "hi" instead of "hin_Deva")
        src_code = src_lang_key
        
        # Check if ONNX supports the detected language
        if src_lang_key not in ONNX_LANG_MAP:
            supported_langs = ", ".join(sorted(ONNX_LANG_MAP.keys()))
            return jsonify({
                "error": f"Language '{src_lang_key}' not supported by ONNX model",
                "details": f"ONNX supports: {supported_langs}",
                "suggestion": "Switch to NLLB translator or use a supported language"
            }), 400
    else:
        target_map = NLLB_LANG_MAP
        translate_fn = translate
        backend_key = "nllb"
        backend = "nllb"
        # For NLLB, use long codes from LANG_MAP (e.g., "hin_Deva")
        src_code, _ = LANG_MAP[src_lang_key]

    # Normalize target and validate it against known mappings or raw NLLB codes
    target_key = LANG_ALIASES.get(target, target)
    target_code = target_map.get(target_key, target_key)
    if target_key not in LANG_MAP and target_key not in target_map and "_" not in target_key:
        return jsonify({"error": f"unsupported target language: {target}"}), 400

    onnx_auto_download = None
    if use_onnx and not ensure_onnx_models():
        return jsonify({
            "error": "onnx models unavailable",
            "details": "Download ONNX models explicitly via /onnx_models/download",
        }), 503

    # Helper to iterate sentences from paragraphs
    sentence_end_re = re.compile(r"(.+?[.!?](?:\"|'|”)?)(\s+|$)", re.S)

    def iter_sentences(blob: str):
        buffer = blob
        while True:
            match = sentence_end_re.search(buffer)
            if not match:
                break
            sent = match.group(1).strip()
            buffer = buffer[match.end():]
            if sent:
                yield sent
        if buffer.strip():
            yield buffer.strip()

    if not stream:
        translated_sentences = []
        for sent in iter_sentences(text):
            try:
                translated = translate_fn(sent, src_code, target_code, max_tokens)
            except Exception as e:
                return jsonify({
                    "error": f"{backend} translation failed",
                    "details": str(e)
                }), 503
            translated_sentences.append({
                "source": sent,
                "translated": translated,
            })

        combined = " ".join(item["translated"] for item in translated_sentences)
        return jsonify({
            "input": text,
            "detected_lang": src_lang_key,
            "target_lang": target_key,
            "translated_text": combined,
            "sentences": translated_sentences,
            "backend": backend,
            "backend_key": backend_key,
            "auto_download": onnx_auto_download,
        })

    def event_stream():
        meta = {
            "type": "meta",
            "detected_lang": src_lang_key,
            "target_lang": target_key,
            "backend": backend,
            "backend_key": backend_key,
            "auto_download": onnx_auto_download,
        }
        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

        try:
            for idx, sent in enumerate(iter_sentences(text), start=1):
                try:
                    translated = translate_fn(sent, src_code, target_code, max_tokens)
                except Exception as e:
                    err = {"type": "error", "message": f"{backend} translation failed: {str(e)}"}
                    yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
                    return
                payload = {
                    "type": "sentence",
                    "index": idx,
                    "source": sent,
                    "translated": translated,
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
        except Exception as e:
            err = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    headers = {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return Response(stream_with_context(event_stream()), headers=headers)


@bp.get("/system/metrics")
def ep_system_metrics():
    """
    Stream system metrics via SSE: CPU%, RAM (system + current process), and VRAM if available.

    Query params:
      - interval_ms: polling interval in milliseconds (default 1000)
      - include_process: whether to include current python process stats (default true)
    """
    try:
        interval_ms = int(request.args.get("interval_ms", "1000"))
    except ValueError:
        interval_ms = 1000
    include_process = (request.args.get("include_process", "true").lower() in ("true", "1", "yes"))

    process = psutil.Process(os.getpid()) if include_process else None

    # Prime psutil CPU percent to compute over interval
    psutil.cpu_percent(interval=None)

    def get_vram():
        try:
            import torch
            if torch.cuda.is_available():
                total = torch.cuda.get_device_properties(0).total_memory
                used = torch.cuda.memory_allocated(0)
                reserved = torch.cuda.memory_reserved(0)
                return {
                    "available": True,
                    "total_bytes": int(total),
                    "used_bytes": int(used),
                    "reserved_bytes": int(reserved),
                }
        except Exception as e:
            return {"available": False, "error": str(e)}
        return {"available": False}

    def event_stream():
        try:
            while True:
                cpu_pct = psutil.cpu_percent(interval=None)
                vm = psutil.virtual_memory()
                swap = psutil.swap_memory()

                payload = {
                    "type": "metrics",
                    "timestamp": time.time(),
                    "cpu_percent": cpu_pct,
                    "ram": {
                        "total_bytes": int(vm.total),
                        "used_bytes": int(vm.used),
                        "available_bytes": int(vm.available),
                        "percent": float(vm.percent),
                    },
                    "swap": {
                        "total_bytes": int(swap.total),
                        "used_bytes": int(swap.used),
                        "percent": float(swap.percent),
                    },
                    "vram": get_vram(),
                }

                if process is not None:
                    with process.oneshot():
                        mem_info = process.memory_info()
                        cpu_proc = process.cpu_percent(interval=None)
                        payload["process"] = {
                            "pid": process.pid,
                            "cpu_percent": cpu_proc,
                            "rss_bytes": int(mem_info.rss),  # resident set size
                            "vms_bytes": int(mem_info.vms),  # virtual memory size
                        }

                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                time.sleep(max(0.05, interval_ms / 1000.0))
        except GeneratorExit:
            # Client disconnected; stop the stream
            return
        except Exception as e:
            err = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    headers = {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return Response(stream_with_context(event_stream()), headers=headers)

@bp.post("/infer")
def ep_infer():
    body = request.get_json() or {}
    text = body.get("text")
    lang = body.get("lang")
    max_tokens = int(body.get("max_new_tokens", 128))

    if not text:
        return jsonify({"error": "text required"}), 400

    # Normalize language input and optionally auto-detect
    lang = (lang or "").lower()
    detected_lang = None
    if lang in ("", "auto"):
        detected_lang = detect_supported_language(text)
        if not detected_lang:
            # Fallback for unsupported/short/ASCII text where detector returns None.
            detected_lang = "en"
        lang = detected_lang
    else:
        lang = LANG_ALIASES.get(lang, lang)

    if not lang or lang not in LANG_MAP:
        return jsonify({"error": f"unsupported lang: {lang}"}), 400

    src_lang_key = lang
    src_lang, en_lang = LANG_MAP[src_lang_key]
    is_source_english = (src_lang == "eng_Latn")
    print(f"Translating from {src_lang} to {en_lang} and back.")

    active_translator = _get_effective_active_translator()
    translation_backend = active_translator
    if active_translator == "onnx" and src_lang_key not in ONNX_LANG_MAP:
        translation_backend = "nllb"
        print(f"[INFER] ONNX does not support source language '{src_lang_key}'. Falling back to NLLB.")

    def _translate_pipeline(text_value: str, src_code: str, tgt_code: str) -> str:
        if translation_backend == "onnx":
            return translate_onnx(text_value, src_code, tgt_code)
        return translate(text_value, src_code, tgt_code)
    
    try:
        # 1. Convert user input → English (using short codes)
        _t0 = time.perf_counter()
        if is_source_english:
            english_text = text
        else:
            if translation_backend == "onnx":
                english_text = _translate_pipeline(text, src_lang_key, "en")
            else:
                english_text = _translate_pipeline(text, src_lang, "en")
        _t1 = time.perf_counter()
        translate_in_s = _t1 - _t0
        print(f"Translated input to English: {english_text}")
    except RuntimeError as e:
        if "torchvision" in str(e) or "nms" in str(e):
            return jsonify({
                "error": "translator initialization failed due to dependency conflict",
                "details": str(e),
                "suggestion": "Try: pip install --upgrade transformers torch torchvision"
            }), 503
        raise

    # If client requests streaming (default True), stream sentence-by-sentence
    stream = bool(body.get("stream", True))

    # Query caching: check if similar query exists and reuse RAG docs
    cache_hit = False
    cache_similarity = None
    qcache = get_query_cache()

    if qcache is not None:
        try:
            embed_model = get_embed_model()
            query_embedding = embed_model.encode([english_text])[0].tolist()

            cached_result = qcache.find_similar_query(query_embedding)
            if cached_result is not None:
                rag_docs, cache_similarity = cached_result
                cache_hit = True
                print(f"[Query Cache] Hit! Similarity={cache_similarity:.3f}")
        except Exception as e:
            print(f"[Query Cache] Warning: Cache lookup failed: {e}")

    # 2. RAG retrieve
    _r0 = time.perf_counter()
    if not cache_hit:
        rag_docs = rag_retrieve(english_text, top_k=3)

        # Add to cache for future queries
        if qcache is not None:
            try:
                embed_model = get_embed_model()
                query_embedding = embed_model.encode([english_text])[0].tolist()
                qcache.add_query(english_text, query_embedding, rag_docs)
            except Exception as e:
                print(f"[Query Cache] Warning: Failed to cache query: {e}")
    _r1 = time.perf_counter()
    rag_retrieval_s = _r1 - _r0
    context = ""
    out_of_bounds = False
    if rag_docs:
        # Clean RAG context to remove tables, incomplete text, etc.
        context = _clean_rag_context(rag_docs)
        if not context.strip():
            # Even after cleaning, no meaningful content
            out_of_bounds = True
    else:
        # no relevant documents found -> out-of-knowledge query
        out_of_bounds = True

    # 3. If query is out-of-bounds, send a fallback message in user language
    if out_of_bounds:
        fallback_en = "I'm sorry, I don't have information on that topic."
        if is_source_english:
            answer_native = fallback_en
        else:
            if translation_backend == "onnx":
                answer_native = _translate_pipeline(fallback_en, "en", src_lang_key)
            else:
                answer_native = _translate_pipeline(fallback_en, "en", src_lang)

        if not stream:
            return jsonify({
                "input": text,
                "english_in": english_text,
                "rag_used": rag_docs,
                "cache_hit": cache_hit,
                "cache_similarity": cache_similarity,
                "llm_prompt": None,
                "llm_output_en": None,
                "final_output": answer_native,
                "lang_used": lang,
                "detected_lang": detected_lang,
                "translator_backend": _public_translator_name(translation_backend),
                "translator_backend_key": translation_backend,
                "out_of_bounds": True,
            })

        # streaming: emit meta + single sentence, then end
        def event_stream():
            meta = {
                "type": "meta",
                "english_in": english_text,
                "prompt": None,
                "lang_used": lang,
                "detected_lang": detected_lang,
                "translator_backend": _public_translator_name(translation_backend),
                "translator_backend_key": translation_backend,
                "rag_used": rag_docs,
                "rag_context": context,
                "cache_hit": cache_hit,
                "cache_similarity": cache_similarity,
                "out_of_bounds": True,
            }
            yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"
            payload = {"type": "sentence", "english": fallback_en, "translated": answer_native}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            # done event is naturally ended by caller when stream exhausts
        return Response(stream_with_context(event_stream()), mimetype="text/event-stream")

    # 3. Build prompt for LLM
    context_block = f"Relevant context:\n{context}\n" if context else ""
    final_prompt = f"""You are a helpful assistant that answers questions based only on provided context.

STRICT RULES:
1. Answer ONLY using information from the CONTEXT below.
2. NEVER invent, guess, or add information not in the context.
3. NEVER repeat or quote the context verbatim or include context framework text.
4. NEVER mention "CONTEXT", "Document", "Table", or "Reference".
5. Generate a COMPLETE, GRAMMATICALLY CORRECT sentence. Never stop mid-sentence or mid-word.
6. If context is insufficient or irrelevant, respond: "I don't have information on that topic."
7. Output exactly ONE sentence or phrase that directly answers the question.

CONTEXT:
{context_block}

QUESTION: {english_text}

ANSWER:"""
    print(context_block)

    # 4/5. Run inference and (optionally) stream translations back
    if not stream:
        # Non-streaming (legacy) behaviour
        try:
            llm_output_en = llm_generate(final_prompt, max_new_tokens=max_tokens)
            llm_output_en = _clean_generation(llm_output_en)
            # ensure complete sentence (no mid-word cutoff)
            llm_output_en = _ensure_complete_sentence(llm_output_en)
        except RuntimeError as e:
            if "llama_decode returned -1" in str(e):
                return jsonify({
                    "error": "LLM generation failed due to context/memory limits",
                    "details": str(e),
                    "suggestion": "Try loading the model with a larger n_ctx or reduce max_new_tokens"
                }), 503
            raise
        
        if is_source_english:
            answer_native = llm_output_en
        else:
            if translation_backend == "onnx":
                answer_native = _translate_pipeline(llm_output_en, "en", src_lang_key)
            else:
                answer_native = _translate_pipeline(llm_output_en, "en", src_lang)
        
        # ensure translated answer is also complete
        answer_native = _ensure_complete_sentence(answer_native)
        
        return jsonify({
            "input": text,
            "english_in": english_text,
            "rag_used": rag_docs,
            "cache_hit": cache_hit,
            "cache_similarity": cache_similarity,
            "llm_prompt": final_prompt,
            "llm_output_en": llm_output_en,
            "final_output": answer_native,
            "lang_used": lang,
            "detected_lang": detected_lang,
            "translator_backend": _public_translator_name(translation_backend),
            "translator_backend_key": translation_backend,
        })

    # Streaming response (SSE). We will yield JSON payloads per translated sentence.
    sentence_end_re = re.compile(r"(.+?[.!?](?:\"|'|”)?)(\s+|$)", re.S)

    def event_stream():
        llm_start = time.perf_counter()
        translate_out_total_s = 0.0
        english_output_raw = ""
        english_output_clean = ""
        # Meta event with initial English input
        meta = {
            "type": "meta",
            "english_in": english_text,
            "prompt": final_prompt,
            "lang_used": lang,
            "detected_lang": detected_lang,
            "translator_backend": _public_translator_name(translation_backend),
            "translator_backend_key": translation_backend,
            "rag_used": rag_docs,
            "rag_context": context,
            "cache_hit": cache_hit,
            "cache_similarity": cache_similarity,
        }
        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

        buffer = ""
        try:
            for chunk in llm_generate_stream(final_prompt, max_new_tokens=max_tokens):
                english_output_raw += chunk
                buffer += chunk
                # extract complete sentences from buffer
                while True:
                    m = sentence_end_re.search(buffer)
                    if not m:
                        break
                    sent = m.group(1).strip()
                    buffer = buffer[m.end():]
                    if not sent:
                        continue
                    # clean and translate the sentence back to the user's language
                    sent_clean = _clean_generation(sent)
                    if not sent_clean:
                        continue
                    _to0 = time.perf_counter()
                    if is_source_english:
                        translated = sent_clean
                    else:
                        if translation_backend == "onnx":
                            translated = _translate_pipeline(sent_clean, "en", src_lang_key)
                        else:
                            translated = _translate_pipeline(sent_clean, "en", src_lang)
                    _to1 = time.perf_counter()
                    translate_out_total_s += (_to1 - _to0)
                    english_output_clean += (sent_clean + " ")
                    # ensure translated answer is complete
                    translated = _ensure_complete_sentence(translated)
                    payload = {"type": "sentence", "english": sent_clean, "translated": translated}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            # flush remaining buffer
            if buffer.strip():
                sent = buffer.strip()
                sent_clean = _clean_generation(sent)
                # ensure final output is complete sentence
                sent_clean = _ensure_complete_sentence(sent_clean)
                if sent_clean:
                    _to0 = time.perf_counter()
                    if is_source_english:
                        translated = sent_clean
                    else:
                        if translation_backend == "onnx":
                            translated = _translate_pipeline(sent_clean, "en", src_lang_key)
                        else:
                            translated = _translate_pipeline(sent_clean, "en", src_lang)
                    _to1 = time.perf_counter()
                    translate_out_total_s += (_to1 - _to0)
                    english_output_clean += (sent_clean + " ")
                    # ensure translated answer is complete
                    translated = _ensure_complete_sentence(translated)
                    payload = {"type": "sentence", "english": sent_clean, "translated": translated}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            llm_end = time.perf_counter()
            total_llm_stream_s = llm_end - llm_start
            total_pipeline_s = (time.perf_counter() - _t0)  # from initial translate start
            final_payload = {
                "type": "done",
                "timing": {
                    "translate_to_en_s": round(translate_in_s, 6),
                    "rag_retrieval_s": round(rag_retrieval_s, 6),
                    "llm_stream_s": round(total_llm_stream_s, 6),
                    "translate_to_source_total_s": round(translate_out_total_s, 6),
                    "total_pipeline_s": round(total_pipeline_s, 6),
                },
                "rag_used": rag_docs,
                "cache_hit": cache_hit,
                "cache_similarity": cache_similarity,
                "llm_output_en_raw": english_output_raw.strip(),
                "llm_output_en_clean": english_output_clean.strip(),
            }
            yield f"data: {json.dumps(final_payload, ensure_ascii=False)}\n\n"
        except Exception as e:
            err = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    headers = {"Content-Type": "text/event-stream; charset=utf-8", "Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return Response(stream_with_context(event_stream()), headers=headers)

@bp.post("/infer_raw")
def ep_infer_raw():
    body = request.get_json() or {}
    prompt = body.get("prompt")
    max_tokens = int(body.get("max_new_tokens", 128))
    use_rag = body.get("use_rag", True)  # Default to using RAG
    if not prompt:
        return jsonify({"error": "prompt required"}), 400
    
    # Retrieve RAG documents if enabled
    rag_docs = []
    if use_rag:
        rag_docs = rag_retrieve(prompt, top_k=3)
    
    # Build prompt with RAG context if available
    if rag_docs:
        rag_block = "\n\n".join(f"Document {i+1}: {d}" for i, d in enumerate(rag_docs))
        final_prompt = f"Relevant context:\n{rag_block}\n\nUser question:\n{prompt}\nAnswer the question clearly and factually. Return just the answer with no further explanantion is as simple words as possible."
    else:
        final_prompt = prompt
    
    output = llm_generate(final_prompt, max_new_tokens=max_tokens)
    return jsonify({
        "prompt": prompt,
        "rag_used": rag_docs,
        "final_prompt": final_prompt,
        "output": output
    })

@bp.get("/health")
def ep_health():
    return jsonify({"status": "alive"})

@bp.post("/rag/add")
def ep_rag_add():
    body = request.get_json(silent=True) or {}
    text = (
        body.get("text")
        or body.get("content")
        or body.get("data")
        or body.get("document")
        or request.form.get("text")
        or request.form.get("content")
    )

    if not text:
        raw_text = request.get_data(as_text=True) or ""
        if raw_text.strip():
            text = raw_text.strip()

    if not text:
        return jsonify({
            "error": "text required",
            "hint": "send JSON with 'text' (or 'content'/'data') or form field 'text'"
        }), 400

    text = text.strip()
    if not text:
        return jsonify({"error": "text required"}), 400

    doc_id = rag_add(text)
    return jsonify({"ok": True, "id": doc_id})

@bp.post("/rag/remove")
def ep_rag_remove():
    body = request.get_json() or {}
    doc_id = body.get("id")
    if not doc_id:
        return jsonify({"error": "id required"}), 400
    if not rag_remove(doc_id):
        return jsonify({"error": "invalid id"}), 400
    return jsonify({"ok": True})

@bp.get("/rag/list")
def ep_rag_list():
    """List all RAG documents with their IDs."""
    docs = rag_list()
    return jsonify({"ok": True, "documents": docs, "count": len(docs)})

@bp.post("/rag/clear")
def ep_rag_clear():
    """Clear all RAG documents."""
    rag_clear()
    return jsonify({"ok": True, "message": "All RAG documents cleared"})


@bp.post("/rag/add_pdf")
def ep_rag_add_pdf():
    """Accept an uploaded PDF file (multipart form, field 'file'), ingest it and add chunks to RAG."""
    if 'file' not in request.files:
        return jsonify({"error": "file required (multipart form, field name 'file')"}), 400

    f = request.files['file']
    if not f or f.filename == '':
        return jsonify({"error": "empty filename or no file provided"}), 400

    filename = secure_filename(f.filename)
    tmp_dir = tempfile.mkdtemp(prefix="rag_upload_")
    tmp_path = os.path.join(tmp_dir, filename)
    try:
        f.save(tmp_path)
        result = add_pdf_to_rag(tmp_path)
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify({"error": "failed to ingest PDF", "details": str(e)}), 500
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        try:
            if os.path.exists(tmp_dir):
                os.rmdir(tmp_dir)
        except Exception:
            pass

@bp.post("/rag/pdf/add")
def ep_rag_pdf_add():
    """Compatibility endpoint: add a PDF to RAG using a JSON body with pdf_path."""
    body = request.get_json() or {}
    pdf_path = body.get("pdf_path")

    if not pdf_path:
        return jsonify({"error": "pdf_path required"}), 400
    if not os.path.exists(pdf_path):
        return jsonify({"error": "pdf_path not found"}), 400
    if not pdf_path.lower().endswith(".pdf"):
        return jsonify({"error": "only .pdf supported"}), 400

    try:
        result = add_pdf_to_rag(pdf_path)
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify({"error": "failed to ingest PDF", "details": str(e)}), 500

@bp.post("/rag/pdf/upload")
def ep_rag_pdf_upload():
    """Compatibility endpoint: upload a PDF file (multipart field name 'file') and add it to RAG."""
    if 'file' not in request.files:
        return jsonify({"error": "file required"}), 400

    f = request.files['file']
    if not f or not f.filename:
        return jsonify({"error": "filename missing"}), 400

    filename = secure_filename(f.filename)
    if not filename.lower().endswith(".pdf"):
        return jsonify({"error": "only .pdf supported"}), 400

    tmp_dir = tempfile.mkdtemp(prefix="rag_upload_")
    tmp_path = os.path.join(tmp_dir, filename)
    try:
        f.save(tmp_path)
        result = add_pdf_to_rag(tmp_path)
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify({"error": "failed to ingest PDF", "details": str(e)}), 500
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        try:
            if os.path.exists(tmp_dir):
                os.rmdir(tmp_dir)
        except Exception:
            pass

@bp.post("/rag/search")
def ep_rag_search():
    """Search RAG documents by query and return top-k matches."""
    body = request.get_json() or {}
    query = body.get("query")
    top_k = int(body.get("top_k", 3))
    similarity_threshold = float(body.get("similarity_threshold", 0.35))
    
    if not query:
        return jsonify({"error": "query required"}), 400
    
    results = rag_retrieve(query, top_k=top_k, similarity_threshold=similarity_threshold)
    return jsonify({
        "ok": True,
        "query": query,
        "top_k": top_k,
        "similarity_threshold": similarity_threshold,
        "results": results,
        "count": len(results)
    })

@bp.get("/rag/backends")
def ep_rag_backends():
    """List available RAG backends and the currently active one."""
    try:
        backends = available_backends()
        active = get_active_backend_name()
        return jsonify({"ok": True, "available": backends, "active": active})
    except Exception as e:
        return jsonify({"error": "failed to list backends", "details": str(e)}), 500

@bp.post("/rag/swap_backend")
def ep_rag_swap_backend():
    """Swap the active RAG backend and rebuild the index from current metadata."""
    body = request.get_json() or {}
    name = body.get("backend")
    if not name:
        return jsonify({"error": "backend name required"}), 400

    try:
        load_backend(name)
        return jsonify({"ok": True, "active": name})
    except Exception as e:
        return jsonify({"error": "failed to load backend", "details": str(e)}), 500

@bp.post("/rag/backend/load")
def ep_rag_backend_load():
    """Compatibility endpoint for loading a RAG backend using 'name' or 'backend'."""
    body = request.get_json() or {}
    name = body.get("name") or body.get("backend")
    if not name:
        return jsonify({"error": "name required"}), 400

    try:
        load_backend(name)
        return jsonify({"ok": True, "active": get_active_backend_name(), "available": available_backends()})
    except Exception as e:
        return jsonify({"error": "failed to load backend", "details": str(e)}), 500

@bp.get("/query_cache/stats")
def ep_query_cache_stats():
    """Get query cache statistics."""
    qcache = get_query_cache()
    if qcache is None:
        return jsonify({"ok": True, "enabled": False, "message": "Query cache is disabled"}), 200

    return jsonify({"ok": True, "enabled": True, **qcache.stats()})

@bp.post("/query_cache/clear")
def ep_query_cache_clear():
    """Clear all cached queries."""
    qcache = get_query_cache()
    if qcache is None:
        return jsonify({"ok": True, "message": "Query cache is disabled"}), 200

    qcache.clear()
    return jsonify({"ok": True, "message": "Query cache cleared"})

@bp.post("/unload_llm")
def ep_unload_llm():
    """Stop the background llama-server."""
    unload_llm()
    translator_unloaded = unload_translator()
    return jsonify({
        "ok": True,
        "message": "LLM server stopped and translator unloaded",
        "translator_unloaded": bool(translator_unloaded)
    })

@bp.get("/benchmark")
def ep_benchmark():
    text = request.args.get("text", "കേരളത്തിൽ മഴ കനത്തിരിക്കുന്നു.")
    lang = request.args.get("lang", "ml")

    lang = LANG_ALIASES.get(lang, lang)

    if lang not in LANG_MAP:
        return jsonify({"error": f"Unsupported language {lang}"}), 400

    src_lang, en_lang = LANG_MAP[lang]

    try:
        results = benchmark_pipeline(
            test_text=text,
            src_lang=src_lang,
            tgt_lang=en_lang,
            max_tokens=64,
        )
        return jsonify({"ok": True, "results": results})
    except RuntimeError as e:
        if "torchvision" in str(e) or "nms" in str(e):
            return jsonify({
                "error": "translator initialization failed due to dependency conflict",
                "details": str(e),
                "suggestion": "Try: pip install --upgrade transformers torch torchvision"
            }), 503
        raise


@bp.post("/benchmark/resource")
def ep_benchmark_resource():
    """
    Comprehensive resource usage benchmark endpoint.
    
    Request body:
    {
        "llm_name": "Qwen2-500M-Instruct-GGUF",
        "prompts": [
            {"text": "भारत का राष्ट्रपति कौन है?", "lang": "hi"},
            {"text": "What is machine learning?", "lang": "en"}
        ],
        "rag_data": ["AI is artificial intelligence", "ML is machine learning"],  // optional
        "n_ctx": 4096,         // optional, default 4096
        "n_gpu_layers": -1,    // optional, default -1 (all)
        "max_tokens": 128      // optional, default 128
    }
    
    Returns comprehensive metrics for:
    - Baseline (unloaded)
    - LLM loaded
    - Translator loaded
    - RAG data loaded
    - Per-prompt full pipeline (translate→infer→translate back) with time, RAM, CPU, VRAM
    """
    body = request.get_json() or {}
    
    llm_name = body.get("llm_name")
    prompts = body.get("prompts", [])
    rag_data = body.get("rag_data", None)
    n_ctx = int(body.get("n_ctx", 4096))
    n_gpu_layers = int(body.get("n_gpu_layers", -1))
    max_tokens = int(body.get("max_tokens", 128))
    
    if not llm_name:
        return jsonify({"error": "llm_name required"}), 400
    if not prompts or not isinstance(prompts, list):
        return jsonify({"error": "prompts must be a non-empty list"}), 400
    
    try:
        results = benchmark_resource_usage(
            llm_name=llm_name,
            prompts=prompts,
            rag_data=rag_data,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            max_tokens=max_tokens
        )
        return jsonify({"ok": True, "results": results})
    except Exception as e:
        return jsonify({
            "error": "benchmark failed",
            "details": str(e)
        }), 500


@bp.post("/llm_metrics")
def ep_llm_metrics():
    """
    Measure detailed LLM performance metrics.
    
    Request body:
    {
        "llm_name": "Qwen2-500M-Instruct-GGUF",
        "n_ctx": 4096,         // optional, default 4096
        "n_gpu_layers": -1,    // optional, default -1 (all)
        "max_tokens": 128      // optional, default 128
    }
    
    Returns:
    {
        "ok": true,
        "model_size_gb": 0.5,
        "load_time_s": 2.3,
        "first_token_latency_ms": 45.2,
        "tokens_per_second": 15.6,
        "output_length_tokens": 128,
        "output_text": "...",
        "total_inference_time_s": 8.2,
        "memory": {
            "baseline_rss_mb": 450.2,
            "loaded_rss_mb": 950.8,
            "peak_rss_mb": 1024.5,
            "load_increase_mb": 500.6,
            "inference_increase_mb": 73.7
        },
        "vram": {
            "baseline_used_mb": 0,
            "loaded_used_mb": 480.5,
            "peak_used_mb": 520.3,
            "total_mb": 8192
        },
        "config": {
            "llm_name": "...",
            "n_ctx": 4096,
            "n_gpu_layers": -1,
            "max_tokens": 128,
            "demo_prompt": "..."
        }
    }
    """
    body = request.get_json() or {}
    
    llm_name = body.get("llm_name")
    n_ctx = int(body.get("n_ctx", 4096))
    n_gpu_layers = int(body.get("n_gpu_layers", -1))
    max_tokens = int(body.get("max_tokens", 128))
    
    if not llm_name:
        return jsonify({"error": "llm_name required"}), 400
    
    try:
        results = benchmark_llm_metrics(
            llm_name=llm_name,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            max_tokens=max_tokens
        )
        
        if "error" in results:
            return jsonify(results), 400
        
        return jsonify({"ok": True, **results})
    except Exception as e:
        return jsonify({
            "error": "benchmark failed",
            "details": str(e)
        }), 500


@bp.post("/translator_metrics")
def ep_translator_metrics():
    """
    Measure detailed translator performance metrics.
    
    Request body:
    {
        "src_lang": "hi",      // source language code
        "tgt_lang": "en"       // optional, target language (default: "en")
    }
    
    Returns:
    {
        "ok": true,
        "input": {
            "text": "...",
            "char_length": 250,
            "token_length_estimate": 45,
            "src_lang": "hi",
            "tgt_lang": "en"
        },
        "throughput": {
            "forward": {
                "chars_per_sec": 180.5,
                "tokens_per_sec": 32.1,
                "time_s": 1.385
            },
            "roundtrip": {
                "chars_per_sec": 195.2,
                "tokens_per_sec": 35.8,
                "time_s": 1.256
            }
        },
        "quality": {
            "bleu_score": 45.2,
            "chrf_score": 62.8,
            "char_length_similarity_pct": 92.5,
            "forward_output_chars": 245,
            "forward_output_tokens": 42,
            "roundtrip_output_chars": 248,
            "roundtrip_output_tokens": 44
        },
        "memory": {
            "baseline_rss_mb": 450.2,
            "after_forward_rss_mb": 650.8,
            "peak_rss_mb": 680.5,
            "translation_increase_mb": 200.6,
            "peak_increase_mb": 230.3
        },
        "vram": {
            "baseline_used_mb": 0,
            "after_forward_used_mb": 380.5,
            "peak_used_mb": 420.3,
            "total_mb": 8192
        },
        "end_to_end_time_s": 2.641,
        "outputs": {
            "forward_translation": "...",
            "roundtrip_translation": "..."
        }
    }
    """
    body = request.get_json() or {}
    
    src_lang = body.get("src_lang")
    tgt_lang = body.get("tgt_lang", "en")
    use_onnx = body.get("use_onnx", USE_ONNX_TRANSLATOR)
    
    if not src_lang:
        return jsonify({"error": "src_lang required"}), 400
    
    # Normalize language codes
    src_lang = LANG_ALIASES.get(src_lang, src_lang)
    tgt_lang = LANG_ALIASES.get(tgt_lang, tgt_lang)
    
    try:
        results = benchmark_translator_metrics(
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            use_onnx=use_onnx
        )
        
        if "error" in results:
            return jsonify(results), 400
        
        return jsonify({"ok": True, **results})
    except Exception as e:
        return jsonify({
            "error": "benchmark failed",
            "details": str(e)
        }), 500


@bp.post("/rag_metrics")
def ep_rag_metrics():
    """
    Measure detailed RAG performance metrics.
    
    Request body:
    {
        "llm_name": "Qwen2-500M-Instruct-GGUF",  // optional, for RAG impact analysis
        "n_ctx": 4096,                           // optional, default 4096
        "n_gpu_layers": -1                       // optional, default -1 (all)
    }
    
    Returns:
    {
        "ok": true,
        "documents_indexed": 10,
        "indexing_time_s": 0.234,
        "index_size": {
            "index_file_mb": 0.0012,
            "metadata_file_mb": 0.0008,
            "total_mb": 0.0020
        },
        "retrieval_performance": {
            "avg_query_time_ms": 2.5,
            "min_query_time_ms": 1.8,
            "max_query_time_ms": 3.2,
            "topk_avg_times_ms": {
                "1": 1.9,
                "3": 2.5,
                "5": 3.1
            }
        },
        "memory": {
            "baseline_rss_mb": 450.2,
            "after_indexing_rss_mb": 452.8,
            "indexing_increase_mb": 2.6
        },
        "relevance": {
            "avg_recall_at_3": 0.85,
            "queries_evaluated": 4,
            "perfect_recalls": 2
        },
        "rag_impact": {
            "query": "What is quantum computing and how does it work?",
            "answer_without_rag": "...",
            "answer_with_rag": "...",
            "answer_length_diff": 45,
            "inference_time_without_rag_s": 1.2,
            "inference_time_with_rag_s": 1.5,
            "rag_overhead_s": 0.3,
            "contexts_used": 3
        },
        "restoration": {
            "original_doc_count": 5,
            "restored_doc_count": 5
        }
    }
    
    Note: This endpoint temporarily clears RAG data, runs benchmarks with demo data,
    then restores the original RAG documents.
    """
    body = request.get_json() or {}
    
    llm_name = body.get("llm_name")
    n_ctx = int(body.get("n_ctx", 4096))
    n_gpu_layers = int(body.get("n_gpu_layers", -1))
    
    try:
        results = benchmark_rag_metrics(
            llm_name=llm_name,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers
        )
        
        if "error" in results:
            return jsonify(results), 400
        
        return jsonify({"ok": True, **results})
    except Exception as e:
        return jsonify({
            "error": "benchmark failed",
            "details": str(e)
        }), 500


@bp.post("/benchmark/cache")
def ep_benchmark_cache():
    """
    Benchmark query cache performance: hit ratio, latency improvement, memory usage.
    
    Measures:
    - Baseline latency (RAG only, no cache)
    - Cache miss latency (first query)
    - Cache hit latency (repeated queries)
    - Hit ratio, speedup, time saved
    
    Request body:
    {
        "queries": [
            {"text": "What is machine learning?", "lang": "en"},
            {"text": "मशीन लर्निंग क्या है?", "lang": "hi"},
            ...
        ],
        "num_repeats": 3,           // optional, default 3 (times to repeat for hit measurement)
        "similarity_threshold": 0.80 // optional, cache hit threshold
    }
    
    Returns:
    {
        "ok": true,
        "summary": {
            "total_queries": 2,
            "total_runs": 8,              // queries * (1 baseline + 1 miss + num_repeats)
            "num_repeats": 3,
            "avg_hit_ratio": 0.95,        // 0-1
            "avg_latency_no_cache_ms": 245.5,
            "avg_latency_cache_miss_ms": 260.2,
            "avg_latency_cache_hit_ms": 15.3,
            "speedup_hit_vs_miss": 17.0,
            "speedup_hit_vs_no_cache": 16.0,
            "total_time_saved_ms": 690.4,
            "cache_efficiency_percent": 93.7
        },
        "per_query": [
            {
                "text": "What is machine learning?",
                "lang": "en",
                "latency_no_cache_ms": 245.5,
                "latency_cache_miss_ms": 260.2,
                "latency_cache_hit_ms": [15.1, 15.5, 15.2],
                "avg_latency_cache_hit_ms": 15.3,
                "hit_ratio": 0.95,
                "speedup_hit_vs_miss": 17.0,
                "speedup_hit_vs_no_cache": 16.0
            },
            ...
        ],
        "cache_state": {
            "final_cache_entries": 2,
            "max_entries": 1000,
            "cache_utilization_percent": 0.2
        }
    }
    """
    body = request.get_json() or {}
    
    queries = body.get("queries", [])
    num_repeats = int(body.get("num_repeats", 3))
    similarity_threshold = float(body.get("similarity_threshold", 0.80))
    
    if not queries or not isinstance(queries, list):
        return jsonify({"error": "queries must be a non-empty list of {text, lang} dicts"}), 400
    
    try:
        results = benchmark_query_cache(
            queries=queries,
            num_repeats=num_repeats,
            cache_similarity_threshold=similarity_threshold
        )
        
        return jsonify({"ok": True, **results})
    except Exception as e:
        return jsonify({
            "error": "cache benchmark failed",
            "details": str(e)
        }), 500
