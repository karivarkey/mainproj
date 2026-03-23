import json
import re
from flask import request, jsonify, Response, stream_with_context
from . import bp
import app.config as app_config
from app.config import (
    LANG_MAP, LANG_ALIASES, NLLB_LANG_MAP,
    USE_ONNX_TRANSLATOR, ONNX_MODEL_FAMILY,
    ONNX_M2M_LANG_MAP, ONNX_NLLB_LANG_MAP,
    TRANSLATION_MAX_NEW_TOKENS,
    TRANSLATION_MIN_NEW_TOKENS,
    TRANSLATION_SHORT_TEXT_MAX_NEW_TOKENS,
    TRANSLATION_SHORT_TEXT_WORDS,
)
from app.services.translator_service import (
    translate, detect_supported_language, 
    unload_translator, preload_translator, local_translator_path, is_translator_loaded
)
from app.services.onnx_translator_service import (
    translate_onnx, get_onnx_status, 
    unload_onnx_translator, preload_onnx_translator, ensure_onnx_models, is_onnx_family_loaded
)
from app.services.onnx_model_download_service import (
    get_onnx_catalog, download_onnx_models, 
    ensure_onnx_tokenizer, list_downloaded_onnx_model_files, 
    ensure_default_onnx_models
)

ACTIVE_TRANSLATOR = "onnx" if USE_ONNX_TRANSLATOR else "nllb"
ACTIVE_ONNX_FAMILY = ONNX_MODEL_FAMILY


def _translation_token_limit_for_text(text: str) -> int:
    words = max(1, len((text or "").split()))
    if words <= TRANSLATION_SHORT_TEXT_WORDS:
        return min(TRANSLATION_SHORT_TEXT_MAX_NEW_TOKENS, max(TRANSLATION_MIN_NEW_TOKENS, words * 2 + 6))
    return max(TRANSLATION_MIN_NEW_TOKENS, min(TRANSLATION_MAX_NEW_TOKENS, words * 2))


def _normalize_onnx_family(family: str | None) -> str:
    selected = (family or ACTIVE_ONNX_FAMILY or "m2m").strip().lower()
    return selected if selected in ("m2m", "nllb") else "m2m"


def _onnx_backend_name(family: str | None = None) -> str:
    selected = _normalize_onnx_family(family)
    return "m2m_onnx" if selected == "m2m" else "nllb_onnx"


def _onnx_lang_map_for_family(family: str | None = None) -> dict:
    return ONNX_M2M_LANG_MAP if _normalize_onnx_family(family) == "m2m" else ONNX_NLLB_LANG_MAP


def get_active_onnx_family() -> str:
    return _normalize_onnx_family(ACTIVE_ONNX_FAMILY)

def _get_effective_active_translator() -> str:
    global ACTIVE_TRANSLATOR
    if ACTIVE_TRANSLATOR == "none":
        return "none"
    if ACTIVE_TRANSLATOR == "onnx":
        try:
            if get_onnx_status().get("available"):
                return "onnx"
        except Exception:
            pass
        return "nllb"
    return "nllb"


def _public_translator_name(translator_key: str) -> str:
    if translator_key == "onnx":
        return _onnx_backend_name(ACTIVE_ONNX_FAMILY)
    if translator_key == "nllb":
        return "nllb"
    return "none"

@bp.get("/translator_status")
def ep_translator_status():
    global ACTIVE_ONNX_FAMILY
    ACTIVE_ONNX_FAMILY = _normalize_onnx_family(ACTIVE_ONNX_FAMILY)
    onnx_status = get_onnx_status(onnx_family=ACTIVE_ONNX_FAMILY)
    onnx_status_m2m = get_onnx_status(onnx_family="m2m")
    onnx_status_nllb = get_onnx_status(onnx_family="nllb")
    active_key = _get_effective_active_translator()
    active = _public_translator_name(active_key)
    loaded_key = "none"
    if is_onnx_family_loaded(ACTIVE_ONNX_FAMILY):
        loaded_key = "onnx"
    elif is_translator_loaded(app_config.NLLB_MODEL):
        loaded_key = "nllb"

    nllb_path = local_translator_path(app_config.NLLB_MODEL)
    nllb_downloaded = nllb_path.exists() and any(nllb_path.iterdir())
    onnx_downloaded_models = list_downloaded_onnx_model_files(family=ACTIVE_ONNX_FAMILY)
    nllb_downloaded_models = [app_config.NLLB_MODEL] if nllb_downloaded else []

    return jsonify({
        "active_translator": active,
        "active_translator_key": active_key,
        "loaded_translator_key": loaded_key,
        "onnx_backend_name": _onnx_backend_name(ACTIVE_ONNX_FAMILY),
        "active_onnx_family": ACTIVE_ONNX_FAMILY,
        "onnx": onnx_status,
        "onnx_families": {
            "m2m": onnx_status_m2m,
            "nllb": onnx_status_nllb,
        },
        "nllb": {
            "available": True,
            "model": app_config.NLLB_MODEL,
            "downloaded": nllb_downloaded,
            "local_path": str(nllb_path),
        },
        "downloaded_models": {
            "onnx": onnx_downloaded_models,
            _onnx_backend_name(ACTIVE_ONNX_FAMILY): onnx_downloaded_models,
            "nllb": nllb_downloaded_models,
            "all": [
                *[f"{_onnx_backend_name(ACTIVE_ONNX_FAMILY)}:{name}" for name in onnx_downloaded_models],
                *[f"nllb:{name}" for name in nllb_downloaded_models],
            ],
        },
    })

@bp.post("/toggle_translator")
def ep_toggle_translator():
    global ACTIVE_TRANSLATOR, ACTIVE_ONNX_FAMILY
    body = request.get_json() or {}
    use_onnx = body.get("use_onnx", _get_effective_active_translator() == "onnx")
    requested_onnx_family = _normalize_onnx_family(body.get("onnx_family"))
    onnx_status = get_onnx_status(onnx_family=requested_onnx_family)
    if use_onnx and not onnx_status["available"]:
        return jsonify({"error": "ONNX models not available", "details": onnx_status}), 503
    preload_details = None

    try:
        if use_onnx:
            ACTIVE_ONNX_FAMILY = requested_onnx_family
            unload_translator()
            unload_onnx_translator(onnx_family=ACTIVE_ONNX_FAMILY)
            ACTIVE_TRANSLATOR = "onnx"
            preload_details = preload_onnx_translator(onnx_family=ACTIVE_ONNX_FAMILY)
        else:
            unload_onnx_translator(onnx_family=ACTIVE_ONNX_FAMILY)
            ACTIVE_TRANSLATOR = "nllb"
            preload_details = preload_translator()
    except Exception as e:
        return jsonify({"error": f"Failed to load translator backend: {e}"}), 503
    
    return jsonify({
        "ok": True,
        "active_translator": _public_translator_name(ACTIVE_TRANSLATOR),
        "active_translator_key": ACTIVE_TRANSLATOR,
        "onnx_backend_name": _onnx_backend_name(ACTIVE_ONNX_FAMILY),
        "active_onnx_family": ACTIVE_ONNX_FAMILY,
        "preloaded": True,
        "preload_details": preload_details,
    })


@bp.post("/unload_translator")
def ep_unload_translator():
    global ACTIVE_TRANSLATOR
    unload_translator()
    unload_onnx_translator(onnx_family=ACTIVE_ONNX_FAMILY)
    ACTIVE_TRANSLATOR = "none"
    return jsonify({
        "ok": True,
        "active_translator": "none",
        "active_translator_key": "none",
        "onnx_backend_name": _onnx_backend_name(ACTIVE_ONNX_FAMILY),
        "active_onnx_family": ACTIVE_ONNX_FAMILY,
    })

@bp.post("/translate")
def ep_translate():
    global ACTIVE_ONNX_FAMILY
    body = request.get_json() or {}
    text = body.get("text")
    target = (body.get("target") or "en").lower()
    stream = bool(body.get("stream", True))
    requested_max_tokens = int(body.get("max_new_tokens", TRANSLATION_MAX_NEW_TOKENS))
    use_onnx = body.get("use_onnx", USE_ONNX_TRANSLATOR)
    onnx_family = _normalize_onnx_family(body.get("onnx_family") or ACTIVE_ONNX_FAMILY)

    if not text:
        return jsonify({"error": "text required"}), 400

    src_lang_key = detect_supported_language(text)
    if not src_lang_key:
        return jsonify({"error": "could not auto-detect a supported language"}), 400

    if use_onnx:
        target_map = _onnx_lang_map_for_family(onnx_family)
        translate_fn = translate_onnx
        backend_key = "onnx"
        backend = _onnx_backend_name(onnx_family)
        src_code = src_lang_key
        if src_lang_key not in target_map:
            return jsonify({"error": f"Language '{src_lang_key}' not supported by ONNX family '{onnx_family}'"}), 400
        ACTIVE_ONNX_FAMILY = onnx_family
    else:
        target_map = NLLB_LANG_MAP
        translate_fn = translate
        backend_key = "nllb"
        backend = "nllb"
        src_code, _ = LANG_MAP[src_lang_key]

    target_key = LANG_ALIASES.get(target, target)
    target_code = target_map.get(target_key, target_key)
    max_tokens = min(requested_max_tokens, _translation_token_limit_for_text(text))

    sentence_end_re = re.compile(r"(.+?[.!?](?:\"|'|”)?)(\s+|$)", re.S)
    def iter_sentences(blob: str):
        buffer = blob
        while True:
            match = sentence_end_re.search(buffer)
            if not match: break
            sent = match.group(1).strip()
            buffer = buffer[match.end():]
            if sent: yield sent
        if buffer.strip(): yield buffer.strip()

    if not stream:
        combined = (
            translate_fn(text, src_code, target_code, max_tokens, onnx_family=onnx_family)
            if use_onnx
            else translate_fn(text, src_code, target_code, max_tokens)
        )
        return jsonify({
            "translated_text": combined,
            "detected_lang": src_lang_key,
            "backend": backend,
            "backend_key": backend_key,
        })

    def event_stream():
        yield f"data: {json.dumps({'type': 'meta', 'backend': backend, 'backend_key': backend_key, 'onnx_family': onnx_family if use_onnx else None})}\n\n"
        for idx, sent in enumerate(iter_sentences(text), start=1):
            sentence_token_limit = min(max_tokens, _translation_token_limit_for_text(sent))
            translated = (
                translate_fn(sent, src_code, target_code, sentence_token_limit, onnx_family=onnx_family)
                if use_onnx
                else translate_fn(sent, src_code, target_code, sentence_token_limit)
            )
            yield f"data: {json.dumps({'type': 'sentence', 'index': idx, 'translated': translated})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(stream_with_context(event_stream()), content_type="text/event-stream")

# Catalog and Download Aliases
@bp.get("/onnx_models/catalog")
@bp.get("/api/onnx_models/catalog")
def ep_onnx_models_catalog():
    refresh = str(request.args.get("refresh", "false")).lower() in ("1", "true", "yes")
    family = (request.args.get("family") or ONNX_MODEL_FAMILY).strip().lower()
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

@bp.post("/onnx_models/download")
@bp.post("/api/onnx_models/download")
def ep_onnx_models_download():
    body = request.get_json() or {}
    files = body.get("files")
    family = (body.get("family") or ONNX_MODEL_FAMILY).strip().lower()
    return jsonify({"ok": True, **download_onnx_models(selected_files=files, family=family)})


@bp.post("/onnx_tokenizer/ensure")
@bp.post("/api/onnx_tokenizer/ensure")
def ep_onnx_tokenizer_ensure():
    body = request.get_json(silent=True) or {}
    family = (body.get("family") or ONNX_MODEL_FAMILY).strip().lower()
    force_download = bool(body.get("force_download", False))
    result = ensure_onnx_tokenizer(force_download=force_download, family=family)
    return jsonify({"ok": True, **result})