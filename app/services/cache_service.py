import json
from app.config import CACHE_FILE


def _normalize_cache(raw):
    if not isinstance(raw, dict):
        raw = {}

    llms = raw.get("llms", [])
    if not isinstance(llms, list):
        llms = []

    normalized_llms = []
    seen_llms = set()
    for item in llms:
        if not isinstance(item, str):
            continue
        name = item.strip()
        if not name:
            continue
        if not name.endswith(".gguf"):
            name = f"{name}.gguf"
        if name not in seen_llms:
            seen_llms.add(name)
            normalized_llms.append(name)

    translators = raw.get("translators", [])
    if not isinstance(translators, list):
        translators = []

    normalized_translators = []
    seen_translators = set()
    for item in translators:
        if not isinstance(item, str):
            continue
        name = item.strip()
        if not name:
            continue
        if name not in seen_translators:
            seen_translators.add(name)
            normalized_translators.append(name)

    onnx_raw = raw.get("onnx", {})
    if not isinstance(onnx_raw, dict):
        onnx_raw = {}

    normalized_onnx = {}
    for family in ("m2m", "nllb"):
        family_raw = onnx_raw.get(family, {})
        if not isinstance(family_raw, dict):
            family_raw = {}

        files = family_raw.get("downloaded_files", [])
        if not isinstance(files, list):
            files = []

        normalized_files = []
        seen_files = set()
        for item in files:
            if not isinstance(item, str):
                continue
            name = item.strip()
            if not name:
                continue
            if name not in seen_files:
                seen_files.add(name)
                normalized_files.append(name)

        tokenizer_ready = bool(family_raw.get("tokenizer_ready", False))

        normalized_onnx[family] = {
            "downloaded_files": normalized_files,
            "tokenizer_ready": tokenizer_ready,
        }

    return {
        "llms": normalized_llms,
        "translators": normalized_translators,
        "onnx": normalized_onnx,
    }

def load_cache():
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    cache = None
    if CACHE_FILE.exists():
        try:
            content = CACHE_FILE.read_text().strip()
            if content:
                cache = _normalize_cache(json.loads(content))
            else:
                cache = _normalize_cache({})
        except Exception:
            cache = _normalize_cache({})
    else:
        cache = _normalize_cache({})

    save_cache(cache)
    return cache

def save_cache(cache):
    normalized = _normalize_cache(cache)
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(normalized, indent=2))


def set_onnx_family_cache(
    family: str,
    downloaded_files: list[str] | None = None,
    tokenizer_ready: bool | None = None,
):
    selected = (family or "").strip().lower()
    if selected not in ("m2m", "nllb"):
        return

    model_cache.setdefault("onnx", {})
    model_cache["onnx"].setdefault(selected, {
        "downloaded_files": [],
        "tokenizer_ready": False,
    })

    if downloaded_files is not None:
        deduped = []
        seen = set()
        for item in downloaded_files:
            if not isinstance(item, str):
                continue
            name = item.strip()
            if not name:
                continue
            if name in seen:
                continue
            seen.add(name)
            deduped.append(name)
        model_cache["onnx"][selected]["downloaded_files"] = deduped

    if tokenizer_ready is not None:
        model_cache["onnx"][selected]["tokenizer_ready"] = bool(tokenizer_ready)

    save_cache(model_cache)

model_cache = load_cache()