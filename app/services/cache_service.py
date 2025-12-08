import json
from app.config import CACHE_FILE

def load_cache():
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            pass
    return {"llms": [], "translators": []}

def save_cache(cache):
    CACHE_FILE.write_text(json.dumps(cache, indent=2))

model_cache = load_cache()