import os
from pathlib import Path
import torch

os.environ["TRANSFORMERS_NO_TF"] = "1"

DEVICE = "gpu" if torch.cuda.is_available() else "cpu"

BASE = Path("models")
LLM_DIR = BASE / "llms"
TRANS_DIR = BASE / "translators"
CACHE_FILE = BASE / "cache.json"

RAG_DIR = BASE / "rag"
RAG_INDEX_FILE = RAG_DIR / "rag.index"
RAG_META_FILE = RAG_DIR / "metadata.json"

# IndicTrans2 language codes (source_script, target_script)
# Using single bidirectional model: ai4bharat/indictrans2-en-indic-dist-200M
LANG_MAP = {
    "hi": ("hin_Deva", "eng_Latn"),
    "ml": ("mal_Mlym", "eng_Latn"),
    "ta": ("tam_Taml", "eng_Latn"),
    "bn": ("ben_Beng", "eng_Latn"),
    "te": ("tel_Telu", "eng_Latn"),
    "gu": ("guj_Gujr", "eng_Latn"),
    "kn": ("kan_Knda", "eng_Latn"),
    "mr": ("mar_Deva", "eng_Latn"),
    "pa": ("pan_Guru", "eng_Latn"),
}



# Default IndicTrans2 model (supports both directions)
DEFAULT_TRANSLATOR_MODEL = "ai4bharat/indictrans2-en-indic-dist-200M"

def ensure_dirs():
    BASE.mkdir(exist_ok=True)
    LLM_DIR.mkdir(parents=True, exist_ok=True)
    TRANS_DIR.mkdir(parents=True, exist_ok=True)
    RAG_DIR.mkdir(exist_ok=True)

# Local llama binaries directory (release build). If set, use these binaries.
# Example: set env LLAMA_BIN_DIR to something like models/llama-bin or tools/llama
LLAMA_BIN_DIR = Path("C:/Users/karivarkey/Documents/code/mainproj/llama-cli")
