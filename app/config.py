import os
from pathlib import Path
import torch

os.environ["TRANSFORMERS_NO_TF"] = "1"

# Device configuration
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Directory paths
BASE = Path("models")
LLM_DIR = BASE / "llms"
TRANS_DIR = BASE / "translators"
CACHE_FILE = BASE / "cache.json"

RAG_DIR = BASE / "rag"
RAG_INDEX_FILE = RAG_DIR / "rag.index"
RAG_META_FILE = RAG_DIR / "metadata.json"

# NLLB Translator configuration
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
TRANSLATOR_QUANTIZE = os.environ.get("TRANSLATOR_QUANTIZE", "8bit")  # 'none', '8bit', or '4bit'

# NLLB language codes (ISO 639-3 with script)
NLLB_LANG_MAP = {
    "hi": "hin_Deva",      # Hindi
    "en": "eng_Latn",      # English
    "ta": "tam_Taml",      # Tamil
    "te": "tel_Telu",      # Telugu
    "kn": "kan_Knda",      # Kannada
    "ml": "mal_Mlym",      # Malayalam
    "mr": "mar_Deva",      # Marathi
    "gu": "guj_Gujr",      # Gujarati
    "bn": "ben_Beng",      # Bengali
    "pa": "pan_Guru",      # Punjabi
    "ur": "urd_Arab",      # Urdu
    "as": "asm_Beng",      # Assamese
    "or": "ory_Orya",      # Odia
    "sa": "san_Deva",      # Sanskrit
    "fr": "fra_Latn",      # French
    "de": "deu_Latn",      # German
    "es": "spa_Latn",      # Spanish
    "pt": "por_Latn",      # Portuguese
    "ja": "jpn_Jpan",      # Japanese
    "zh": "zho_Hans",      # Chinese (Simplified)
    "ru": "rus_Cyrl",      # Russian
}

# Legacy LANG_MAP for /infer endpoint (IndicTrans2 format: source, target pairs)
# Note: This is currently unused in favor of NLLB_LANG_MAP
LANG_MAP = {
    # Indo-Aryan
    "hi": ("hin_Deva", "eng_Latn"),      # Hindi
    "bn": ("ben_Beng", "eng_Latn"),      # Bengali
    "mr": ("mar_Deva", "eng_Latn"),      # Marathi
    "gu": ("guj_Gujr", "eng_Latn"),      # Gujarati
    "pa": ("pan_Guru", "eng_Latn"),      # Punjabi
    "ur": ("urd_Arab", "eng_Latn"),      # Urdu
    "as": ("asm_Beng", "eng_Latn"),      # Assamese
    "bho": ("bho_Deva", "eng_Latn"),     # Bhojpuri
    "mag": ("mag_Deva", "eng_Latn"),     # Magahi
    "mai": ("mai_Deva", "eng_Latn"),     # Maithili
    "hne": ("hne_Deva", "eng_Latn"),     # Chhattisgarhi
    "or": ("ory_Orya", "eng_Latn"),      # Odia
    "ks_ar": ("kas_Arab", "eng_Latn"),   # Kashmiri (Arabic)
    "ks_de": ("kas_Deva", "eng_Latn"),   # Kashmiri (Devanagari)
    "sd": ("snd_Arab", "eng_Latn"),      # Sindhi
    "sa": ("san_Deva", "eng_Latn"),      # Sanskrit
    "sat": ("sat_Olck", "eng_Latn"),     # Santali
    "mni": ("mni_Beng", "eng_Latn"),     # Manipuri
    # Dravidian
    "ta": ("tam_Taml", "eng_Latn"),      # Tamil
    "te": ("tel_Telu", "eng_Latn"),      # Telugu
    "kn": ("kan_Knda", "eng_Latn"),      # Kannada
    "ml": ("mal_Mlym", "eng_Latn"),      # Malayalam
}

# RAG configuration
RAG_EMBEDDING_MODEL = "sentence-transformers/paraphrase-mpnet-base-v2"
RAG_EMBEDDING_DIM = 768
RAG_SIMILARITY_THRESHOLD = 0.35
RAG_TOP_K = 3

# LLM configuration defaults
LLM_DEFAULT_N_CTX = 4096
LLM_DEFAULT_N_GPU_LAYERS = -1  # -1 = max GPU offload
LLM_DEFAULT_TEMPERATURE = 0.3
LLM_DEFAULT_TOP_P = 0.9
LLM_DEFAULT_TOP_K = 40
LLM_DEFAULT_REPEAT_PENALTY = 1.1
LLM_DEFAULT_MAX_TOKENS = 128

def ensure_dirs():
    """Create required directories if they don't exist."""
    BASE.mkdir(exist_ok=True)
    LLM_DIR.mkdir(parents=True, exist_ok=True)
    TRANS_DIR.mkdir(parents=True, exist_ok=True)
    RAG_DIR.mkdir(exist_ok=True)
