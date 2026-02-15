import os
from pathlib import Path
import torch

os.environ["TRANSFORMERS_NO_TF"] = "1"

# Device configuration
# Set CPU_ONLY=1 or CPU_ONLY=true in environment to force CPU inference (no GPU/VRAM usage)
CPU_ONLY = os.environ.get("CPU_ONLY", "false").lower() in ("1", "true", "yes")
DEVICE = "cpu" if CPU_ONLY else ("cuda:0" if torch.cuda.is_available() else "cpu")

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

# Language configuration used by /infer. LANG_CONF is the single source of truth; LANG_MAP and
# LANG_ALIASES are derived for backwards compatibility and convenience.
LANG_CONF = {
    # Indo-Aryan
    "hi": {"src": "hin_Deva", "tgt": "eng_Latn", "aliases": ["hin"]},
    "bn": {"src": "ben_Beng", "tgt": "eng_Latn", "aliases": ["ben"]},
    "mr": {"src": "mar_Deva", "tgt": "eng_Latn", "aliases": ["mar"]},
    "gu": {"src": "guj_Gujr", "tgt": "eng_Latn", "aliases": ["guj"]},
    "pa": {"src": "pan_Guru", "tgt": "eng_Latn", "aliases": ["pan"]},
    "ur": {"src": "urd_Arab", "tgt": "eng_Latn", "aliases": ["urd"]},
    "as": {"src": "asm_Beng", "tgt": "eng_Latn", "aliases": ["asm"]},
    "bho": {"src": "bho_Deva", "tgt": "eng_Latn", "aliases": []},
    "mag": {"src": "mag_Deva", "tgt": "eng_Latn", "aliases": []},
    "mai": {"src": "mai_Deva", "tgt": "eng_Latn", "aliases": []},
    "hne": {"src": "hne_Deva", "tgt": "eng_Latn", "aliases": []},
    "or": {"src": "ory_Orya", "tgt": "eng_Latn", "aliases": ["ory"]},
    "ks_ar": {"src": "kas_Arab", "tgt": "eng_Latn", "aliases": []},
    "ks_de": {"src": "kas_Deva", "tgt": "eng_Latn", "aliases": []},
    "sd": {"src": "snd_Arab", "tgt": "eng_Latn", "aliases": ["snd"]},
    "sa": {"src": "san_Deva", "tgt": "eng_Latn", "aliases": []},
    "sat": {"src": "sat_Olck", "tgt": "eng_Latn", "aliases": []},
    "mni": {"src": "mni_Beng", "tgt": "eng_Latn", "aliases": []},
    # Dravidian
    "ta": {"src": "tam_Taml", "tgt": "eng_Latn", "aliases": ["tam"]},
    "te": {"src": "tel_Telu", "tgt": "eng_Latn", "aliases": ["tel"]},
    "kn": {"src": "kan_Knda", "tgt": "eng_Latn", "aliases": ["kan"]},
    "ml": {"src": "mal_Mlym", "tgt": "eng_Latn", "aliases": ["mal"]},
}

LANG_ALIASES = {}
for key, conf in LANG_CONF.items():
    for alias in {key, *conf.get("aliases", [])}:
        LANG_ALIASES[alias] = key

# Legacy tuple map retained for existing call sites
LANG_MAP = {key: (conf["src"], conf["tgt"]) for key, conf in LANG_CONF.items()}

# RAG configuration
RAG_EMBEDDING_MODEL = "sentence-transformers/paraphrase-mpnet-base-v2"
RAG_EMBEDDING_DIM = 768
RAG_SIMILARITY_THRESHOLD = 0.35
RAG_TOP_K = 3

# Query cache configuration (caches similar query embeddings → RAG results)
QUERY_CACHE_FILE = RAG_DIR / "query_cache.json"
QUERY_CACHE_SIMILARITY_THRESHOLD = 0.80  # min cosine similarity to reuse cached RAG docs
QUERY_CACHE_MAX_ENTRIES = 1000  # max number of queries to cache
QUERY_CACHE_ENABLED = os.environ.get("QUERY_CACHE_ENABLED", "true").lower() in ("1", "true", "yes")

# LLM configuration defaults
LLM_DEFAULT_N_CTX = 4096
LLM_DEFAULT_N_GPU_LAYERS = 0 if CPU_ONLY else -1  # 0 = CPU only, -1 = max GPU offload
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
    RAG_DIR.mkdir(exist_ok=True)  # also creates query cache directory
