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
RAG_EMBEDDING_LOCAL_DIR = RAG_DIR / "embedding_model"
RAG_EMBEDDING_CACHE_DIR = RAG_DIR / "hf_cache"

# NLLB Translator configuration
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
TRANSLATOR_QUANTIZE = os.environ.get("TRANSLATOR_QUANTIZE", "8bit")  # 'none', '8bit', or '4bit'

# ONNX Translator configuration
USE_ONNX_TRANSLATOR = os.environ.get("USE_ONNX_TRANSLATOR", "true").lower() in ("1", "true", "yes")  # Default to ONNX
ONNX_MODEL_FAMILY = os.environ.get("ONNX_MODEL_FAMILY", "m2m").strip().lower()  # m2m | nllb
if ONNX_MODEL_FAMILY not in ("m2m", "nllb"):
    ONNX_MODEL_FAMILY = "m2m"

ONNX_DRIVE_FOLDERS = {
    "m2m": {
        "url": "https://drive.google.com/drive/folders/1tN4wqRMMCWfdy-nXOCjaMTv-H5Paj8Zi?usp=drive_link",
        "id": "1tN4wqRMMCWfdy-nXOCjaMTv-H5Paj8Zi",
    },
    "nllb": {
        "url": "https://drive.google.com/drive/folders/1WCd3JSVEGF-r38j1KDSykjs23I8_kfVq?usp=sharing",
        "id": "1WCd3JSVEGF-r38j1KDSykjs23I8_kfVq",
    },
}

ONNX_FAMILY_CONFIG = {
    "m2m": {
        "models_dir": TRANS_DIR / "m2m100_onnx",
        "tokenizer_dir": TRANS_DIR / "m2m100_tokenizer",
        "tokenizer_model": "facebook/m2m100_418M",
        "encoder_model": "m2m100_encoder_w8a32_SAFE.onnx",
        "decoder_model": "m2m100_decoder_w8a32.onnx",
        "lm_head_model": "m2m100_lm_head.onnx",
        "default_files": [
            "m2m100_encoder_w8a32_SAFE.onnx",
            "m2m100_decoder_w8a32.onnx",
            "m2m100_lm_head.onnx",
            "m2m100_lm_head.onnx.data",
        ],
    },
    "nllb": {
        "models_dir": TRANS_DIR / "nllb_onnx",
        "tokenizer_dir": TRANS_DIR / "nllb_onnx_tokenizer",
        "tokenizer_model": "facebook/nllb-200-distilled-600M",
        "encoder_model": "nllb_encoder_w8a32_safe.onnx",
        "decoder_model": "nllb_decoder_w8a32_safe.onnx",
        "lm_head_model": "nllb_lm_head_w8a32_safe.onnx",
        "default_files": [
            "nllb_encoder_w8a32_safe.onnx",
            "nllb_decoder_w8a32_safe.onnx",
            "nllb_lm_head_w8a32_safe.onnx",
        ],
    },
}

_onnx_selected = ONNX_FAMILY_CONFIG[ONNX_MODEL_FAMILY]
ONNX_MODELS_DIR = _onnx_selected["models_dir"]
ONNX_TOKENIZER_DIR = _onnx_selected["tokenizer_dir"]  # Local tokenizer for offline use
ONNX_TOKENIZER_MODEL = _onnx_selected["tokenizer_model"]
ONNX_ENCODER_MODEL = _onnx_selected["encoder_model"]
ONNX_DECODER_MODEL = _onnx_selected["decoder_model"]
ONNX_LM_HEAD_MODEL = _onnx_selected["lm_head_model"]
ONNX_DEFAULT_MODEL_FILES = list(_onnx_selected["default_files"])
ONNX_DRIVE_FOLDER_URL = ONNX_DRIVE_FOLDERS[ONNX_MODEL_FAMILY]["url"]
ONNX_DRIVE_FOLDER_ID = ONNX_DRIVE_FOLDERS[ONNX_MODEL_FAMILY]["id"]

# M2M-100 language codes (ISO 639-1, simple two-letter codes)
# M2M-100 tokenizer uses just "hi", "en", "ta", etc.
ONNX_M2M_LANG_MAP = {
    "hi": "hi",            # Hindi
    "en": "en",            # English
    "ta": "ta",            # Tamil
    "te": "te",            # Telugu
    "kn": "kn",            # Kannada
    "ml": "ml",            # Malayalam
    "mr": "mr",            # Marathi
    "gu": "gu",            # Gujarati
    "bn": "bn",            # Bengali
    "pa": "pa",            # Punjabi
    "ur": "ur",            # Urdu
    "as": "as",            # Assamese
    "or": "or",            # Odia
    "sa": "sa",            # Sanskrit
    "fr": "fr",            # French
    "de": "de",            # German
    "es": "es",            # Spanish
    "pt": "pt",            # Portuguese
    "ja": "ja",            # Japanese
    "zh": "zh",            # Chinese
    "ru": "ru",            # Russian
}

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

ONNX_NLLB_LANG_MAP = dict(NLLB_LANG_MAP)
ONNX_LANG_MAP = ONNX_M2M_LANG_MAP if ONNX_MODEL_FAMILY == "m2m" else ONNX_NLLB_LANG_MAP
ONNX_BACKEND_NAME = "m2m_onnx" if ONNX_MODEL_FAMILY == "m2m" else "nllb_onnx"

# ONNX Runtime CPU tuning (important for edge devices like Raspberry Pi)
ONNX_INTRA_OP_THREADS = max(1, int(os.environ.get("ONNX_INTRA_OP_THREADS", str(min(4, os.cpu_count() or 1)))))
ONNX_INTER_OP_THREADS = max(1, int(os.environ.get("ONNX_INTER_OP_THREADS", "1")))
ONNX_ENABLE_ALL_OPTIMIZATIONS = os.environ.get("ONNX_ENABLE_ALL_OPTIMIZATIONS", "true").lower() in ("1", "true", "yes")
ONNX_VERBOSE_LOGS = os.environ.get("ONNX_VERBOSE_LOGS", "false").lower() in ("1", "true", "yes")

# Language configuration used by /infer. LANG_CONF is the single source of truth; LANG_MAP and
# LANG_ALIASES are derived for backwards compatibility and convenience.
LANG_CONF = {
    # English
    "en": {"src": "eng_Latn", "tgt": "eng_Latn", "aliases": ["eng"]},
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
    # European languages
    "fr": {"src": "fra_Latn", "tgt": "eng_Latn", "aliases": []},
    "de": {"src": "deu_Latn", "tgt": "eng_Latn", "aliases": []},
    "es": {"src": "spa_Latn", "tgt": "eng_Latn", "aliases": []},
    "pt": {"src": "por_Latn", "tgt": "eng_Latn", "aliases": []},
    "ru": {"src": "rus_Cyrl", "tgt": "eng_Latn", "aliases": []},
    # East Asian languages
    "ja": {"src": "jpn_Jpan", "tgt": "eng_Latn", "aliases": []},
    "zh": {"src": "zho_Hans", "tgt": "eng_Latn", "aliases": []},
}

LANG_ALIASES = {}
for key, conf in LANG_CONF.items():
    for alias in {key, *conf.get("aliases", [])}:
        LANG_ALIASES[alias] = key

# Legacy tuple map retained for existing call sites
LANG_MAP = {key: (conf["src"], conf["tgt"]) for key, conf in LANG_CONF.items()}

# RAG configuration
RAG_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RAG_EMBEDDING_DIM = 384
RAG_SIMILARITY_THRESHOLD = 0.35
RAG_TOP_K = 3

# Query cache configuration
QUERY_CACHE_FILE = BASE / "query_cache.json"
QUERY_CACHE_SIMILARITY_THRESHOLD = 0.80
QUERY_CACHE_MAX_ENTRIES = 3
QUERY_CACHE_ENABLED = True

# LLM configuration defaults
LLM_DEFAULT_N_CTX = 4096
LLM_DEFAULT_N_GPU_LAYERS = 0 if CPU_ONLY else -1  # 0 = CPU only, -1 = max GPU offload
LLM_DEFAULT_TEMPERATURE = 0.45
LLM_DEFAULT_TOP_P = 0.92
LLM_DEFAULT_TOP_K = 40
LLM_DEFAULT_REPEAT_PENALTY = 1.15
LLM_DEFAULT_MAX_TOKENS = 384

# Translator generation limits. Lower defaults keep sentence-level streaming snappy.
TRANSLATION_MAX_NEW_TOKENS = max(16, int(os.environ.get("TRANSLATION_MAX_NEW_TOKENS", "48")))
TRANSLATION_MIN_NEW_TOKENS = max(6, int(os.environ.get("TRANSLATION_MIN_NEW_TOKENS", "12")))
TRANSLATION_SHORT_TEXT_MAX_NEW_TOKENS = max(12, int(os.environ.get("TRANSLATION_SHORT_TEXT_MAX_NEW_TOKENS", "28")))
TRANSLATION_SHORT_TEXT_WORDS = max(4, int(os.environ.get("TRANSLATION_SHORT_TEXT_WORDS", "14")))

# ONNX decode safeguards for low-power CPUs
ONNX_TRANSLATION_MAX_SOURCE_TOKENS = max(64, int(os.environ.get("ONNX_TRANSLATION_MAX_SOURCE_TOKENS", "192")))
ONNX_TRANSLATION_LENGTH_RATIO = max(1.0, float(os.environ.get("ONNX_TRANSLATION_LENGTH_RATIO", "1.35")))
ONNX_TRANSLATION_MIN_STEPS = max(4, int(os.environ.get("ONNX_TRANSLATION_MIN_STEPS", "8")))

def ensure_dirs():
    """Create required directories if they don't exist."""
    BASE.mkdir(exist_ok=True)
    LLM_DIR.mkdir(parents=True, exist_ok=True)
    TRANS_DIR.mkdir(parents=True, exist_ok=True)
    RAG_DIR.mkdir(exist_ok=True)
    RAG_EMBEDDING_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    RAG_EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)