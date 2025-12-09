import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from app.config import DEVICE, TRANS_DIR
from app.services.cache_service import model_cache, save_cache

sys.excepthook = lambda exc_type, exc, tb: (
    print("UNCAUGHT ERROR:", exc_type.__name__, exc, file=sys.stderr),
    __import__('traceback').print_tb(tb)
)

translator_cache = {}

# NLLB model: smaller, faster, multilingual
NLLB_MODEL = "facebook/nllb-200-distilled-600M"

# NLLB language codes (ISO 639-3 with script)
NLLB_LANG_MAP = {
    "hi": "hin_Deva",      # Hindi
    "en": "eng_Latn",      # English
    "ta": "tam_Taml",      # Tamil
    "te": "tel_Telu",      # Telugu
    "ka": "kan_Knda",      # Kannada
    "ml": "mal_Mlym",      # Malayalam
    "mr": "mar_Deva",      # Marathi
    "gu": "guj_Gujr",      # Gujarati
    "bn": "ben_Beng",      # Bengali
    "pa": "pan_Guru",      # Punjabi
    "ur": "urd_Arab",      # Urdu
    "fr": "fra_Latn",      # French
    "de": "deu_Latn",      # German
    "es": "spa_Latn",      # Spanish
    "pt": "por_Latn",      # Portuguese
    "ja": "jpn_Jpan",      # Japanese
    "zh": "zho_Hans",      # Chinese (Simplified)
    "ru": "rus_Cyrl",      # Russian
}


def local_translator_path(model_id: str) -> Path:
    """Get local cache path for translator model."""
    return Path(TRANS_DIR) / model_id.replace("/", "__")


def download_translator(model_id: str, trust_remote_code=True) -> Path:
    """
    Download NLLB model weights to local cache.
    Tokenizer is always loaded from HuggingFace to avoid corruption.
    Wraps in try-catch to handle torchvision dependency issues.
    """
    path = local_translator_path(model_id)

    if path.exists() and any(path.iterdir()):
        if model_id not in model_cache.get("translators", []):
            model_cache.setdefault("translators", []).append(model_id)
            save_cache(model_cache)
        return path

    print(f"[NLLB] Downloading model weights for {model_id}...")

    try:
        # Load tokenizer (not saved locally)
        _ = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)

        # Load and save model weights locally
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code
        )

        path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(path)

        model_cache.setdefault("translators", []).append(model_id)
        save_cache(model_cache)
        print(f"[NLLB] Model saved to {path}")
        return path
    except RuntimeError as e:
        if "torchvision" in str(e) or "nms" in str(e):
            print(f"[NLLB] WARNING: torchvision/transformers version conflict. Attempting workaround...")
            print(f"[NLLB] Error: {e}")
            # If cached already, use it; otherwise this will fail on translate()
            if path.exists():
                return path
            raise RuntimeError(
                f"Cannot load NLLB model due to dependency conflict: {e}\n"
                "Try: pip install --upgrade transformers torch torchvision"
            ) from e
        raise

def get_translator(model_id: str):
    """
    Load NLLB tokenizer + model on forced CUDA when available.
    Keeps a single cached instance.
    """

    if model_id in translator_cache:
        return translator_cache[model_id]

    local_path = download_translator(model_id)

    print(f"[NLLB] Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    # FORCE CUDA
    use_cuda = torch.cuda.is_available()
    device = "cuda:0" if use_cuda else "cpu"

    print(f"[NLLB] Loading model from {local_path} on {device}...")

    if use_cuda:
        # Full model on GPU, FP16 to fit in 4GB VRAM
        model = AutoModelForSeq2SeqLM.from_pretrained(
            str(local_path),
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        model = model.to(device)
    else:
        # CPU fallback (slow)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            str(local_path),
            trust_remote_code=True,
            torch_dtype=torch.float32
        )

    model.eval()

    translator_cache[model_id] = (tok, model, device)
    print(f"[NLLB] Model ready on {device}.")
    return tok, model, device



def translate(text: str, src_lang: str, tgt_lang: str, max_tokens: int = 256) -> str:
    """
    Translate text from src_lang to tgt_lang using NLLB.
    
    Args:
        text: Text to translate
        src_lang: Source language code (e.g., "hi", "en") or NLLB code (e.g., "hin_Deva")
        tgt_lang: Target language code (e.g., "hi", "en") or NLLB code (e.g., "eng_Latn")
        max_tokens: Maximum output tokens
    
    Returns:
        Translated text
    """
    # Map short codes to NLLB codes if needed
    src_code = NLLB_LANG_MAP.get(src_lang, src_lang)
    tgt_code = NLLB_LANG_MAP.get(tgt_lang, tgt_lang)

    print(f"\n[NLLB] Translating {src_code} → {tgt_code}")
    print(f"Input: {text[:100]}..." if len(text) > 100 else f"Input: {text}")

    tok, model, device = get_translator(NLLB_MODEL)

    # Tokenize with source language token
    inputs = tok(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    if device.startswith("cuda"):
        inputs = {k: v.to(device) for k, v in inputs.items()}

    # Force target language at generation start
    gen_kwargs = {
        "forced_bos_token_id": tok.convert_tokens_to_ids(tgt_code),
        "max_new_tokens": max_tokens,
        "num_beams": 1,  # Greedy decode for speed
        "use_cache": True,  # Avoid meta tensor issues
    }

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    decoded = tok.batch_decode(out, skip_special_tokens=True)
    output = decoded[0].strip()

    print(f"Output: {output[:100]}..." if len(output) > 100 else f"Output: {output}\n")
    return output
