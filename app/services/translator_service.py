import sys
import torch
from pathlib import Path
from langdetect import detect, DetectorFactory, LangDetectException
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BitsAndBytesConfig
from app.config import (
    DEVICE,
    TRANS_DIR,
    NLLB_MODEL,
    NLLB_LANG_MAP,
    TRANSLATOR_QUANTIZE,
    CPU_ONLY,
    LANG_ALIASES,
)
from app.services.cache_service import model_cache, save_cache

sys.excepthook = lambda exc_type, exc, tb: (
    print("UNCAUGHT ERROR:", exc_type.__name__, exc, file=sys.stderr),
    __import__('traceback').print_tb(tb)
)

translator_cache = {}


def detect_supported_language(text: str):
    """
    Detect a supported language code from free-form text.

    Returns:
        The canonical LANG_CONF key (e.g., "hi") when detected and supported;
        otherwise None.
    """
    if not text or not text.strip():
        return None

    try:
        detected = detect(text)
    except LangDetectException:
        return None

    # langdetect may return variants like "zh-cn"; we only care about the prefix
    short = detected.split("-")[0].lower()
    return LANG_ALIASES.get(short)

# Fix seed for deterministic language detection
DetectorFactory.seed = 42


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

    cache_key = f"{model_id}__{TRANSLATOR_QUANTIZE}"
    if cache_key in translator_cache:
        return translator_cache[cache_key]

    local_path = download_translator(model_id)

    print(f"[NLLB] Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    # Device selection: respect CPU_ONLY flag
    use_cuda = torch.cuda.is_available() and not CPU_ONLY
    device = "cuda:0" if use_cuda else "cpu"

    print(f"[NLLB] Loading model from {local_path} on {device} (quantize={TRANSLATOR_QUANTIZE})...")

    # Attempt quantized load when requested and CUDA is available
    model = None
    if use_cuda and TRANSLATOR_QUANTIZE in ("8bit", "4bit"):
        try:
            if TRANSLATOR_QUANTIZE == "8bit":
                qconfig = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                qconfig = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )

            # Use device_map for automatic placement when quantized
            model = AutoModelForSeq2SeqLM.from_pretrained(
                str(local_path),
                trust_remote_code=True,
                quantization_config=qconfig,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            print(f"[NLLB] Loaded quantized model ({TRANSLATOR_QUANTIZE}).")
        except Exception as e:
            print(f"[NLLB] Quantized load failed: {e}. Falling back to FP16/FP32 load.")

    if model is None:
        # Fallback: full model load (FP16 on CUDA, FP32 on CPU)
        if use_cuda:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                str(local_path),
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            model = model.to(device)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                str(local_path),
                trust_remote_code=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )

    model.eval()

    translator_cache[cache_key] = (tok, model, device)
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


def unload_translator(model_id: str = NLLB_MODEL):
    """
    Unload a translator from cache to free VRAM/RAM.
    Safe for both quantized and non-quantized models.
    """
    cache_key = f"{model_id}__{TRANSLATOR_QUANTIZE}"
    entry = translator_cache.pop(cache_key, None)
    if entry:
        tok, model, device = entry
        try:
            del tok
            del model
        except Exception as e:
            print(f"[NLLB] Warning during unload: {e}")
        
        if torch.cuda.is_available() and not CPU_ONLY:
            torch.cuda.empty_cache()
        print(f"[NLLB] Unloaded translator: {model_id}")
        return True
    return False


def preload_translator(model_id: str = NLLB_MODEL):
    """Preload NLLB tokenizer and model without translating."""
    get_translator(model_id)
    return {
        "model": model_id,
    }
