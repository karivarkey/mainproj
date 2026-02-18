"""
ONNX Runtime-based M2M-100 Translator Service
Provides edge-optimized translation using quantized ONNX models.
Falls back gracefully if models are unavailable.
"""

import sys
import onnxruntime as ort
import numpy as np
from pathlib import Path
from app.config import (
    ONNX_MODELS_DIR,
    ONNX_TOKENIZER_DIR,
    ONNX_LANG_MAP,
    ONNX_ENCODER_MODEL,
    ONNX_DECODER_MODEL,
    ONNX_LM_HEAD_MODEL,
    DEVICE,
    CPU_ONLY,
)
from app.services.cache_service import model_cache, save_cache

# Global translator cache
onnx_translator_cache = {}

# Try to import tokenizer - we'll use transformers' tokenizer for now
try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False

tokenizer = None
device = "cpu"  # ONNX Runtime with CPU for now


def get_onnx_models_dir() -> Path:
    """Get the ONNX models directory."""
    return ONNX_MODELS_DIR


def ensure_onnx_models() -> bool:
    """Check if ONNX model files exist (checks for configured model variants)."""
    models_dir = get_onnx_models_dir()
    required = [
        models_dir / "encoder" / ONNX_ENCODER_MODEL,
        models_dir / "decoder" / ONNX_DECODER_MODEL,
        models_dir / "lm_head" / ONNX_LM_HEAD_MODEL,
    ]
    return all(f.exists() for f in required)


def load_onnx_tokenizer():
    """Load M2M-100 tokenizer from local directory (offline mode)."""
    global tokenizer
    if tokenizer is not None:
        return tokenizer
    
    if not TOKENIZER_AVAILABLE:
        raise RuntimeError("transformers library required for tokenization")
    
    try:
        print("[ONNX] Loading M2M-100 tokenizer from local directory...")
        tokenizer_path = str(ONNX_TOKENIZER_DIR)
        if not ONNX_TOKENIZER_DIR.exists():
            raise RuntimeError(f"Tokenizer directory not found: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        print("[ONNX] Tokenizer loaded (offline mode).")
        return tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer: {e}") from e


def get_onnx_session(model_path: str, providers=None):
    """
    Create or retrieve an ONNX Runtime session.
    
    Args:
        model_path: Path to .onnx file
        providers: List of execution providers (default: ['CPUExecutionProvider'])
    
    Returns:
        InferenceSession
    """
    cache_key = str(model_path)
    if cache_key in onnx_translator_cache:
        return onnx_translator_cache[cache_key]
    
    if providers is None:
        providers = ["CPUExecutionProvider"]
    
    try:
        session = ort.InferenceSession(model_path, providers=providers)
        onnx_translator_cache[cache_key] = session
        print(f"[ONNX] Loaded session for {Path(model_path).name}")
        return session
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model {model_path}: {e}") from e


def encode_with_onnx(text: str, src_lang: str) -> dict:
    """
    Encode input text using ONNX encoder.
    
    Args:
        text: Input text
        src_lang: Source language code (e.g., "hi", "en")
    
    Returns:
        Dict with encoder outputs
    """
    tok = load_onnx_tokenizer()
    
    # Map short codes to M2M-100 codes
    src_code = ONNX_LANG_MAP.get(src_lang, src_lang)
    
    print(f"[ONNX] Encoding {src_lang} ({src_code}): {text[:50]}...")
    
    # Tokenize with forced language token
    tok.src_lang = src_code
    inputs = tok(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    
    # Run encoder
    models_dir = get_onnx_models_dir()
    encoder_path = models_dir / "encoder" / ONNX_ENCODER_MODEL
    encoder_session = get_onnx_session(str(encoder_path))
    
    # Prepare inputs as numpy arrays
    input_ids = inputs["input_ids"].numpy().astype(np.int64)
    attention_mask = inputs["attention_mask"].numpy().astype(np.int64)
    
    encoder_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    
    encoder_outputs = encoder_session.run(None, encoder_inputs)
    
    return {
        "last_hidden_state": encoder_outputs[0],  # (batch, seq_len, hidden_size)
        "attention_mask": attention_mask,
        "tokenizer": tok,
    }


def decode_with_onnx(encoder_outputs: dict, tgt_lang: str, max_tokens: int = 256) -> str:
    """
    Decode encoder outputs to target language using ONNX decoder + LM head.
    Uses W8A32 quantized models for best performance.
    
    Args:
        encoder_outputs: Output from encode_with_onnx
        tgt_lang: Target language code (e.g., "en", "hi")
        max_tokens: Max tokens to generate
    
    Returns:
        Translated text
    """
    tok = encoder_outputs["tokenizer"]
    encoder_hidden = encoder_outputs["last_hidden_state"].astype(np.float32)
    encoder_attention_mask = encoder_outputs["attention_mask"]
    
    tgt_code = ONNX_LANG_MAP.get(tgt_lang, tgt_lang)
    print(f"[ONNX] Decoding to {tgt_lang} ({tgt_code})...")
    
    # Get forced BOS token (language ID)
    try:
        forced_bos = tok.get_lang_id(tgt_code)
        print(f"[ONNX] Forced BOS token ID: {forced_bos}")
    except Exception as e:
        print(f"[ONNX] Warning: Could not get language ID for {tgt_code}: {e}")
        forced_bos = tok.eos_token_id
    
    # Initialize decoder input with EOS token
    decoder_input_ids = np.array([[tok.eos_token_id]], dtype=np.int64)
    
    models_dir = get_onnx_models_dir()
    decoder_path = models_dir / "decoder" / ONNX_DECODER_MODEL
    lm_head_path = models_dir / "lm_head" / ONNX_LM_HEAD_MODEL
    
    decoder_session = get_onnx_session(str(decoder_path))
    lm_head_session = get_onnx_session(str(lm_head_path))
    
    # Greedy decoding loop
    for step in range(max_tokens):
        try:
            # Prepare decoder inputs with attention mask (must be int64)
            decoder_attention_mask = np.ones_like(decoder_input_ids, dtype=np.int64)
            
            decoder_inputs = {
                "decoder_input_ids": decoder_input_ids,
                "decoder_attention_mask": decoder_attention_mask,
                "encoder_hidden_states": encoder_hidden,
                "encoder_attention_mask": encoder_attention_mask.astype(np.int64),
            }
            
            # Run decoder
            decoder_outputs = decoder_session.run(None, decoder_inputs)
            dec_hidden = decoder_outputs[0].astype(np.float32)  # (batch, seq_len, hidden_size)
            
        except Exception as e:
            print(f"[ONNX] Decoder error at step {step}: {e}")
            break
        
        # Get last hidden state
        last_hidden = dec_hidden[:, -1:, :]  # (batch, 1, hidden_size)
        
        # Run LM head
        try:
            lm_head_inputs = {"hidden_states": last_hidden}
            lm_head_outputs = lm_head_session.run(None, lm_head_inputs)
            logits = lm_head_outputs[0]  # (batch, 1, vocab_size)
        except Exception as e:
            print(f"[ONNX] LM head error at step {step}: {e}")
            break
        
        # Force first token to be language ID
        if decoder_input_ids.shape[1] == 1:
            next_token = np.array([[forced_bos]], dtype=np.int64)
            print(f"[ONNX] Forced first token: {forced_bos} (language ID)")
        else:
            # Greedy: pick max logit
            next_token = np.argmax(logits, axis=-1).astype(np.int64)
        
        # Concatenate token to sequence
        decoder_input_ids = np.concatenate([decoder_input_ids, next_token], axis=1)
        
        # Stop if EOS
        if next_token.item() == tok.eos_token_id:
            print(f"[ONNX] Generated {decoder_input_ids.shape[1]} tokens (EOS reached)")
            break
    
    # Decode entire sequence
    output_text = tok.batch_decode(decoder_input_ids, skip_special_tokens=True)[0]
    print(f"[ONNX] Output: {output_text[:100]}..." if len(output_text) > 100 else f"[ONNX] Output: {output_text}\n")
    
    return output_text


def translate_onnx(text: str, src_lang: str, tgt_lang: str, max_tokens: int = 256) -> str:
    """
    Translate text using ONNX M2M-100 models.
    
    Args:
        text: Input text
        src_lang: Source language code
        tgt_lang: Target language code
        max_tokens: Max tokens to generate
    
    Returns:
        Translated text
    """
    if not ensure_onnx_models():
        raise RuntimeError("ONNX models not found. Ensure models are copied to models/translators/m2m100_onnx")
    
    if not TOKENIZER_AVAILABLE:
        raise RuntimeError("transformers library required. Install with: pip install transformers")
    
    try:
        encoder_outputs = encode_with_onnx(text, src_lang)
        translated = decode_with_onnx(encoder_outputs, tgt_lang, max_tokens)
        return translated
    except Exception as e:
        print(f"[ONNX] Translation failed: {e}")
        raise


def unload_onnx_translator():
    """Clear cached ONNX sessions to free memory."""
    global onnx_translator_cache, tokenizer
    onnx_translator_cache.clear()
    tokenizer = None
    print("[ONNX] Sessions unloaded.")


def preload_onnx_translator():
    """Preload ONNX tokenizer and sessions without running a translation."""
    if not ensure_onnx_models():
        raise RuntimeError("ONNX models not available")

    load_onnx_tokenizer()

    models_dir = get_onnx_models_dir()
    encoder_path = models_dir / "encoder" / ONNX_ENCODER_MODEL
    decoder_path = models_dir / "decoder" / ONNX_DECODER_MODEL
    lm_head_path = models_dir / "lm_head" / ONNX_LM_HEAD_MODEL

    get_onnx_session(str(encoder_path))
    get_onnx_session(str(decoder_path))
    get_onnx_session(str(lm_head_path))

    return {
        "tokenizer": True,
        "models": {
            "encoder": encoder_path.name,
            "decoder": decoder_path.name,
            "lm_head": lm_head_path.name,
        },
    }


def get_onnx_status() -> dict:
    """Get status of ONNX translator."""
    models_dir = get_onnx_models_dir()
    return {
        "available": ensure_onnx_models(),
        "models_dir": str(models_dir),
        "models": {
            "encoder": (models_dir / "encoder" / ONNX_ENCODER_MODEL).exists(),
            "decoder": (models_dir / "decoder" / ONNX_DECODER_MODEL).exists(),
            "lm_head": (models_dir / "lm_head" / ONNX_LM_HEAD_MODEL).exists(),
        },
        "active_models": {
            "encoder": ONNX_ENCODER_MODEL,
            "decoder": ONNX_DECODER_MODEL,
            "lm_head": ONNX_LM_HEAD_MODEL,
        },
        "tokenizer_available": TOKENIZER_AVAILABLE,
    }
