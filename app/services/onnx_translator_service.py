"""ONNX Runtime translator service with runtime-selectable ONNX family (m2m/nllb)."""

import onnxruntime as ort
import numpy as np
from pathlib import Path
from app.config import (
    ONNX_MODEL_FAMILY,
    ONNX_FAMILY_CONFIG,
    ONNX_M2M_LANG_MAP,
    ONNX_NLLB_LANG_MAP,
    ONNX_INTRA_OP_THREADS,
    ONNX_INTER_OP_THREADS,
    ONNX_ENABLE_ALL_OPTIMIZATIONS,
    ONNX_VERBOSE_LOGS,
    ONNX_TRANSLATION_MAX_SOURCE_TOKENS,
    ONNX_TRANSLATION_LENGTH_RATIO,
    ONNX_TRANSLATION_MIN_STEPS,
)

# Global translator cache
onnx_translator_cache = {}
onnx_tokenizer_cache = {}

# Try to import tokenizer - we'll use transformers' tokenizer for now
try:
    from transformers import AutoTokenizer, M2M100Tokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False

device = "cpu"  # ONNX Runtime with CPU for now


def _log(message: str):
    if ONNX_VERBOSE_LOGS:
        print(message)


def _normalize_family(onnx_family: str | None = None) -> str:
    selected = (onnx_family or ONNX_MODEL_FAMILY or "m2m").strip().lower()
    return selected if selected in ONNX_FAMILY_CONFIG else "m2m"


def _family_cfg(onnx_family: str | None = None) -> dict:
    family = _normalize_family(onnx_family)
    return {
        "family": family,
        **ONNX_FAMILY_CONFIG[family],
    }


def _lang_map_for_family(onnx_family: str | None = None) -> dict:
    family = _normalize_family(onnx_family)
    return ONNX_M2M_LANG_MAP if family == "m2m" else ONNX_NLLB_LANG_MAP


def get_onnx_models_dir(onnx_family: str | None = None) -> Path:
    """Get ONNX models directory for selected family."""
    return _family_cfg(onnx_family)["models_dir"]


def _active_model_names(onnx_family: str | None = None) -> dict:
    cfg = _family_cfg(onnx_family)
    return {
        "encoder": cfg["encoder_model"],
        "decoder": cfg["decoder_model"],
        "lm_head": cfg["lm_head_model"],
    }


def ensure_onnx_models(onnx_family: str | None = None) -> bool:
    """Check if ONNX model files exist (checks for configured model variants)."""
    models_dir = get_onnx_models_dir(onnx_family)
    names = _active_model_names(onnx_family)
    required = [
        models_dir / "encoder" / names["encoder"],
        models_dir / "decoder" / names["decoder"],
        models_dir / "lm_head" / names["lm_head"],
    ]

    for file_path in required:
        if not file_path.exists() or file_path.stat().st_size < 1024:
            return False
        try:
            with file_path.open("rb") as handle:
                prefix = handle.read(256).lower()
                if b"<!doctype html" in prefix or b"<html" in prefix:
                    return False
        except Exception:
            return False

    return True


def _inspect_onnx_asset(file_path: Path) -> dict:
    if not file_path.exists():
        return {
            "exists": False,
            "size_bytes": 0,
            "is_html": False,
            "valid": False,
            "reason": "missing",
        }

    try:
        size_bytes = int(file_path.stat().st_size)
    except Exception:
        return {
            "exists": True,
            "size_bytes": 0,
            "is_html": False,
            "valid": False,
            "reason": "unreadable",
        }

    if size_bytes < 1024:
        return {
            "exists": True,
            "size_bytes": size_bytes,
            "is_html": False,
            "valid": False,
            "reason": "too_small",
        }

    try:
        with file_path.open("rb") as handle:
            prefix = handle.read(256).lower()
        is_html = (b"<!doctype html" in prefix) or (b"<html" in prefix)
    except Exception:
        return {
            "exists": True,
            "size_bytes": size_bytes,
            "is_html": False,
            "valid": False,
            "reason": "unreadable",
        }

    if is_html:
        return {
            "exists": True,
            "size_bytes": size_bytes,
            "is_html": True,
            "valid": False,
            "reason": "html_payload",
        }

    return {
        "exists": True,
        "size_bytes": size_bytes,
        "is_html": False,
        "valid": True,
        "reason": "ok",
    }


def load_onnx_tokenizer(onnx_family: str | None = None):
    """Load ONNX translator tokenizer from local directory (offline mode)."""
    family = _normalize_family(onnx_family)
    if family in onnx_tokenizer_cache:
        return onnx_tokenizer_cache[family]
    
    if not TOKENIZER_AVAILABLE:
        raise RuntimeError("transformers library required for tokenization")
    
    try:
        cfg = _family_cfg(family)
        tokenizer_dir: Path = cfg["tokenizer_dir"]
        tokenizer_path = str(tokenizer_dir)
        print(f"[ONNX:{family}] Loading tokenizer from local directory...")
        try:
            from app.services.onnx_model_download_service import is_onnx_tokenizer_ready
        except Exception as prep_import_error:
            raise RuntimeError(f"Tokenizer helper unavailable: {prep_import_error}") from prep_import_error

        if (not tokenizer_dir.exists()) or (not is_onnx_tokenizer_ready(family=family)):
            raise RuntimeError(
                f"Tokenizer files missing/incomplete at: {tokenizer_path}. "
                "Download tokenizer explicitly via /onnx_tokenizer/ensure"
            )

        try:
            tokenizer_value = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        except Exception:
            try:
                tokenizer_value = M2M100Tokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            except Exception:
                raise RuntimeError(
                    f"Failed to load tokenizer from local path: {tokenizer_path}. "
                    "Re-download tokenizer explicitly via /onnx_tokenizer/ensure"
                )
        onnx_tokenizer_cache[family] = tokenizer_value
        print(f"[ONNX:{family}] Tokenizer loaded (offline mode).")
        return tokenizer_value
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
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = ONNX_INTRA_OP_THREADS
        session_options.inter_op_num_threads = ONNX_INTER_OP_THREADS
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if ONNX_ENABLE_ALL_OPTIMIZATIONS
            else ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
        session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)
        onnx_translator_cache[cache_key] = session
        _log(
            f"[ONNX] Loaded session for {Path(model_path).name} "
            f"(intra={ONNX_INTRA_OP_THREADS}, inter={ONNX_INTER_OP_THREADS})"
        )
        return session
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model {model_path}: {e}") from e


def _numpy_dtype_for_onnx_type(onnx_type: str):
    t = (onnx_type or "").lower()
    if "float16" in t:
        return np.float16
    if "float" in t:
        return np.float32
    if "int32" in t:
        return np.int32
    if "int16" in t:
        return np.int16
    if "int8" in t:
        return np.int8
    if "uint8" in t:
        return np.uint8
    if "bool" in t:
        return np.bool_
    return np.int64


def _cast_inputs_for_session(session: ort.InferenceSession, inputs: dict) -> dict:
    casted = {}
    input_meta = {meta.name: meta for meta in session.get_inputs()}
    for name, value in inputs.items():
        if name not in input_meta:
            continue
        expected_dtype = _numpy_dtype_for_onnx_type(input_meta[name].type)
        arr = value
        if isinstance(arr, np.ndarray):
            if arr.dtype != expected_dtype:
                arr = arr.astype(expected_dtype)
        else:
            arr = np.asarray(arr, dtype=expected_dtype)
        casted[name] = arr
    return casted


def _punctuation_token_ids(tok) -> set[int]:
    ids = set()
    for symbol in (".", "!", "?", "।"):
        try:
            for token_id in tok.encode(symbol, add_special_tokens=False):
                ids.add(int(token_id))
        except Exception:
            pass
    return ids


def encode_with_onnx(text: str, src_lang: str, onnx_family: str | None = None) -> dict:
    """
    Encode input text using ONNX encoder.
    
    Args:
        text: Input text
        src_lang: Source language code (e.g., "hi", "en")
    
    Returns:
        Dict with encoder outputs
    """
    family = _normalize_family(onnx_family)
    tok = load_onnx_tokenizer(family)
    
    # Map short codes to M2M-100 codes
    src_code = _lang_map_for_family(family).get(src_lang, src_lang)
    
    _log(f"[ONNX:{family}] Encoding {src_lang} ({src_code}): {text[:50]}...")
    
    # Tokenize with forced language token
    tok.src_lang = src_code
    inputs = tok(
        text,
        return_tensors="pt",
        max_length=ONNX_TRANSLATION_MAX_SOURCE_TOKENS,
        truncation=True,
        padding=True,
    )
    
    # Run encoder
    models_dir = get_onnx_models_dir(family)
    model_names = _active_model_names(family)
    encoder_path = models_dir / "encoder" / model_names["encoder"]
    encoder_session = get_onnx_session(str(encoder_path))
    
    # Prepare inputs as numpy arrays
    input_ids = inputs["input_ids"].numpy().astype(np.int64)
    attention_mask = inputs["attention_mask"].numpy().astype(np.int64)
    
    encoder_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    encoder_inputs = _cast_inputs_for_session(encoder_session, encoder_inputs)
    encoder_outputs = encoder_session.run(None, encoder_inputs)
    
    return {
        "family": family,
        "last_hidden_state": encoder_outputs[0],  # (batch, seq_len, hidden_size)
        "attention_mask": attention_mask,
        "source_token_count": int(np.sum(attention_mask)),
        "tokenizer": tok,
    }


def decode_with_onnx(encoder_outputs: dict, tgt_lang: str, max_tokens: int = 256, onnx_family: str | None = None) -> str:
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
    family = _normalize_family(onnx_family or encoder_outputs.get("family"))
    encoder_hidden = encoder_outputs["last_hidden_state"].astype(np.float32)
    encoder_attention_mask = encoder_outputs["attention_mask"]
    source_token_count = int(encoder_outputs.get("source_token_count", 0))
    
    tgt_code = _lang_map_for_family(family).get(tgt_lang, tgt_lang)
    _log(f"[ONNX:{family}] Decoding to {tgt_lang} ({tgt_code})...")
    
    # Get forced BOS token (language ID)
    try:
        if hasattr(tok, "get_lang_id"):
            forced_bos = tok.get_lang_id(tgt_code)
        elif hasattr(tok, "lang_code_to_id") and tgt_code in tok.lang_code_to_id:
            forced_bos = int(tok.lang_code_to_id[tgt_code])
        else:
            forced_bos = int(tok.convert_tokens_to_ids(tgt_code))
            if forced_bos < 0:
                raise ValueError(f"Unknown target language token: {tgt_code}")
        _log(f"[ONNX:{family}] Forced BOS token ID: {forced_bos}")
    except Exception as e:
        _log(f"[ONNX:{family}] Warning: Could not get language ID for {tgt_code}: {e}")
        forced_bos = tok.eos_token_id
    
    # Start directly with target language token to avoid a wasted decode step.
    decoder_input_ids = np.array([[forced_bos]], dtype=np.int64)
    
    models_dir = get_onnx_models_dir(family)
    model_names = _active_model_names(family)
    decoder_path = models_dir / "decoder" / model_names["decoder"]
    lm_head_path = models_dir / "lm_head" / model_names["lm_head"]
    
    decoder_session = get_onnx_session(str(decoder_path))
    lm_head_session = get_onnx_session(str(lm_head_path))
    
    effective_max_tokens = min(
        max_tokens,
        max(ONNX_TRANSLATION_MIN_STEPS + 2, int(source_token_count * ONNX_TRANSLATION_LENGTH_RATIO) + 6),
    )
    sentence_end_ids = _punctuation_token_ids(tok)

    # Greedy decoding loop
    for step in range(effective_max_tokens):
        try:
            # Prepare decoder inputs with attention mask (must be int64)
            decoder_attention_mask = np.ones_like(decoder_input_ids, dtype=np.int64)
            
            decoder_inputs = {
                "decoder_input_ids": decoder_input_ids,
                "decoder_attention_mask": decoder_attention_mask,
                "encoder_hidden_states": encoder_hidden,
                "encoder_attention_mask": encoder_attention_mask,
            }

            decoder_inputs = _cast_inputs_for_session(decoder_session, decoder_inputs)
            
            # Run decoder
            decoder_outputs = decoder_session.run(None, decoder_inputs)
            dec_hidden = decoder_outputs[0].astype(np.float32)  # (batch, seq_len, hidden_size)
            
        except Exception as e:
            _log(f"[ONNX:{family}] Decoder error at step {step}: {e}")
            break
        
        # Get last hidden state
        last_hidden = dec_hidden[:, -1:, :]  # (batch, 1, hidden_size)
        
        # Run LM head
        try:
            lm_head_inputs = _cast_inputs_for_session(lm_head_session, {"hidden_states": last_hidden})
            lm_head_outputs = lm_head_session.run(None, lm_head_inputs)
            logits = lm_head_outputs[0]  # (batch, 1, vocab_size)
        except Exception as e:
            _log(f"[ONNX:{family}] LM head error at step {step}: {e}")
            break
        
        # Greedy: pick max logit
        next_token = np.argmax(logits, axis=-1).astype(np.int64)
        
        # Concatenate token to sequence
        decoder_input_ids = np.concatenate([decoder_input_ids, next_token], axis=1)
        
        # Stop if EOS
        next_id = int(next_token.item())
        if next_id == tok.eos_token_id:
            _log(f"[ONNX:{family}] Generated {decoder_input_ids.shape[1]} tokens (EOS reached)")
            break
        if step >= ONNX_TRANSLATION_MIN_STEPS and next_id in sentence_end_ids:
            _log(f"[ONNX:{family}] Generated {decoder_input_ids.shape[1]} tokens (sentence-end reached)")
            break
    
    # Decode entire sequence
    output_text = tok.batch_decode(decoder_input_ids, skip_special_tokens=True)[0]
    _log(
        f"[ONNX:{family}] Output: {output_text[:100]}..."
        if len(output_text) > 100
        else f"[ONNX:{family}] Output: {output_text}\n"
    )
    
    return output_text


def translate_onnx(text: str, src_lang: str, tgt_lang: str, max_tokens: int = 256, onnx_family: str | None = None) -> str:
    """
    Translate text using ONNX models (M2M or NLLB family).
    
    Args:
        text: Input text
        src_lang: Source language code
        tgt_lang: Target language code
        max_tokens: Max tokens to generate
    
    Returns:
        Translated text
    """
    family = _normalize_family(onnx_family)
    if not ensure_onnx_models(family):
        raise RuntimeError(
            f"ONNX models not found for family '{family}'. Download models explicitly via /onnx_models/download"
        )
    
    if not TOKENIZER_AVAILABLE:
        raise RuntimeError("transformers library required. Install with: pip install transformers")
    
    try:
        encoder_outputs = encode_with_onnx(text, src_lang, family)
        translated = decode_with_onnx(encoder_outputs, tgt_lang, max_tokens, family)
        return translated
    except Exception as e:
        _log(f"[ONNX:{family}] Translation failed: {e}")
        raise


def unload_onnx_translator(onnx_family: str | None = None):
    """Clear cached ONNX sessions to free memory."""
    family = _normalize_family(onnx_family)
    global onnx_translator_cache, onnx_tokenizer_cache
    onnx_translator_cache.clear()
    if onnx_family is None:
        onnx_tokenizer_cache.clear()
    else:
        onnx_tokenizer_cache.pop(family, None)
    _log(f"[ONNX:{family}] Sessions unloaded.")


def preload_onnx_translator(onnx_family: str | None = None):
    """Preload ONNX tokenizer and sessions without running a translation."""
    family = _normalize_family(onnx_family)
    if not ensure_onnx_models(family):
        raise RuntimeError(
            f"ONNX models not available for family '{family}'. Download models explicitly via /onnx_models/download"
        )

    load_onnx_tokenizer(family)

    model_names = _active_model_names(family)
    models_dir = get_onnx_models_dir(family)
    encoder_path = models_dir / "encoder" / model_names["encoder"]
    decoder_path = models_dir / "decoder" / model_names["decoder"]
    lm_head_path = models_dir / "lm_head" / model_names["lm_head"]

    get_onnx_session(str(encoder_path))
    get_onnx_session(str(decoder_path))
    get_onnx_session(str(lm_head_path))

    return {
        "family": family,
        "tokenizer": True,
        "auto_download": None,
        "models": {
            "encoder": encoder_path.name,
            "decoder": decoder_path.name,
            "lm_head": lm_head_path.name,
        },
    }


def is_onnx_family_loaded(onnx_family: str | None = None) -> bool:
    family = _normalize_family(onnx_family)
    if family not in onnx_tokenizer_cache:
        return False

    model_names = _active_model_names(family)
    models_dir = get_onnx_models_dir(family)
    required_keys = {
        str(models_dir / "encoder" / model_names["encoder"]),
        str(models_dir / "decoder" / model_names["decoder"]),
        str(models_dir / "lm_head" / model_names["lm_head"]),
    }
    return required_keys.issubset(set(onnx_translator_cache.keys()))


def get_onnx_status(onnx_family: str | None = None) -> dict:
    """Get status of ONNX translator."""
    family = _normalize_family(onnx_family)
    model_names = _active_model_names(family)
    cfg = _family_cfg(family)
    models_dir = cfg["models_dir"]
    model_ready = ensure_onnx_models(family)

    tokenizer_ready = False
    tokenizer_prepare_error = None
    try:
        from app.services.onnx_model_download_service import is_onnx_tokenizer_ready
        tokenizer_ready = is_onnx_tokenizer_ready(family=family)
    except Exception:
        tokenizer_ready = cfg["tokenizer_dir"].exists()

    encoder_path = models_dir / "encoder" / model_names["encoder"]
    decoder_path = models_dir / "decoder" / model_names["decoder"]
    lm_head_path = models_dir / "lm_head" / model_names["lm_head"]

    asset_checks = {
        "encoder": _inspect_onnx_asset(encoder_path),
        "decoder": _inspect_onnx_asset(decoder_path),
        "lm_head": _inspect_onnx_asset(lm_head_path),
    }

    issues = []
    for key, check in asset_checks.items():
        if not check.get("valid", False):
            issues.append(f"{key}:{check.get('reason', 'invalid')}")

    if not tokenizer_ready:
        issues.append("tokenizer:missing_or_incomplete")
        if tokenizer_prepare_error:
            issues.append("tokenizer:auto_prepare_failed")

    return {
        "family": family,
        "available": model_ready and tokenizer_ready,
        "models_dir": str(models_dir),
        "models": {
            "encoder": encoder_path.exists(),
            "decoder": decoder_path.exists(),
            "lm_head": lm_head_path.exists(),
        },
        "active_models": {
            "encoder": model_names["encoder"],
            "decoder": model_names["decoder"],
            "lm_head": model_names["lm_head"],
        },
        "asset_checks": asset_checks,
        "issues": issues,
        "model_auto_prepare": {
            "attempted": False,
            "ok": model_ready,
            "error": None,
            "requested_files": [
                model_names["encoder"],
                model_names["decoder"],
                model_names["lm_head"],
                f"{model_names['lm_head']}.data",
            ],
            "downloaded_files": [],
        },
        "tokenizer_available": TOKENIZER_AVAILABLE,
        "tokenizer_ready": tokenizer_ready,
        "tokenizer_dir": str(cfg["tokenizer_dir"]),
    }
