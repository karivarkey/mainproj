# app/services/translator_service.py
import sys
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor

sys.excepthook = lambda exc_type, exc, tb: (
    print("UNCAUGHT ERROR:", exc_type.__name__, exc),
    __import__("traceback").print_tb(tb)
)

# ---------------------------------------------------------------------
# MODELS FOR EACH DIRECTION
# ---------------------------------------------------------------------
MODEL_EN_INDIC = "ai4bharat/indictrans2-en-indic-dist-200M"
MODEL_INDIC_EN = "ai4bharat/indictrans2-indic-en-dist-200M"
MODEL_INDIC_INDIC = "ai4bharat/indictrans2-indic-indic-dist-320M"

translator_cache = {}

# ---------------------------------------------------------------------
# MODEL PICKER
# ---------------------------------------------------------------------
def pick_model(src_lang: str, tgt_lang: str) -> str:
    if src_lang == "eng_Latn":
        return MODEL_EN_INDIC          # English → Indic

    if tgt_lang == "eng_Latn":
        return MODEL_INDIC_EN          # Indic → English

    return MODEL_INDIC_INDIC           # Indic → Indic


# ---------------------------------------------------------------------
# MODEL LOADER (CPU SAFE)
# ---------------------------------------------------------------------
def get_translator(model_id: str):
    if model_id in translator_cache:
        return translator_cache[model_id]

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map=None
    ).to("cpu")

    model.eval()

    try:
        proc = IndicProcessor(inference=True)
    except:
        proc = IndicProcessor()

    translator_cache[model_id] = (tok, model, proc)
    return tok, model, proc


# ---------------------------------------------------------------------
# TRANSLATE
# ---------------------------------------------------------------------
def translate(text: str, src_lang: str, tgt_lang: str, max_tokens: int = 256):
    print(f"\n[TRANSLATE] {src_lang} → {tgt_lang}")
    print(f"Input: {text}")

    model_id = pick_model(src_lang, tgt_lang)
    tok, model, proc = get_translator(model_id)

    processed = proc.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tok(
        processed,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    ).to("cpu")

    gen_kwargs = dict(
        use_cache=False,
        num_beams=1,
        num_return_sequences=1,
        max_new_tokens=max_tokens
    )

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    decoded = tok.batch_decode(
        out,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    final = proc.postprocess_batch(decoded, lang=tgt_lang)
    output = final[0].strip()

    print(f"Output: {output}\n")
    return output
