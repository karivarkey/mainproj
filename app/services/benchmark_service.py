import time
import psutil
import json
from app.services.translator_service import translate
from app.services.llm_service import llm_generate
from app.services.rag_service import rag_retrieve

def measure_time(fn, *args, **kwargs):
    """Utility to time any function call."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start

def memory_snapshot():
    """Capture current process memory usage."""
    p = psutil.Process()
    mem = p.memory_info()
    return {
        "rss_mb": mem.rss / (1024 * 1024),
        "vms_mb": mem.vms / (1024 * 1024),
    }

def benchmark_pipeline(test_text, src_lang, tgt_lang, max_tokens=64):
    bleu = BLEU()
    chrf = CHRF()

    # 1) Input translation latency
    t0 = time.time()
    en_text = translate(test_text, src_lang, "en")
    t1 = time.time()
    input_translation_latency = (t1 - t0) * 1000

    # 2) LLM English-only reasoning latency (baseline)
    prompt = f"Answer in one short sentence: {en_text}"
    t2 = time.time()
    llm_out_en = llm_generate(prompt, max_new_tokens=max_tokens)
    t3 = time.time()
    llm_reasoning_latency = (t3 - t2) * 1000

    # 3) Output translation latency
    t4 = time.time()
    translated_back = translate(llm_out_en, "en", src_lang)  # return to user’s language
    t5 = time.time()
    output_translation_latency = (t5 - t4) * 1000

    # 4) Total pipeline latency
    total_pipeline_latency = (t5 - t0) * 1000

    # 5) VRAM usage
    vram = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
    vram_free, vram_total = vram
    vram_used = vram_total - vram_free

    # 6) Translation quality
    round_trip_bleu = bleu.sentence_score(translated_back, [test_text]).score
    round_trip_chrf = chrf.sentence_score(translated_back, [test_text]).score

    # 7) Multilingual preservation (simple automatic check)
    meaning_preserved = round_trip_bleu > 25 or round_trip_chrf > 40

    # 8) Supported languages (very light check)
    supported_languages = 0
    errors = 0
    quick_langs = ["en", "hi", "ml", "ta", "es", "fr"]
    for lang in quick_langs:
        try:
            translate(test_text, src_lang, lang)
            supported_languages += 1
        except Exception:
            errors += 1

    # 9) Model compatibility success rate (mocking hooks)
    model_compatibility = {
        "llama": True,
        "qwen": True,
        "phi": True,
        "mistral": True,
        "gemma": True,
    }

    # 10) Throughput estimate
    request_time = total_pipeline_latency / 1000
    rps = 1 / request_time if request_time > 0 else 0

    return {
        "texts": {
            "input": test_text,
            "english": en_text,
            "llm_output_en": llm_out_en,
            "round_trip": translated_back
        },
        "latency_ms": {
            "input_translation": input_translation_latency,
            "llm_reasoning": llm_reasoning_latency,
            "output_translation": output_translation_latency,
            "total_pipeline": total_pipeline_latency,
        },
        "vram": {
            "used_mb": vram_used / (1024 * 1024),
            "total_mb": vram_total / (1024 * 1024) if torch.cuda.is_available() else 0
        },
        "quality": {
            "bleu": round_trip_bleu,
            "chrf": round_trip_chrf,
            "meaning_preserved": meaning_preserved,
        },
        "scalability": {
            "supported_language_count": supported_languages,
            "translation_errors": errors,
            "model_compatibility": model_compatibility
        },
        "throughput_rps": rps
    }