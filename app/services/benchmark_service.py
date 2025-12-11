import time
import psutil
import json
import gc
import torch
from sacrebleu.metrics import BLEU, CHRF
from app.services.translator_service import translate, unload_translator
from app.services.llm_service import llm_generate, load_llm, unload_llm
from app.services.rag_service import rag_add, rag_clear

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
    
    # CPU usage (percentage over 0.1s interval)
    cpu_percent = p.cpu_percent(interval=0.1)
    
    # VRAM usage (CUDA)
    vram_used_mb = 0
    vram_total_mb = 0
    vram_free_mb = 0
    if torch.cuda.is_available():
        vram_free, vram_total = torch.cuda.mem_get_info()
        vram_used = vram_total - vram_free
        vram_used_mb = vram_used / (1024 * 1024)
        vram_total_mb = vram_total / (1024 * 1024)
        vram_free_mb = vram_free / (1024 * 1024)
    
    return {
        "rss_mb": mem.rss / (1024 * 1024),
        "vms_mb": mem.vms / (1024 * 1024),
        "cpu_percent": cpu_percent,
        "vram_used_mb": vram_used_mb,
        "vram_total_mb": vram_total_mb,
        "vram_free_mb": vram_free_mb,
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


def benchmark_resource_usage(
    llm_name: str,
    prompts: list[dict],
    rag_data: list[str] = None,
    n_ctx: int = 4096,
    n_gpu_layers: int = -1,
    max_tokens: int = 128
):
    """
    Comprehensive resource usage benchmark for LLM + Translator + RAG pipeline.
    
    Steps:
    1. Unload all models and measure baseline RAM/VRAM
    2. Load LLM and measure
    3. Load translator and measure
    4. Add RAG data and measure
    5. Run full translate→infer→translate pipeline for each prompt, measuring time, RAM, CPU, VRAM
    
    Args:
        llm_name: Name of the LLM model to load
        prompts: List of prompt objects with {"text": str, "lang": str}
        rag_data: Optional list of documents to add to RAG
        n_ctx: Context size for LLM
        n_gpu_layers: GPU layers for LLM (-1 = all)
        max_tokens: Max tokens per generation
    
    Returns:
        Detailed metrics dict with baseline, load, and per-prompt inference stats
    """
    results = {
        "config": {
            "llm_name": llm_name,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "max_tokens": max_tokens,
            "num_prompts": len(prompts),
            "num_rag_docs": len(rag_data) if rag_data else 0,
        },
        "stages": {}
    }
    
    # Stage 0: Unload everything and measure baseline
    print("[BENCHMARK] Stage 0: Unloading all models...")
    unload_llm()
    unload_translator()
    rag_clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(1)  # Let system stabilize
    
    baseline = memory_snapshot()
    results["stages"]["0_baseline"] = {
        "description": "Baseline (all models unloaded)",
        "metrics": baseline
    }
    print(f"[BENCHMARK] Baseline: RAM={baseline['rss_mb']:.1f}MB, VRAM={baseline['vram_used_mb']:.1f}MB")
    
    # Stage 1: Load LLM
    print(f"[BENCHMARK] Stage 1: Loading LLM '{llm_name}'...")
    t_start = time.perf_counter()
    try:
        load_llm(llm_name, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
        t_load_llm = time.perf_counter() - t_start
        after_llm = memory_snapshot()
        results["stages"]["1_llm_loaded"] = {
            "description": f"LLM loaded: {llm_name}",
            "load_time_s": t_load_llm,
            "metrics": after_llm,
            "delta_from_baseline": {
                "rss_mb": after_llm["rss_mb"] - baseline["rss_mb"],
                "vram_mb": after_llm["vram_used_mb"] - baseline["vram_used_mb"],
            }
        }
        print(f"[BENCHMARK] LLM loaded in {t_load_llm:.2f}s, RAM delta={after_llm['rss_mb'] - baseline['rss_mb']:.1f}MB, VRAM delta={after_llm['vram_used_mb'] - baseline['vram_used_mb']:.1f}MB")
    except Exception as e:
        results["stages"]["1_llm_loaded"] = {
            "description": f"LLM load failed: {llm_name}",
            "error": str(e)
        }
        print(f"[BENCHMARK] LLM load failed: {e}")
        return results
    
    # Stage 2: Load Translator (trigger by calling translate once)
    print("[BENCHMARK] Stage 2: Loading Translator...")
    t_start = time.perf_counter()
    try:
        # Trigger translator load with a dummy translation
        _ = translate("test", "en", "hi", max_tokens=10)
        t_load_translator = time.perf_counter() - t_start
        after_translator = memory_snapshot()
        results["stages"]["2_translator_loaded"] = {
            "description": "Translator loaded (NLLB)",
            "load_time_s": t_load_translator,
            "metrics": after_translator,
            "delta_from_llm": {
                "rss_mb": after_translator["rss_mb"] - after_llm["rss_mb"],
                "vram_mb": after_translator["vram_used_mb"] - after_llm["vram_used_mb"],
            }
        }
        print(f"[BENCHMARK] Translator loaded in {t_load_translator:.2f}s, RAM delta={after_translator['rss_mb'] - after_llm['rss_mb']:.1f}MB, VRAM delta={after_translator['vram_used_mb'] - after_llm['vram_used_mb']:.1f}MB")
    except Exception as e:
        results["stages"]["2_translator_loaded"] = {
            "description": "Translator load failed",
            "error": str(e)
        }
        print(f"[BENCHMARK] Translator load failed: {e}")
        return results
    
    # Stage 3: Add RAG data
    if rag_data:
        print(f"[BENCHMARK] Stage 3: Adding {len(rag_data)} documents to RAG...")
        t_start = time.perf_counter()
        try:
            for doc in rag_data:
                rag_add(doc)
            t_load_rag = time.perf_counter() - t_start
            after_rag = memory_snapshot()
            results["stages"]["3_rag_loaded"] = {
                "description": f"RAG data added ({len(rag_data)} docs)",
                "load_time_s": t_load_rag,
                "metrics": after_rag,
                "delta_from_translator": {
                    "rss_mb": after_rag["rss_mb"] - after_translator["rss_mb"],
                    "vram_mb": after_rag["vram_used_mb"] - after_translator["vram_used_mb"],
                }
            }
            print(f"[BENCHMARK] RAG loaded in {t_load_rag:.2f}s, RAM delta={after_rag['rss_mb'] - after_translator['rss_mb']:.1f}MB")
        except Exception as e:
            results["stages"]["3_rag_loaded"] = {
                "description": "RAG load failed",
                "error": str(e)
            }
            print(f"[BENCHMARK] RAG load failed: {e}")
            return results
    else:
        after_rag = after_translator
        results["stages"]["3_rag_loaded"] = {
            "description": "No RAG data provided",
            "metrics": after_rag
        }
    
    # Stage 4: Run full translate→infer→translate pipeline for each prompt
    print(f"[BENCHMARK] Stage 4: Running full pipeline for {len(prompts)} prompts...")
    results["inference_runs"] = []
    
    for idx, prompt_obj in enumerate(prompts):
        text = prompt_obj.get("text", "")
        lang = prompt_obj.get("lang", "en")
        
        print(f"[BENCHMARK] Prompt {idx + 1}/{len(prompts)} [{lang}]: {text[:50]}...")
        
        before_run = memory_snapshot()
        t_start = time.perf_counter()
        
        try:
            # Step 1: Translate to English
            t_translate_in_start = time.perf_counter()
            english_text = translate(text, lang, "en", max_tokens=256)
            t_translate_in = time.perf_counter() - t_translate_in_start
            after_translate_in = memory_snapshot()
            
            # Step 2: LLM inference in English
            t_infer_start = time.perf_counter()
            llm_output_en = llm_generate(english_text, max_new_tokens=max_tokens)
            t_infer = time.perf_counter() - t_infer_start
            after_infer = memory_snapshot()
            
            # Step 3: Translate back to source language
            t_translate_out_start = time.perf_counter()
            final_output = translate(llm_output_en, "en", lang, max_tokens=256)
            t_translate_out = time.perf_counter() - t_translate_out_start
            after_translate_out = memory_snapshot()
            
            t_total = time.perf_counter() - t_start
            
            run_result = {
                "prompt_idx": idx,
                "input_text": text,
                "input_lang": lang,
                "english_text": english_text,
                "llm_output_en": llm_output_en,
                "final_output": final_output,
                "timing": {
                    "translate_to_en_s": t_translate_in,
                    "llm_inference_s": t_infer,
                    "translate_to_source_s": t_translate_out,
                    "total_s": t_total
                },
                "metrics_before": before_run,
                "metrics_after_translate_in": after_translate_in,
                "metrics_after_infer": after_infer,
                "metrics_after_translate_out": after_translate_out,
                "delta": {
                    "rss_mb": after_translate_out["rss_mb"] - before_run["rss_mb"],
                    "vram_mb": after_translate_out["vram_used_mb"] - before_run["vram_used_mb"],
                    "cpu_percent": after_translate_out["cpu_percent"],
                }
            }
            results["inference_runs"].append(run_result)
            print(f"[BENCHMARK]   Total: {t_total:.2f}s (translate_in: {t_translate_in:.2f}s, infer: {t_infer:.2f}s, translate_out: {t_translate_out:.2f}s)")
            print(f"[BENCHMARK]   CPU: {after_translate_out['cpu_percent']:.1f}%, RAM: {after_translate_out['rss_mb']:.1f}MB, VRAM: {after_translate_out['vram_used_mb']:.1f}MB")
            
        except Exception as e:
            run_result = {
                "prompt_idx": idx,
                "input_text": text,
                "input_lang": lang,
                "error": str(e),
                "time_s": time.perf_counter() - t_start,
            }
            results["inference_runs"].append(run_result)
            print(f"[BENCHMARK]   Error: {e}")
    
    # Summary statistics
    successful_runs = [r for r in results["inference_runs"] if "error" not in r]
    if successful_runs:
        times = [r["timing"]["total_s"] for r in successful_runs]
        translate_in_times = [r["timing"]["translate_to_en_s"] for r in successful_runs]
        infer_times = [r["timing"]["llm_inference_s"] for r in successful_runs]
        translate_out_times = [r["timing"]["translate_to_source_s"] for r in successful_runs]
        
        results["summary"] = {
            "total_prompts": len(prompts),
            "successful_runs": len(successful_runs),
            "failed_runs": len(prompts) - len(successful_runs),
            "avg_total_time_s": sum(times) / len(times),
            "avg_translate_to_en_s": sum(translate_in_times) / len(translate_in_times),
            "avg_llm_inference_s": sum(infer_times) / len(infer_times),
            "avg_translate_to_source_s": sum(translate_out_times) / len(translate_out_times),
            "min_total_time_s": min(times),
            "max_total_time_s": max(times),
            "total_time_s": sum(times),
        }
    
    print("[BENCHMARK] Complete!")
    return results