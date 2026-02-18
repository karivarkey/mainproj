import time
import psutil
import json
import gc
import re
import os
import torch
from pathlib import Path
from sacrebleu.metrics import BLEU, CHRF
from app.config import CPU_ONLY, LLM_DIR, RAG_INDEX_FILE, RAG_META_FILE, USE_ONNX_TRANSLATOR
from app.services.translator_service import translate, unload_translator
from app.services.onnx_translator_service import translate_onnx, unload_onnx_translator
from app.services.llm_service import llm_generate, llm_generate_stream, load_llm, unload_llm, local_gguf_path
from app.services.rag_service import rag_add, rag_clear, rag_list, rag_retrieve

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
    
    # VRAM usage (CUDA) - only track if not in CPU_ONLY mode
    vram_used_mb = 0
    vram_total_mb = 0
    vram_free_mb = 0
    if torch.cuda.is_available() and not CPU_ONLY:
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
    if torch.cuda.is_available() and not CPU_ONLY:
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
            
            # Step 2: LLM inference with streaming + batched translation
            # As LLM generates sentences, translate them incrementally
            t_infer_start = time.perf_counter()
            
            llm_output_en = ""
            translated_sentences = []
            sentence_buffer = ""
            sentence_end_re = re.compile(r'(.+?[.!?](?:\"|\'|")?)?(\s+|$)', re.S)
            
            translation_batches = []
            
            for chunk in llm_generate_stream(english_text, max_new_tokens=max_tokens):
                if chunk is None or not chunk:
                    continue
                    
                sentence_buffer += chunk
                llm_output_en += chunk
                
                # Extract complete sentences from buffer
                while True:
                    m = sentence_end_re.search(sentence_buffer)
                    if not m:
                        break
                    
                    sentence = m.group(1)
                    if sentence:
                        sentence = sentence.strip()
                    sentence_buffer = sentence_buffer[m.end():]
                    
                    if not sentence:
                        continue
                    
                    # Translate sentence immediately (batched processing)
                    t_batch_start = time.perf_counter()
                    translated_sentence = translate(sentence, "en", lang, max_tokens=256)
                    t_batch = time.perf_counter() - t_batch_start
                    
                    translated_sentences.append(translated_sentence)
                    translation_batches.append({
                        "english_sentence": sentence,
                        "translated_sentence": translated_sentence,
                        "translation_time_s": t_batch
                    })
            
            # Translate any remaining buffer content
            if sentence_buffer and sentence_buffer.strip():
                t_batch_start = time.perf_counter()
                translated_sentence = translate(sentence_buffer.strip(), "en", lang, max_tokens=256)
                t_batch = time.perf_counter() - t_batch_start
                
                translated_sentences.append(translated_sentence)
                translation_batches.append({
                    "english_sentence": sentence_buffer.strip(),
                    "translated_sentence": translated_sentence,
                    "translation_time_s": t_batch
                })
            
            t_infer = time.perf_counter() - t_infer_start
            after_infer = memory_snapshot()
            
            # Combine all translated sentences
            final_output = " ".join(translated_sentences) if translated_sentences else ""
            
            # Total translation time from batches
            t_translate_out = sum(b["translation_time_s"] for b in translation_batches) if translation_batches else 0
            
            t_total = time.perf_counter() - t_start
            
            run_result = {
                "prompt_idx": idx,
                "input_text": text,
                "input_lang": lang,
                "english_text": english_text,
                "llm_output_en": llm_output_en,
                "final_output": final_output,
                "num_translation_batches": len(translation_batches),
                "translation_batches": translation_batches,
                "timing": {
                    "translate_to_en_s": t_translate_in,
                    "llm_inference_total_s": t_infer,
                    "translate_to_source_total_s": t_translate_out,
                    "translate_to_source_avg_batch_s": t_translate_out / len(translation_batches) if translation_batches else 0,
                    "total_s": t_total
                },
                "metrics_before": before_run,
                "metrics_after_translate_in": after_translate_in,
                "metrics_after_infer": after_infer,
                "delta": {
                    "rss_mb": after_infer["rss_mb"] - before_run["rss_mb"],
                    "vram_mb": after_infer["vram_used_mb"] - before_run["vram_used_mb"],
                    "cpu_percent": after_infer["cpu_percent"],
                }
            }
            results["inference_runs"].append(run_result)
            print(f"[BENCHMARK]   Total: {t_total:.2f}s (translate_in: {t_translate_in:.2f}s, infer+translate: {t_infer:.2f}s)")
            print(f"[BENCHMARK]   Batched {len(translation_batches)} sentences, avg translation: {t_translate_out / len(translation_batches) if translation_batches else 0:.2f}s/sentence")
            print(f"[BENCHMARK]   CPU: {after_infer['cpu_percent']:.1f}%, RAM: {after_infer['rss_mb']:.1f}MB, VRAM: {after_infer['vram_used_mb']:.1f}MB")
            
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
        infer_times = [r["timing"]["llm_inference_total_s"] for r in successful_runs]
        translate_out_times = [r["timing"]["translate_to_source_total_s"] for r in successful_runs]
        total_batches = sum(r["num_translation_batches"] for r in successful_runs)
        
        results["summary"] = {
            "total_prompts": len(prompts),
            "successful_runs": len(successful_runs),
            "failed_runs": len(prompts) - len(successful_runs),
            "total_translation_batches": total_batches,
            "avg_batches_per_prompt": total_batches / len(successful_runs),
            "avg_total_time_s": sum(times) / len(times),
            "avg_translate_to_en_s": sum(translate_in_times) / len(translate_in_times),
            "avg_llm_inference_s": sum(infer_times) / len(infer_times),
            "avg_translate_to_source_s": sum(translate_out_times) / len(translate_out_times),
            "min_total_time_s": min(times),
            "max_total_time_s": max(times),
            "total_time_s": sum(times),
        }
    
    print("[BENCHMARK] Complete!")

def benchmark_llm_metrics(llm_name: str, n_ctx: int = 4096, n_gpu_layers: int = -1, max_tokens: int = 128):
    """
    Comprehensive LLM performance metrics benchmark.
    
    Measures:
    - Model size (GB)
    - Load time (seconds)
    - First token latency (ms)
    - Tokens per second
    - Peak RAM usage (MB)
    - Output length (tokens)
    
    Uses a predefined complex question for consistent benchmarking.
    """
    DEMO_PROMPT = (
        "Explain the concept of quantum entanglement and its implications "
        "for quantum computing in a detailed manner."
    )
    
    # Ensure we start clean
    unload_llm()
    unload_translator()
    gc.collect()
    if torch.cuda.is_available() and not CPU_ONLY:
        torch.cuda.empty_cache()
    
    time.sleep(1)  # Let system stabilize
    
    results = {}
    
    # 1. Model size
    try:
        model_path = local_gguf_path(llm_name)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model_size_bytes = model_path.stat().st_size
        model_size_gb = model_size_bytes / (1024 ** 3)
        results["model_size_gb"] = round(model_size_gb, 3)
        results["model_path"] = str(model_path)
    except Exception as e:
        return {"error": f"Failed to get model size: {e}"}
    
    # Baseline memory before loading
    baseline_mem = memory_snapshot()
    
    # 2. Load time
    try:
        load_start = time.perf_counter()
        load_llm(llm_name, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
        load_end = time.perf_counter()
        load_time_s = load_end - load_start
        results["load_time_s"] = round(load_time_s, 3)
    except Exception as e:
        unload_llm()
        return {"error": f"Failed to load model: {e}"}
    
    # Memory after loading
    loaded_mem = memory_snapshot()
    
    # 3-6. Inference metrics using streaming
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    
    try:
        # Track first token latency
        first_token_time = None
        inference_start = time.perf_counter()
        token_count = 0
        output_text = ""
        
        for chunk in llm_generate_stream(DEMO_PROMPT, max_new_tokens=max_tokens):
            if first_token_time is None:
                first_token_time = time.perf_counter()
            output_text += chunk
            token_count += 1
            
            # Update peak memory
            current_rss = process.memory_info().rss
            peak_rss = max(peak_rss, current_rss)
        
        inference_end = time.perf_counter()
        
        # Calculate metrics
        first_token_latency_ms = ((first_token_time - inference_start) * 1000) if first_token_time else 0
        total_inference_time_s = inference_end - inference_start
        tokens_per_second = token_count / total_inference_time_s if total_inference_time_s > 0 else 0
        
        results["first_token_latency_ms"] = round(first_token_latency_ms, 2)
        results["tokens_per_second"] = round(tokens_per_second, 2)
        results["output_length_tokens"] = token_count
        results["output_text"] = output_text
        results["total_inference_time_s"] = round(total_inference_time_s, 3)
        
    except Exception as e:
        unload_llm()
        return {"error": f"Inference failed: {e}"}
    
    # Peak memory after inference
    peak_mem = memory_snapshot()
    
    # 5. Memory metrics
    results["memory"] = {
        "baseline_rss_mb": round(baseline_mem["rss_mb"], 2),
        "loaded_rss_mb": round(loaded_mem["rss_mb"], 2),
        "peak_rss_mb": round(peak_rss / (1024 * 1024), 2),
        "load_increase_mb": round(loaded_mem["rss_mb"] - baseline_mem["rss_mb"], 2),
        "inference_increase_mb": round((peak_rss / (1024 * 1024)) - loaded_mem["rss_mb"], 2),
    }
    
    # VRAM if available
    if torch.cuda.is_available() and not CPU_ONLY:
        results["vram"] = {
            "baseline_used_mb": round(baseline_mem["vram_used_mb"], 2),
            "loaded_used_mb": round(loaded_mem["vram_used_mb"], 2),
            "peak_used_mb": round(peak_mem["vram_used_mb"], 2),
            "total_mb": round(loaded_mem["vram_total_mb"], 2),
        }
    
    # Metadata
    results["config"] = {
        "llm_name": llm_name,
        "n_ctx": n_ctx,
        "n_gpu_layers": n_gpu_layers,
        "max_tokens": max_tokens,
        "demo_prompt": DEMO_PROMPT,
    }
    
    # Clean up
    unload_llm()
    
    return results


def benchmark_translator_metrics(src_lang: str, tgt_lang: str = "en", use_onnx: bool = None):
    """
    Comprehensive translator performance metrics.
    
    Measures:
    - Input text length (characters and tokens)
    - Translation throughput (chars/sec, tokens/sec)
    - Memory usage during translation
    - Translation quality (BLEU, CHRF for round-trip)
    - End-to-end response time
    
    Uses a predefined complex multilingual text for consistent benchmarking.
    
    Args:
        src_lang: Source language code
        tgt_lang: Target language code
        use_onnx: Use ONNX backend if True, NLLB if False, auto-detect if None
    """
    # Determine which backend to use
    if use_onnx is None:
        use_onnx = USE_ONNX_TRANSLATOR
    
    translate_fn = translate_onnx if use_onnx else translate
    unload_fn = unload_onnx_translator if use_onnx else unload_translator
    backend = "onnx" if use_onnx else "nllb"
    # Complex demo texts for different languages
    DEMO_TEXTS = {
        # Indian Languages
        "hi": "प्रौद्योगिकी दुनिया भर के लोगों को जोड़ती है। नई भाषाएं सीखना कई अवसर खोलता है।",
        "bn": "প্রযুক্তি বিশ্বজুড়ে মানুষকে সংযুক্ত করে। নতুন ভাষা শেখা অনেক সুযোগের দ্বার খুলে দেয়।",
        "mr": "तंत्रज्ञान जगभरातील लोकांना जोडते। नवीन भाषा शिकल्याने अनेक संधी उपलब्ध होतात।",
        "gu": "ટેકનોલોજી વિશ્વભરના લોકોને જોડે છે. નવી ભાષાઓ શીખવાથી ઘણી તકો ખુલે છે.",
        "pa": "ਤਕਨਾਲੋਜੀ ਦੁਨੀਆ ਭਰ ਦੇ ਲੋਕਾਂ ਨੂੰ ਜੋੜਦੀ ਹੈ। ਨਵੀਆਂ ਭਾਸ਼ਾਵਾਂ ਸਿੱਖਣਾ ਕਈ ਮੌਕੇ ਖੋਲ੍ਹਦਾ ਹੈ।",
        "ur": "ٹیکنالوجی دنیا بھر کے لوگوں کو جوڑتی ہے۔ نئی زبانیں سیکھنا کئی مواقع فراہم کرتا ہے۔",
        "as": "প্রযুক্তিয়ে বিশ্বজুৰি মানুহক সংযোগ কৰে। নতুন ভাষা শিকিলে বহু সুযোগ উন্মুক্ত হয়।",
        "or": "ଟେକ୍ନୋଲୋଜି ବିଶ୍ୱବ୍ୟାପୀ ଲୋକଙ୍କୁ ସଂଯୋଗ କରେ | ନୂତନ ଭାଷା ଶିଖିବା ଅନେକ ସୁଯୋଗ ଖୋଲିଥାଏ |",
        "ta": "தொழில்நுட்பம் உலகம் முழுவதும் உள்ள மக்களை இணைக்கிறது. புதிய மொழிகளைக் கற்றுக்கொள்வது பல வாய்ப்புகளைத் திறக்கிறது.",
        "te": "తెలుగు భాష ద్రావిడ భాషా కుటుంబంలో అతిపెద్ద భాషగా పరిగణించబడుతుంది మరియు భారతదేశంలో అత్యధికంగా మాట్లాడే భాషల్లో ఒకటి. ఇది ఆంధ్రప్రదేశ్ మరియు తెలంగాణ రాష్ట్రాల అధికారిక భాష. తెలుగు సాహిత్యం గొప్ప చారిత్రక సంప్రదాయాన్ని కలిగి ఉంది.",
        "kn": "ತಂತ್ರಜ್ಞಾನವು ಪ್ರಪಂಚದಾದ್ಯಂತದ ಜನರನ್ನು ಸಂಪರ್ಕಿಸುತ್ತದೆ. ಹೊಸ ಭಾಷೆಗಳನ್ನು ಕಲಿಯುವುದು ಅನೇಕ ಅವಕಾಶಗಳನ್ನು ತೆರೆಯುತ್ತದೆ.",
        "ml": "സാങ്കേതികവിദ്യ ലോകമെമ്പാടുമുള്ള ആളുകളെ ബന്ധിപ്പിക്കുന്നു. പുതിയ ഭാഷകൾ പഠിക്കുന്നത് നിരവധി അവസരങ്ങൾ തുറക്കുന്നു.",
        # International Languages
        "en": "Technology connects people across the world. Learning new languages opens many opportunities.",
        "fr": "La technologie connecte les gens à travers le monde. Apprendre de nouvelles langues ouvre de nombreuses opportunités.",
        "de": "Technologie verbindet Menschen auf der ganzen Welt. Das Lernen neuer Sprachen eröffnet viele Möglichkeiten.",
        "es": "La tecnología conecta a las personas en todo el mundo. Aprender nuevos idiomas abre muchas oportunidades.",
        "pt": "A tecnologia conecta pessoas em todo o mundo. Aprender novos idiomas abre muitas oportunidades.",
        "ru": "Технологии объединяют людей по всему миру. Изучение новых языков открывает много возможностей.",
        "ja": "テクノロジーは世界中の人々をつなぎます。新しい言語を学ぶことは多くの機会を開きます。",
        "zh": "技术将世界各地的人们联系在一起。学习新语言会带来许多机会。",
    }
    
    # Get demo text or default to English
    demo_text = DEMO_TEXTS.get(src_lang, DEMO_TEXTS["en"])
    
    # Warm-up: Pre-load models before benchmarking
    # This ensures we measure "warm" inference latency, not cold start
    print(f"[BENCHMARK] Warming up {backend} translator...")
    try:
        # Run a quick translation to load all models into memory
        _ = translate_fn("Technology connects people.", src_lang, tgt_lang, max_tokens=50)
        print(f"[BENCHMARK] {backend} models loaded and cached")
    except Exception as e:
        print(f"[BENCHMARK] Warm-up failed: {e}")
        return {"error": f"Model warm-up failed: {e}"}
    
    # Clean memory counters (not models)
    gc.collect()
    if torch.cuda.is_available() and not CPU_ONLY:
        torch.cuda.empty_cache()
    time.sleep(0.3)
    
    results = {}
    bleu = BLEU()
    chrf = CHRF()
    
    # 1. Input metrics
    input_char_length = len(demo_text)
    # Rough token estimation (whitespace split + punctuation)
    input_token_length = len(demo_text.split())
    
    results["input"] = {
        "text": demo_text,
        "char_length": input_char_length,
        "token_length_estimate": input_token_length,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "backend": backend,
    }
    
    # Baseline memory
    baseline_mem = memory_snapshot()
    
    # 5. End-to-end timing starts
    e2e_start = time.perf_counter()
    
    # Forward translation (src -> tgt)
    try:
        fwd_start = time.perf_counter()
        translated_text = translate_fn(demo_text, src_lang, tgt_lang, max_tokens=512)
        fwd_end = time.perf_counter()
        fwd_time_s = fwd_end - fwd_start
        
        fwd_char_length = len(translated_text)
        fwd_token_length = len(translated_text.split())
        
    except Exception as e:
        unload_fn()
        return {"error": f"Forward translation failed: {e}"}
    
    # Memory after forward translation
    fwd_mem = memory_snapshot()
    
    # Round-trip translation (tgt -> src) for quality measurement
    try:
        rt_start = time.perf_counter()
        roundtrip_text = translate_fn(translated_text, tgt_lang, src_lang, max_tokens=512)
        rt_end = time.perf_counter()
        rt_time_s = rt_end - rt_start
        
        rt_char_length = len(roundtrip_text)
        rt_token_length = len(roundtrip_text.split())
        
    except Exception as e:
        unload_fn()
        return {"error": f"Round-trip translation failed: {e}"}
    
    # Peak memory after round-trip
    peak_mem = memory_snapshot()
    
    e2e_end = time.perf_counter()
    e2e_time_s = e2e_end - e2e_start
    
    # 2. Translation throughput
    fwd_chars_per_sec = input_char_length / fwd_time_s if fwd_time_s > 0 else 0
    fwd_tokens_per_sec = input_token_length / fwd_time_s if fwd_time_s > 0 else 0
    
    rt_chars_per_sec = fwd_char_length / rt_time_s if rt_time_s > 0 else 0
    rt_tokens_per_sec = fwd_token_length / rt_time_s if rt_time_s > 0 else 0
    
    results["throughput"] = {
        "forward": {
            "chars_per_sec": round(fwd_chars_per_sec, 2),
            "tokens_per_sec": round(fwd_tokens_per_sec, 2),
            "time_s": round(fwd_time_s, 3),
        },
        "roundtrip": {
            "chars_per_sec": round(rt_chars_per_sec, 2),
            "tokens_per_sec": round(rt_tokens_per_sec, 2),
            "time_s": round(rt_time_s, 3),
        },
    }
    
    # 4. Translation quality (effect on downstream accuracy)
    try:
        bleu_score = bleu.sentence_score(roundtrip_text, [demo_text]).score
        chrf_score = chrf.sentence_score(roundtrip_text, [demo_text]).score
    except Exception as e:
        bleu_score = 0.0
        chrf_score = 0.0
    
    # Character-level similarity as additional metric
    char_similarity = (1 - (abs(len(roundtrip_text) - len(demo_text)) / max(len(roundtrip_text), len(demo_text)))) * 100
    
    results["quality"] = {
        "bleu_score": round(bleu_score, 2),
        "chrf_score": round(chrf_score, 2),
        "char_length_similarity_pct": round(char_similarity, 2),
        "forward_output_chars": fwd_char_length,
        "forward_output_tokens": fwd_token_length,
        "roundtrip_output_chars": rt_char_length,
        "roundtrip_output_tokens": rt_token_length,
    }
    
    # 3. Memory usage
    results["memory"] = {
        "baseline_rss_mb": round(baseline_mem["rss_mb"], 2),
        "after_forward_rss_mb": round(fwd_mem["rss_mb"], 2),
        "peak_rss_mb": round(peak_mem["rss_mb"], 2),
        "translation_increase_mb": round(fwd_mem["rss_mb"] - baseline_mem["rss_mb"], 2),
        "peak_increase_mb": round(peak_mem["rss_mb"] - baseline_mem["rss_mb"], 2),
    }
    
    # VRAM if available
    if torch.cuda.is_available() and not CPU_ONLY:
        results["vram"] = {
            "baseline_used_mb": round(baseline_mem["vram_used_mb"], 2),
            "after_forward_used_mb": round(fwd_mem["vram_used_mb"], 2),
            "peak_used_mb": round(peak_mem["vram_used_mb"], 2),
            "total_mb": round(fwd_mem["vram_total_mb"], 2),
        }
    
    # 5. End-to-end response time
    results["end_to_end_time_s"] = round(e2e_time_s, 3)
    
    # Output texts
    results["outputs"] = {
        "forward_translation": translated_text,
        "roundtrip_translation": roundtrip_text,
    }
    
    # Clean up
    unload_fn()
    
    return results


def benchmark_rag_metrics(llm_name: str = None, n_ctx: int = 4096, n_gpu_layers: int = -1):
    """
    Comprehensive RAG performance metrics.
    
    Measures:
    - Documents indexed count
    - Index size on disk (MB)
    - Query retrieval time
    - Top-k retrieval time for different k values
    - Memory usage with FAISS
    - Recall/relevance (simulated with known queries)
    - Impact of RAG on final answer quality
    
    Workflow:
    1. Save existing RAG data
    2. Clear RAG
    3. Add demo data
    4. Measure metrics
    5. Clear RAG
    6. Restore previous data
    """
    # Demo RAG documents for benchmarking
    DEMO_DOCS = [
        "Quantum computing uses quantum bits or qubits that can exist in superposition states, enabling parallel computation.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data without explicit programming.",
        "The Transformer architecture revolutionized NLP by introducing self-attention mechanisms for processing sequential data.",
        "BERT (Bidirectional Encoder Representations from Transformers) uses masked language modeling for pre-training.",
        "GPT models are autoregressive language models trained on vast amounts of text data using next-token prediction.",
        "Neural networks consist of layers of interconnected nodes that process information through weighted connections.",
        "Backpropagation is the algorithm used to train neural networks by computing gradients of the loss function.",
        "Convolutional Neural Networks (CNNs) are primarily used for image processing and computer vision tasks.",
        "Recurrent Neural Networks (RNNs) process sequential data by maintaining hidden states across time steps.",
        "Attention mechanisms allow models to focus on relevant parts of the input when making predictions.",
    ]
    
    # Demo queries with expected relevant doc indices
    DEMO_QUERIES = [
        {"query": "What is quantum computing?", "relevant_docs": [0]},
        {"query": "Explain transformers in NLP", "relevant_docs": [2, 3, 4]},
        {"query": "How do neural networks learn?", "relevant_docs": [5, 6]},
        {"query": "What are CNNs used for?", "relevant_docs": [7]},
    ]
    
    results = {}
    
    # 1. Save existing RAG data
    print("[RAG Benchmark] Saving existing RAG data...")
    existing_docs = rag_list()
    existing_count = len(existing_docs)
    print(f"[RAG Benchmark] Found {existing_count} existing documents")
    
    # 2. Clear RAG
    rag_clear()
    gc.collect()
    if torch.cuda.is_available() and not CPU_ONLY:
        torch.cuda.empty_cache()
    time.sleep(0.5)
    
    # Baseline memory before indexing
    baseline_mem = memory_snapshot()
    
    # 3. Add demo data and measure indexing time
    indexing_start = time.perf_counter()
    doc_ids = []
    for doc in DEMO_DOCS:
        doc_id = rag_add(doc)
        doc_ids.append(doc_id)
    indexing_end = time.perf_counter()
    indexing_time_s = indexing_end - indexing_start
    
    # Memory after indexing
    indexed_mem = memory_snapshot()
    
    # 1. Documents indexed
    indexed_count = len(rag_list())
    results["documents_indexed"] = indexed_count
    results["indexing_time_s"] = round(indexing_time_s, 3)
    
    # 2. Index size on disk
    index_size_mb = 0
    meta_size_mb = 0
    if RAG_INDEX_FILE.exists():
        index_size_mb = RAG_INDEX_FILE.stat().st_size / (1024 * 1024)
    if RAG_META_FILE.exists():
        meta_size_mb = RAG_META_FILE.stat().st_size / (1024 * 1024)
    
    results["index_size"] = {
        "index_file_mb": round(index_size_mb, 4),
        "metadata_file_mb": round(meta_size_mb, 4),
        "total_mb": round(index_size_mb + meta_size_mb, 4),
    }
    
    # 3 & 4. Query retrieval time and top-k retrieval
    retrieval_times = []
    topk_times = {1: [], 3: [], 5: []}
    
    for query_data in DEMO_QUERIES:
        query = query_data["query"]
        
        # Measure single query time (top-3 default)
        start = time.perf_counter()
        _ = rag_retrieve(query, top_k=3)
        end = time.perf_counter()
        retrieval_times.append(end - start)
        
        # Measure different top-k values
        for k in [1, 3, 5]:
            start = time.perf_counter()
            _ = rag_retrieve(query, top_k=k)
            end = time.perf_counter()
            topk_times[k].append(end - start)
    
    avg_retrieval_time_ms = (sum(retrieval_times) / len(retrieval_times)) * 1000
    
    results["retrieval_performance"] = {
        "avg_query_time_ms": round(avg_retrieval_time_ms, 3),
        "min_query_time_ms": round(min(retrieval_times) * 1000, 3),
        "max_query_time_ms": round(max(retrieval_times) * 1000, 3),
        "topk_avg_times_ms": {
            k: round((sum(times) / len(times)) * 1000, 3)
            for k, times in topk_times.items()
        },
    }
    
    # 5. Memory usage with FAISS
    results["memory"] = {
        "baseline_rss_mb": round(baseline_mem["rss_mb"], 2),
        "after_indexing_rss_mb": round(indexed_mem["rss_mb"], 2),
        "indexing_increase_mb": round(indexed_mem["rss_mb"] - baseline_mem["rss_mb"], 2),
    }
    
    if torch.cuda.is_available() and not CPU_ONLY:
        results["vram"] = {
            "baseline_used_mb": round(baseline_mem["vram_used_mb"], 2),
            "after_indexing_used_mb": round(indexed_mem["vram_used_mb"], 2),
        }
    
    # 6. Recall/relevance measurement
    recall_scores = []
    for query_data in DEMO_QUERIES:
        query = query_data["query"]
        relevant_indices = set(query_data["relevant_docs"])
        
        # Get top-3 results
        retrieved_docs = rag_retrieve(query, top_k=3)
        
        # Map retrieved docs back to original indices
        retrieved_indices = set()
        for retrieved_doc in retrieved_docs:
            for idx, demo_doc in enumerate(DEMO_DOCS):
                if demo_doc == retrieved_doc:
                    retrieved_indices.add(idx)
                    break
        
        # Calculate recall: how many relevant docs were retrieved
        if len(relevant_indices) > 0:
            recall = len(retrieved_indices & relevant_indices) / len(relevant_indices)
            recall_scores.append(recall)
    
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    
    results["relevance"] = {
        "avg_recall_at_3": round(avg_recall, 3),
        "queries_evaluated": len(DEMO_QUERIES),
        "perfect_recalls": sum(1 for r in recall_scores if r == 1.0),
    }
    
    # 7. Impact of RAG on final answer (if LLM is available)
    if llm_name:
        try:
            # Load LLM for answer quality comparison
            load_llm(llm_name, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
            
            test_query = "What is quantum computing and how does it work?"
            
            # Answer WITHOUT RAG
            no_rag_start = time.perf_counter()
            answer_no_rag = llm_generate(test_query, max_new_tokens=64)
            no_rag_time = time.perf_counter() - no_rag_start
            
            # Answer WITH RAG
            rag_docs = rag_retrieve(test_query, top_k=3)
            context = "\n".join(f"Context {i+1}: {d}" for i, d in enumerate(rag_docs))
            prompt_with_rag = f"Context:\n{context}\n\nQuestion: {test_query}\nAnswer:"
            
            with_rag_start = time.perf_counter()
            answer_with_rag = llm_generate(prompt_with_rag, max_new_tokens=64)
            with_rag_time = time.perf_counter() - with_rag_start
            
            # Simple quality metrics
            rag_impact = {
                "query": test_query,
                "answer_without_rag": answer_no_rag.strip(),
                "answer_with_rag": answer_with_rag.strip(),
                "answer_length_diff": len(answer_with_rag) - len(answer_no_rag),
                "inference_time_without_rag_s": round(no_rag_time, 3),
                "inference_time_with_rag_s": round(with_rag_time, 3),
                "rag_overhead_s": round(with_rag_time - no_rag_time, 3),
                "contexts_used": len(rag_docs),
            }
            
            results["rag_impact"] = rag_impact
            
            # Unload LLM
            unload_llm()
            
        except Exception as e:
            results["rag_impact"] = {"error": f"LLM comparison failed: {e}"}
    else:
        results["rag_impact"] = {"skipped": "No LLM specified for impact analysis"}
    
    # 5. Clear RAG and restore previous data
    print("[RAG Benchmark] Clearing demo data and restoring previous state...")
    rag_clear()
    
    # Restore previous documents
    if existing_docs:
        for doc_data in existing_docs:
            rag_add(doc_data["text"])
    
    restored_count = len(rag_list())
    print(f"[RAG Benchmark] Restored {restored_count} documents")
    
    results["restoration"] = {
        "original_doc_count": existing_count,
        "restored_doc_count": restored_count,
    }
    
    return results