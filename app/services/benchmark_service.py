import time
import psutil
import json
import gc
import re
import os
import torch
from pathlib import Path
from sacrebleu.metrics import BLEU, CHRF
from app.config import CPU_ONLY, LLM_DIR
from app.services.translator_service import translate, unload_translator
from app.services.llm_service import llm_generate, llm_generate_stream, load_llm, unload_llm, local_gguf_path
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


def benchmark_translator_metrics(src_lang: str, tgt_lang: str = "en"):
    """
    Comprehensive translator performance metrics.
    
    Measures:
    - Input text length (characters and tokens)
    - Translation throughput (chars/sec, tokens/sec)
    - Memory usage during translation
    - Translation quality (BLEU, CHRF for round-trip)
    - End-to-end response time
    
    Uses a predefined complex multilingual text for consistent benchmarking.
    """
    # Complex demo texts for different languages
    DEMO_TEXTS = {
        "hi": "भारतीय संविधान विश्व का सबसे लंबा लिखित संविधान है जो 26 जनवरी 1950 को लागू हुआ था। यह दस्तावेज़ भारत के नागरिकों के मौलिक अधिकारों, राज्य के नीति निदेशक तत्वों और सरकार की संरचना को परिभाषित करता है। संविधान सभा ने इसे तीन वर्षों में तैयार किया था।",
        "ta": "தமிழ் மொழி உலகின் மிகப் பழமையான மொழிகளில் ஒன்றாகும் மற்றும் இது சங்க இலக்கியத்தின் வளமான பாரம்பரியத்தைக் கொண்டுள்ளது। இந்த மொழி தமிழ்நாடு மற்றும் புதுச்சேரியில் உள்ள மக்களின் தாய்மொழியாக விளங்குகிறது மற்றும் இலங்கை மற்றும் சிங்கப்பூரில் குறிப்பிடத்தக்க எண்ணிக்கையில் பேசப்படுகிறது।",
        "ml": "കേരളം ഇന്ത്യയുടെ തെക്കുപടിഞ്ഞാറൻ തീരത്ത് സ്ഥിതി ചെയ്യുന്ന ഒരു സംസ്ഥാനമാണ്. ഇത് അതിന്റെ പ്രകൃതി സൗന്ദര്യം, ബാക്ക്‌വാട്ടറുകൾ, ആയുർവേദ ചികിത്സ, കത്തകളി നൃത്തം എന്നിവയ്ക്ക് പ്രസിദ്ധമാണ്. കേരളത്തിന് നൂറ് ശതമാനം സാക്ഷരതയും ഉയർന്ന ജീവിത നിലവാരവും ഉണ്ട്।",
        "te": "తెలుగు భాష ద్రావిడ భాషా కుటుంబంలో అతిపెద్ద భాషగా పరిగణించబడుతుంది మరియు భారతదేశంలో అత్యధికంగా మాట్లాడే భాషల్లో ఒకటి. ఇది ఆంధ్రప్రదేశ్ మరియు తెలంగాణ రాష్ట్రాల అధికారిక భాష. తెలుగు సాహిత్యం గొప్ప చారిత్రక సంప్రదాయాన్ని కలిగి ఉంది.",
        "bn": "বাংলা ভাষা বিশ্বের সপ্তম সর্বাধিক কথিত ভাষা এবং বাংলাদেশ ও ভারতের পশ্চিমবঙ্গে প্রধান ভাষা হিসেবে ব্যবহৃত হয়। এই ভাষার সমৃদ্ধ সাহিত্যিক ঐতিহ্য রয়েছে যা রবীন্দ্রনাথ ঠাকুর এবং কাজী নজরুল ইসলামের মতো মহান কবিদের অবদান অন্তর্ভুক্ত করে।",
        "en": "The Renaissance was a period of cultural, artistic, political and economic rebirth following the Middle Ages. Generally described as taking place from the 14th century to the 17th century, it promoted the rediscovery of classical philosophy, literature and art. This transformative era laid the foundation for modern Western civilization.",
    }
    
    # Get demo text or default to English
    demo_text = DEMO_TEXTS.get(src_lang, DEMO_TEXTS["en"])
    
    # Ensure clean state
    unload_translator()
    gc.collect()
    if torch.cuda.is_available() and not CPU_ONLY:
        torch.cuda.empty_cache()
    time.sleep(0.5)
    
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
    }
    
    # Baseline memory
    baseline_mem = memory_snapshot()
    
    # 5. End-to-end timing starts
    e2e_start = time.perf_counter()
    
    # Forward translation (src -> tgt)
    try:
        fwd_start = time.perf_counter()
        translated_text = translate(demo_text, src_lang, tgt_lang, max_tokens=512)
        fwd_end = time.perf_counter()
        fwd_time_s = fwd_end - fwd_start
        
        fwd_char_length = len(translated_text)
        fwd_token_length = len(translated_text.split())
        
    except Exception as e:
        unload_translator()
        return {"error": f"Forward translation failed: {e}"}
    
    # Memory after forward translation
    fwd_mem = memory_snapshot()
    
    # Round-trip translation (tgt -> src) for quality measurement
    try:
        rt_start = time.perf_counter()
        roundtrip_text = translate(translated_text, tgt_lang, src_lang, max_tokens=512)
        rt_end = time.perf_counter()
        rt_time_s = rt_end - rt_start
        
        rt_char_length = len(roundtrip_text)
        rt_token_length = len(roundtrip_text.split())
        
    except Exception as e:
        unload_translator()
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
    unload_translator()
    
    return results