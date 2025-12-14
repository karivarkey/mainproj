from flask import Blueprint, request, jsonify, Response, stream_with_context
import json
import re
import time
import os
import psutil
from app.config import LANG_MAP, LANG_ALIASES, NLLB_LANG_MAP
from app.services.cache_service import model_cache
from app.services.llm_service import download_gguf, load_llm_from_gguf, llm_generate, llm_generate_stream, unload_llm, get_current_name, SERVER_URL
from app.services.translator_service import translate, detect_supported_language, unload_translator
from app.services.rag_service import rag_add, rag_remove, rag_retrieve, rag_list, rag_clear
from app.services.benchmark_service import benchmark_pipeline, benchmark_resource_usage, benchmark_llm_metrics


def _clean_generation(text: str) -> str:
    """Remove code fences and noisy prefixes from model output."""
    cleaned = text
    cleaned = re.sub(r"```.*?```", " ", cleaned, flags=re.S)  # drop fenced blocks
    cleaned = re.sub(r"`+", "", cleaned)  # drop stray backticks
    cleaned = re.sub(r"^\s*Answer\s*:\s*", "", cleaned, flags=re.I)  # drop leading Answer:
    return cleaned.strip()

bp = Blueprint("api", __name__)

@bp.post("/download_llm")
def ep_download_llm():
    body = request.get_json() or {}
    url = body.get("url")
    name = body.get("name")
    if not url:
        return jsonify({"error": "url required"}), 400
    if not name:
        return jsonify({"error": "name required"}), 400
    path = download_gguf(url, name)
    return jsonify({"ok": True, "path": str(path)})

@bp.get("/list_llms")
def ep_list_llms():
    return jsonify({
        "downloaded_llms": model_cache["llms"],
        "loaded_llm": get_current_name(),
        "server_url": SERVER_URL,
    })

@bp.post("/load_llm")
def ep_load_llm():
    body = request.get_json() or {}
    name = body.get("name")
    port = int(body.get("port", 8080))  # Ignored for llama-cpp, kept for compatibility
    ctx_size = int(body.get("ctx_size", 4096))
    n_gpu_layers = int(body.get("n_gpu_layers", -1))
    if not name:
        return jsonify({"error": "name required"}), 400
    # load_llm_from_gguf accepts n_ctx and n_gpu_layers, not port or ctx_size
    load_llm_from_gguf(name, n_ctx=ctx_size, n_gpu_layers=n_gpu_layers)
    return jsonify({"ok": True, "loaded": name, "server_url": SERVER_URL})

@bp.get("/current_llm")
def ep_current_llm():
    """Get the currently loaded LLM name."""
    return jsonify({
        "ok": True,
        "loaded_llm": get_current_name(),
        "server_url": SERVER_URL
    })

@bp.post("/translate")
def ep_translate():
    body = request.get_json() or {}
    text = body.get("text")
    target = (body.get("target") or "en").lower()
    stream = bool(body.get("stream", True))
    max_tokens = int(body.get("max_new_tokens", 256))

    if not text:
        return jsonify({"error": "text required"}), 400

    # Auto-detect source language (must be in LANG_CONF)
    src_lang_key = detect_supported_language(text)
    if not src_lang_key:
        return jsonify({"error": "could not auto-detect a supported language"}), 400

    src_code, _ = LANG_MAP[src_lang_key]

    # Normalize target and validate it against known mappings or raw NLLB codes
    target_key = LANG_ALIASES.get(target, target)
    target_code = NLLB_LANG_MAP.get(target_key, target_key)
    if target_key not in LANG_MAP and target_key not in NLLB_LANG_MAP and "_" not in target_key:
        return jsonify({"error": f"unsupported target language: {target}"}), 400

    # Helper to iterate sentences from paragraphs
    sentence_end_re = re.compile(r"(.+?[.!?](?:\"|'|”)?)(\s+|$)", re.S)

    def iter_sentences(blob: str):
        buffer = blob
        while True:
            match = sentence_end_re.search(buffer)
            if not match:
                break
            sent = match.group(1).strip()
            buffer = buffer[match.end():]
            if sent:
                yield sent
        if buffer.strip():
            yield buffer.strip()

    if not stream:
        translated_sentences = []
        for sent in iter_sentences(text):
            translated_sentences.append({
                "source": sent,
                "translated": translate(sent, src_code, target_code, max_tokens),
            })

        combined = " ".join(item["translated"] for item in translated_sentences)
        return jsonify({
            "input": text,
            "detected_lang": src_lang_key,
            "target_lang": target_key,
            "translated_text": combined,
            "sentences": translated_sentences,
        })

    def event_stream():
        meta = {
            "type": "meta",
            "detected_lang": src_lang_key,
            "target_lang": target_key,
        }
        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

        try:
            for idx, sent in enumerate(iter_sentences(text), start=1):
                translated = translate(sent, src_code, target_code, max_tokens)
                payload = {
                    "type": "sentence",
                    "index": idx,
                    "source": sent,
                    "translated": translated,
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
        except Exception as e:
            err = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    headers = {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return Response(stream_with_context(event_stream()), headers=headers)


@bp.get("/system/metrics")
def ep_system_metrics():
    """
    Stream system metrics via SSE: CPU%, RAM (system + current process), and VRAM if available.

    Query params:
      - interval_ms: polling interval in milliseconds (default 1000)
      - include_process: whether to include current python process stats (default true)
    """
    try:
        interval_ms = int(request.args.get("interval_ms", "1000"))
    except ValueError:
        interval_ms = 1000
    include_process = (request.args.get("include_process", "true").lower() in ("true", "1", "yes"))

    process = psutil.Process(os.getpid()) if include_process else None

    # Prime psutil CPU percent to compute over interval
    psutil.cpu_percent(interval=None)

    def get_vram():
        try:
            import torch
            if torch.cuda.is_available():
                total = torch.cuda.get_device_properties(0).total_memory
                used = torch.cuda.memory_allocated(0)
                reserved = torch.cuda.memory_reserved(0)
                return {
                    "available": True,
                    "total_bytes": int(total),
                    "used_bytes": int(used),
                    "reserved_bytes": int(reserved),
                }
        except Exception as e:
            return {"available": False, "error": str(e)}
        return {"available": False}

    def event_stream():
        try:
            while True:
                cpu_pct = psutil.cpu_percent(interval=None)
                vm = psutil.virtual_memory()
                swap = psutil.swap_memory()

                payload = {
                    "type": "metrics",
                    "timestamp": time.time(),
                    "cpu_percent": cpu_pct,
                    "ram": {
                        "total_bytes": int(vm.total),
                        "used_bytes": int(vm.used),
                        "available_bytes": int(vm.available),
                        "percent": float(vm.percent),
                    },
                    "swap": {
                        "total_bytes": int(swap.total),
                        "used_bytes": int(swap.used),
                        "percent": float(swap.percent),
                    },
                    "vram": get_vram(),
                }

                if process is not None:
                    with process.oneshot():
                        mem_info = process.memory_info()
                        cpu_proc = process.cpu_percent(interval=None)
                        payload["process"] = {
                            "pid": process.pid,
                            "cpu_percent": cpu_proc,
                            "rss_bytes": int(mem_info.rss),  # resident set size
                            "vms_bytes": int(mem_info.vms),  # virtual memory size
                        }

                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                time.sleep(max(0.05, interval_ms / 1000.0))
        except GeneratorExit:
            # Client disconnected; stop the stream
            return
        except Exception as e:
            err = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    headers = {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return Response(stream_with_context(event_stream()), headers=headers)

@bp.post("/infer")
def ep_infer():
    body = request.get_json() or {}
    text = body.get("text")
    lang = body.get("lang")
    max_tokens = int(body.get("max_new_tokens", 128))

    if not text:
        return jsonify({"error": "text required"}), 400

    # Normalize language input and optionally auto-detect
    lang = (lang or "").lower()
    detected_lang = None
    if lang == "auto":
        detected_lang = detect_supported_language(text)
        if not detected_lang:
            return jsonify({"error": "could not auto-detect a supported language"}), 400
        lang = detected_lang
    else:
        lang = LANG_ALIASES.get(lang, lang)

    if not lang or lang not in LANG_MAP:
        return jsonify({"error": f"unsupported lang: {lang}"}), 400

    src_lang, en_lang = LANG_MAP[lang]
    print(f"Translating from {src_lang} to {en_lang} and back.")
    
    try:
        # 1. Convert user input → English (using short codes)
        english_text = translate(text, src_lang, "en")
        print(f"Translated input to English: {english_text}")
    except RuntimeError as e:
        if "torchvision" in str(e) or "nms" in str(e):
            return jsonify({
                "error": "translator initialization failed due to dependency conflict",
                "details": str(e),
                "suggestion": "Try: pip install --upgrade transformers torch torchvision"
            }), 503
        raise

    # If client requests streaming (default True), stream sentence-by-sentence
    stream = bool(body.get("stream", True))

    # 2. RAG retrieve
    rag_docs = rag_retrieve(english_text, top_k=3)
    context = ""
    if rag_docs:
        context = "\n".join(f"Document {i+1}: {d}" for i, d in enumerate(rag_docs))

    # 3. Build prompt for LLM
    context_block = f"Relevant context:\n{context}\n" if context else ""
    final_prompt = (
        f"User question:\n{english_text}\n\n"
        f"{context_block}"
        "Answer clearly in simple English. Do not use code fences or Markdown. Respond with plain text only."
    )

    # 4/5. Run inference and (optionally) stream translations back
    if not stream:
        # Non-streaming (legacy) behaviour
        try:
            llm_output_en = llm_generate(final_prompt, max_new_tokens=max_tokens)
            llm_output_en = _clean_generation(llm_output_en)
        except RuntimeError as e:
            if "llama_decode returned -1" in str(e):
                return jsonify({
                    "error": "LLM generation failed due to context/memory limits",
                    "details": str(e),
                    "suggestion": "Try loading the model with a larger n_ctx or reduce max_new_tokens"
                }), 503
            raise
        
        answer_native = translate(llm_output_en, "en", src_lang)
        return jsonify({
            "input": text,
            "english_in": english_text,
            "rag_used": rag_docs,
            "llm_prompt": final_prompt,
            "llm_output_en": llm_output_en,
            "final_output": answer_native,
            "lang_used": lang,
            "detected_lang": detected_lang,
        })

    # Streaming response (SSE). We will yield JSON payloads per translated sentence.
    sentence_end_re = re.compile(r"(.+?[.!?](?:\"|'|”)?)(\s+|$)", re.S)

    def event_stream():
        # Meta event with initial English input
        meta = {"type": "meta", "english_in": english_text, "prompt": final_prompt, "lang_used": lang, "detected_lang": detected_lang}
        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

        buffer = ""
        try:
            for chunk in llm_generate_stream(final_prompt, max_new_tokens=max_tokens):
                buffer += chunk
                # extract complete sentences from buffer
                while True:
                    m = sentence_end_re.search(buffer)
                    if not m:
                        break
                    sent = m.group(1).strip()
                    buffer = buffer[m.end():]
                    if not sent:
                        continue
                    # clean and translate the sentence back to the user's language
                    sent_clean = _clean_generation(sent)
                    if not sent_clean:
                        continue
                    translated = translate(sent_clean, "en", src_lang)
                    payload = {"type": "sentence", "english": sent_clean, "translated": translated}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            # flush remaining buffer
            if buffer.strip():
                sent = buffer.strip()
                sent_clean = _clean_generation(sent)
                if sent_clean:
                    translated = translate(sent_clean, "en", src_lang)
                    payload = {"type": "sentence", "english": sent_clean, "translated": translated}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
        except Exception as e:
            err = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    headers = {"Content-Type": "text/event-stream; charset=utf-8", "Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return Response(stream_with_context(event_stream()), headers=headers)

@bp.post("/infer_raw")
def ep_infer_raw():
    body = request.get_json() or {}
    prompt = body.get("prompt")
    max_tokens = int(body.get("max_new_tokens", 128))
    use_rag = body.get("use_rag", True)  # Default to using RAG
    if not prompt:
        return jsonify({"error": "prompt required"}), 400
    
    # Retrieve RAG documents if enabled
    rag_docs = []
    if use_rag:
        rag_docs = rag_retrieve(prompt, top_k=3)
    
    # Build prompt with RAG context if available
    if rag_docs:
        rag_block = "\n\n".join(f"Document {i+1}: {d}" for i, d in enumerate(rag_docs))
        final_prompt = f"Relevant context:\n{rag_block}\n\nUser question:\n{prompt}\nAnswer the question clearly and factually. Return just the answer with no further explanantion is as simple words as possible."
    else:
        final_prompt = prompt
    
    output = llm_generate(final_prompt, max_new_tokens=max_tokens)
    return jsonify({
        "prompt": prompt,
        "rag_used": rag_docs,
        "final_prompt": final_prompt,
        "output": output
    })

@bp.get("/health")
def ep_health():
    return jsonify({"status": "alive"})

@bp.post("/rag/add")
def ep_rag_add():
    body = request.get_json() or {}
    text = body.get("text")
    if not text:
        return jsonify({"error": "text required"}), 400
    doc_id = rag_add(text)
    return jsonify({"ok": True, "id": doc_id})

@bp.post("/rag/remove")
def ep_rag_remove():
    body = request.get_json() or {}
    doc_id = body.get("id")
    if not doc_id:
        return jsonify({"error": "id required"}), 400
    if not rag_remove(doc_id):
        return jsonify({"error": "invalid id"}), 400
    return jsonify({"ok": True})

@bp.get("/rag/list")
def ep_rag_list():
    """List all RAG documents with their IDs."""
    docs = rag_list()
    return jsonify({"ok": True, "documents": docs, "count": len(docs)})

@bp.post("/rag/clear")
def ep_rag_clear():
    """Clear all RAG documents."""
    rag_clear()
    return jsonify({"ok": True, "message": "All RAG documents cleared"})

@bp.post("/rag/search")
def ep_rag_search():
    """Search RAG documents by query and return top-k matches."""
    body = request.get_json() or {}
    query = body.get("query")
    top_k = int(body.get("top_k", 3))
    similarity_threshold = float(body.get("similarity_threshold", 0.35))
    
    if not query:
        return jsonify({"error": "query required"}), 400
    
    results = rag_retrieve(query, top_k=top_k, similarity_threshold=similarity_threshold)
    return jsonify({
        "ok": True,
        "query": query,
        "top_k": top_k,
        "similarity_threshold": similarity_threshold,
        "results": results,
        "count": len(results)
    })

@bp.post("/unload_llm")
def ep_unload_llm():
    """Stop the background llama-server."""
    unload_llm()
    translator_unloaded = unload_translator()
    return jsonify({
        "ok": True,
        "message": "LLM server stopped and translator unloaded",
        "translator_unloaded": bool(translator_unloaded)
    })

@bp.get("/benchmark")
def ep_benchmark():
    text = request.args.get("text", "കേരളത്തിൽ മഴ കനത്തിരിക്കുന്നു.")
    lang = request.args.get("lang", "ml")

    lang = LANG_ALIASES.get(lang, lang)

    if lang not in LANG_MAP:
        return jsonify({"error": f"Unsupported language {lang}"}), 400

    src_lang, en_lang = LANG_MAP[lang]

    try:
        results = benchmark_pipeline(
            test_text=text,
            src_lang=src_lang,
            tgt_lang=en_lang,
            max_tokens=64,
        )
        return jsonify({"ok": True, "results": results})
    except RuntimeError as e:
        if "torchvision" in str(e) or "nms" in str(e):
            return jsonify({
                "error": "translator initialization failed due to dependency conflict",
                "details": str(e),
                "suggestion": "Try: pip install --upgrade transformers torch torchvision"
            }), 503
        raise


@bp.post("/benchmark/resource")
def ep_benchmark_resource():
    """
    Comprehensive resource usage benchmark endpoint.
    
    Request body:
    {
        "llm_name": "Qwen2-500M-Instruct-GGUF",
        "prompts": [
            {"text": "भारत का राष्ट्रपति कौन है?", "lang": "hi"},
            {"text": "What is machine learning?", "lang": "en"}
        ],
        "rag_data": ["AI is artificial intelligence", "ML is machine learning"],  // optional
        "n_ctx": 4096,         // optional, default 4096
        "n_gpu_layers": -1,    // optional, default -1 (all)
        "max_tokens": 128      // optional, default 128
    }
    
    Returns comprehensive metrics for:
    - Baseline (unloaded)
    - LLM loaded
    - Translator loaded
    - RAG data loaded
    - Per-prompt full pipeline (translate→infer→translate back) with time, RAM, CPU, VRAM
    """
    body = request.get_json() or {}
    
    llm_name = body.get("llm_name")
    prompts = body.get("prompts", [])
    rag_data = body.get("rag_data", None)
    n_ctx = int(body.get("n_ctx", 4096))
    n_gpu_layers = int(body.get("n_gpu_layers", -1))
    max_tokens = int(body.get("max_tokens", 128))
    
    if not llm_name:
        return jsonify({"error": "llm_name required"}), 400
    if not prompts or not isinstance(prompts, list):
        return jsonify({"error": "prompts must be a non-empty list"}), 400
    
    try:
        results = benchmark_resource_usage(
            llm_name=llm_name,
            prompts=prompts,
            rag_data=rag_data,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            max_tokens=max_tokens
        )
        return jsonify({"ok": True, "results": results})
    except Exception as e:
        return jsonify({
            "error": "benchmark failed",
            "details": str(e)
        }), 500


@bp.post("/llm_metrics")
def ep_llm_metrics():
    """
    Measure detailed LLM performance metrics.
    
    Request body:
    {
        "llm_name": "Qwen2-500M-Instruct-GGUF",
        "n_ctx": 4096,         // optional, default 4096
        "n_gpu_layers": -1,    // optional, default -1 (all)
        "max_tokens": 128      // optional, default 128
    }
    
    Returns:
    {
        "ok": true,
        "model_size_gb": 0.5,
        "load_time_s": 2.3,
        "first_token_latency_ms": 45.2,
        "tokens_per_second": 15.6,
        "output_length_tokens": 128,
        "output_text": "...",
        "total_inference_time_s": 8.2,
        "memory": {
            "baseline_rss_mb": 450.2,
            "loaded_rss_mb": 950.8,
            "peak_rss_mb": 1024.5,
            "load_increase_mb": 500.6,
            "inference_increase_mb": 73.7
        },
        "vram": {
            "baseline_used_mb": 0,
            "loaded_used_mb": 480.5,
            "peak_used_mb": 520.3,
            "total_mb": 8192
        },
        "config": {
            "llm_name": "...",
            "n_ctx": 4096,
            "n_gpu_layers": -1,
            "max_tokens": 128,
            "demo_prompt": "..."
        }
    }
    """
    body = request.get_json() or {}
    
    llm_name = body.get("llm_name")
    n_ctx = int(body.get("n_ctx", 4096))
    n_gpu_layers = int(body.get("n_gpu_layers", -1))
    max_tokens = int(body.get("max_tokens", 128))
    
    if not llm_name:
        return jsonify({"error": "llm_name required"}), 400
    
    try:
        results = benchmark_llm_metrics(
            llm_name=llm_name,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            max_tokens=max_tokens
        )
        
        if "error" in results:
            return jsonify(results), 400
        
        return jsonify({"ok": True, **results})
    except Exception as e:
        return jsonify({
            "error": "benchmark failed",
            "details": str(e)
        }), 500
