from flask import Blueprint, request, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename
from app.services.translator_service import translate, detect_supported_language

import json
import re
import time
from pathlib import Path

from app.config import (
    LANG_MAP,
    QUERY_CACHE_FILE,
    QUERY_CACHE_SIMILARITY_THRESHOLD,
    QUERY_CACHE_MAX_ENTRIES,
    QUERY_CACHE_ENABLED,
)
from app.services.llm_service import (
    download_gguf,
    load_llm_from_gguf,
    llm_generate,
    llm_generate_stream,
    unload_llm,
    current_name,
    SERVER_URL,
    list_all_llms,
)
from app.services.translator_service import translate
from app.services.translator_service import detect_supported_language

from app.services.rag_service import (
    rag_add,
    rag_remove,
    rag_retrieve,
    rag_list,
    rag_clear,
    add_pdf_to_rag,
    get_embed_model,
)
from app.services.rag_backend import (
    available_backends,
    load_backend,
    get_active_backend_name,
)
from app.services.query_cache_service import QueryCache
from app.services.benchmark_service import (
    benchmark_pipeline,
    benchmark_resource_usage,
    benchmark_rag_metrics,
)


from app.services.rag_service import get_embed_model
import numpy as np

# Create cache ONCE globally
# query_cache = QueryCache(
#     cache_file=QUERY_CACHE_FILE,
#     similarity_threshold=0.70   # Lowered for demo reliability
# )
import os
import tempfile
from werkzeug.utils import secure_filename
import psutil
from app.config import LANG_MAP, LANG_ALIASES, NLLB_LANG_MAP, USE_ONNX_TRANSLATOR, ONNX_LANG_MAP
from app.services.cache_service import model_cache
from app.services.llm_service import download_gguf, load_llm_from_gguf, llm_generate, llm_generate_stream, unload_llm, get_current_name, SERVER_URL
from app.services.translator_service import translate, detect_supported_language, unload_translator, preload_translator
from app.services.rag_service import rag_add, rag_remove, rag_retrieve, rag_list, rag_clear, add_pdf_to_rag
from app.services.benchmark_service import benchmark_pipeline, benchmark_resource_usage, benchmark_llm_metrics, benchmark_translator_metrics, benchmark_rag_metrics
from app.services.onnx_translator_service import translate_onnx, get_onnx_status, unload_onnx_translator, preload_onnx_translator


def _clean_generation(text: str) -> str:
    """Remove code fences and noisy prefixes from model output."""
    cleaned = text
    cleaned = re.sub(r"```.*?```", " ", cleaned, flags=re.S)  # drop fenced blocks
    cleaned = re.sub(r"`+", "", cleaned)  # drop stray backticks
    cleaned = re.sub(r"^\s*Answer\s*:\s*", "", cleaned, flags=re.I)  # drop leading Answer:
    return cleaned.strip()


bp = Blueprint("api", __name__)

# Query cache global instance
query_cache = None


def get_query_cache():
    """Lazy-load query cache on first access."""
    global query_cache
    if query_cache is None and QUERY_CACHE_ENABLED:
        query_cache = QueryCache(
            cache_file=QUERY_CACHE_FILE,
            similarity_threshold=QUERY_CACHE_SIMILARITY_THRESHOLD,
            max_entries=QUERY_CACHE_MAX_ENTRIES,
        )
    return query_cache


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

@bp.post("/translator_metrics")
def ep_translator_metrics():
    body = request.get_json() or {}
    src_lang = body.get("src_lang", "hi")
    tgt_lang = body.get("tgt_lang", "en")

    test_text = "नमस्ते दुनिया"

    try:
        import time

        start = time.time()
        forward = translate(test_text, src_lang, tgt_lang)
        mid = time.time()
        roundtrip = translate(forward, tgt_lang, src_lang)
        end = time.time()

        return jsonify({
            "ok": True,
            "input": {
                "text": test_text,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "char_length": len(test_text),
                "token_length_estimate": len(test_text.split())
            },
            "outputs": {
                "forward_translation": forward,
                "roundtrip_translation": roundtrip
            },
            "end_to_end_time_s": end - start,
            "throughput": {
                "forward": {
                    "time_s": mid - start,
                    "tokens_per_sec": 0,
                    "chars_per_sec": 0
                },
                "roundtrip": {
                    "time_s": end - mid,
                    "tokens_per_sec": 0,
                    "chars_per_sec": 0
                }
            },
            "memory": {
                "baseline_rss_mb": 0,
                "peak_rss_mb": 0,
                "peak_increase_mb": 0,
                "after_forward_rss_mb": 0,
                "translation_increase_mb": 0
            },
            "vram": {
                "total_mb": 0,
                "baseline_used_mb": 0,
                "peak_used_mb": 0,
                "after_forward_used_mb": 0
            },
            "quality": {
                "bleu_score": 0,
                "chrf_score": 0,
                "char_length_similarity_pct": 100,
                "forward_output_tokens": len(forward.split()),
                "forward_output_chars": len(forward),
                "roundtrip_output_tokens": len(roundtrip.split()),
                "roundtrip_output_chars": len(roundtrip)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.get("/translator_status")
def ep_translator_status():
    """Get status of available translators (NLLB and ONNX)."""
    onnx_status = get_onnx_status()
    return jsonify({
        "active_translator": "onnx" if USE_ONNX_TRANSLATOR and onnx_status["available"] else "nllb",
        "onnx": onnx_status,
        "nllb": {
            "available": True,
            "model": "facebook/nllb-200-distilled-600M"
        }
    })

@bp.post("/toggle_translator")
def ep_toggle_translator():
    """Toggle between ONNX and NLLB translator."""
    body = request.get_json() or {}
    use_onnx = body.get("use_onnx", not USE_ONNX_TRANSLATOR)
    
    onnx_status = get_onnx_status()
    if use_onnx and not onnx_status["available"]:
        return jsonify({
            "error": "ONNX models not available",
            "details": onnx_status
        }), 503
    
    if use_onnx:
        print("[API] Switching to ONNX translator")
        unload_translator()  # Clean up NLLB if loaded
    else:
        print("[API] Switching to NLLB translator")
        unload_onnx_translator()  # Clean up ONNX if loaded
    
    return jsonify({
        "ok": True,
        "active_translator": "onnx" if use_onnx else "nllb"
    })


@bp.post("/translator_preload")
def ep_translator_preload():
    """Preload translator models without running a translation."""
    body = request.get_json() or {}
    use_onnx = body.get("use_onnx", USE_ONNX_TRANSLATOR)

    if use_onnx:
        onnx_status = get_onnx_status()
        if not onnx_status["available"]:
            return jsonify({
                "error": "ONNX models not available",
                "details": onnx_status
            }), 503
        details = preload_onnx_translator()
    else:
        details = preload_translator()

    return jsonify({
        "ok": True,
        "active_translator": "onnx" if use_onnx else "nllb",
        "details": details,
    })

@bp.post("/translate")
def ep_translate():
    body = request.get_json() or {}
    text = body.get("text")
    target = (body.get("target") or "en").lower()
    stream = bool(body.get("stream", True))
    max_tokens = int(body.get("max_new_tokens", 256))
    use_onnx = body.get("use_onnx", USE_ONNX_TRANSLATOR)

    if not text:
        return jsonify({"error": "text required"}), 400

    # Auto detection
    if src_lang == "auto":
        detected = detect_supported_language(text)
        if not detected:
            return jsonify({"error": "Could not detect language"}), 400
        src_lang = detected

    try:
        translated = translate(text, src_lang, tgt_lang)
    # Auto-detect source language (must be in LANG_CONF)
    src_lang_key = detect_supported_language(text)
    if not src_lang_key:
        return jsonify({"error": "could not auto-detect a supported language"}), 400

    # Determine which language map and translation function to use
    if use_onnx:
        target_map = ONNX_LANG_MAP
        translate_fn = translate_onnx
        backend = "onnx"
        # For ONNX, use short codes (e.g., "hi" instead of "hin_Deva")
        src_code = src_lang_key
        
        # Check if ONNX supports the detected language
        if src_lang_key not in ONNX_LANG_MAP:
            supported_langs = ", ".join(sorted(ONNX_LANG_MAP.keys()))
            return jsonify({
                "error": f"Language '{src_lang_key}' not supported by ONNX model",
                "details": f"ONNX supports: {supported_langs}",
                "suggestion": "Switch to NLLB translator or use a supported language"
            }), 400
    else:
        target_map = NLLB_LANG_MAP
        translate_fn = translate
        backend = "nllb"
        # For NLLB, use long codes from LANG_MAP (e.g., "hin_Deva")
        src_code, _ = LANG_MAP[src_lang_key]

    # Normalize target and validate it against known mappings or raw NLLB codes
    target_key = LANG_ALIASES.get(target, target)
    target_code = target_map.get(target_key, target_key)
    if target_key not in LANG_MAP and target_key not in target_map and "_" not in target_key:
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
            try:
                translated = translate_fn(sent, src_code, target_code, max_tokens)
            except Exception as e:
                return jsonify({
                    "error": f"{backend} translation failed",
                    "details": str(e)
                }), 503
            translated_sentences.append({
                "source": sent,
                "translated": translated,
            })

        combined = " ".join(item["translated"] for item in translated_sentences)
        return jsonify({
            "ok": True,
            "input": text,
            "detected_lang": src_lang_key,
            "target_lang": target_key,
            "translated_text": combined,
            "sentences": translated_sentences,
            "backend": backend,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.get("/list_llms")
def ep_list_llms():
    return jsonify(
        {
            "downloaded_llms": list_all_llms(),
            "loaded_llm": current_name,
            "server_url": SERVER_URL,
        }
    )


@bp.get("/current_llm")
def ep_current_llm():
    return jsonify({"loaded_llm": current_name})


@bp.get("/system/metrics")
def ep_system_metrics():
    """
    Returns basic system metrics for frontend dashboard.
    """
    try:
        import os
        import psutil
        import time

        process = psutil.Process(os.getpid())

        # Prime CPU measurement
        process.cpu_percent(None)
        time.sleep(0.1)
        cpu = process.cpu_percent(None)

        mem_info = process.memory_info()

        return jsonify({
            "cpu_percent": cpu,
            "ram_used_mb": round(mem_info.rss / (1024 * 1024), 2),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@bp.post("/llm_metrics")
def ep_llm_metrics():
    body = request.get_json() or {}
    llm_name = body.get("llm_name") or body.get("name")  # accept both

    if not llm_name:
        return jsonify({"error": "llm_name required"}), 400

    n_ctx = int(body.get("ctx_size", body.get("n_ctx", 4096)))
    n_gpu_layers = int(body.get("n_gpu_layers", -1))
    max_tokens = int(body.get("max_new_tokens", body.get("max_tokens", 128)))

    try:
        from app.services.benchmark_service import benchmark_llm_metrics

        results = benchmark_llm_metrics(
            llm_name=llm_name,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            max_tokens=max_tokens,
        )
        return jsonify({"ok": True, "results": results})

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@bp.post("/load_llm")
def ep_load_llm():
    body = request.get_json() or {}
    name = body.get("name")
    ctx_size = int(body.get("ctx_size", 4096))
    n_gpu_layers = int(body.get("n_gpu_layers", -1))

    if not name:
        return jsonify({"error": "name required"}), 400

    load_llm_from_gguf(name, n_ctx=ctx_size, n_gpu_layers=n_gpu_layers)
    return jsonify({"ok": True, "loaded": name, "server_url": SERVER_URL})


@bp.post("/infer")
def ep_infer():
    import time

    start_total = time.perf_counter()

    body = request.get_json() or {}
    text = body.get("text")
    lang = body.get("lang")
    max_tokens = int(body.get("max_new_tokens", 128))

    if not text:
        return jsonify({"error": "text required"}), 400

    # Handle auto-detection
    if lang == "auto":
        detected = detect_supported_language(text)
        if not detected:
            return jsonify({"error": "Could not detect language"}), 400
        lang = detected

    if not lang or lang not in LANG_MAP:
        return jsonify({"error": f"unsupported lang: {lang}"}), 400

    src_lang, _ = LANG_MAP[lang]

    # ---------------------------
    # INPUT TRANSLATION
    # ---------------------------
    t0 = time.perf_counter()
    english_text = translate(text, src_lang, "en")
    translation_in_time = time.perf_counter() - t0

    stream = bool(body.get("stream", True))

    # ---------------------------
    # EMBEDDING + CACHE
    # ---------------------------
    cache_hit = False
    cache_similarity = None
    embedding_time = 0.0
    rag_time = 0.0

    qcache = get_query_cache()
    embed_model = get_embed_model()

    t0 = time.perf_counter()
    query_embedding = embed_model.encode([english_text])[0].tolist()
    embedding_time = time.perf_counter() - t0

    if qcache is not None:
        cached_result = qcache.find_similar_query(query_embedding)
        if cached_result is not None:
            rag_docs, cache_similarity = cached_result
            cache_hit = True
        else:
            rag_docs = []
    else:
        rag_docs = []

    # ---------------------------
    # RAG RETRIEVAL
    # ---------------------------
    if not cache_hit:
        t0 = time.perf_counter()
        rag_docs = rag_retrieve(english_text, top_k=3)
        # ---- RELEVANCE GATE ----
        CONFIDENCE_THRESHOLD = 0.35

        if not rag_docs or max(d.get("similarity", 0) for d in rag_docs) < CONFIDENCE_THRESHOLD:

            refusal_en = "Sorry, the question is beyond the scope of the uploaded knowledge base."
            refusal_native = translate(refusal_en, "en", src_lang)

            if not stream:
                return jsonify({
                    "input": text,
                    "english_in": english_text,
                    "rag_used": [],
                    "final_output": refusal_native,
                    "metrics": {
                        "embedding_time_sec": embedding_time,
                        "rag_time_sec": 0,
                        "llm_time_sec": 0,
                        "translation_in_time_sec": translation_in_time,
                        "translation_out_time_sec": 0,
                        "total_time_sec": time.perf_counter() - start_total
                    }
                })

            # 🔥 STREAM MODE REFUSAL
            def refusal_stream():
                meta = {"type": "meta"}
                yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

                sentence = {
                    "type": "sentence",
                    "english": refusal_en,
                    "translated": refusal_native
                }
                yield f"data: {json.dumps(sentence, ensure_ascii=False)}\n\n"

                done = {"type": "done"}
                yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"

            headers = {
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }

            return Response(stream_with_context(refusal_stream()), headers=headers)



        rag_time = time.perf_counter() - t0

        if qcache is not None:
            qcache.add_query(english_text, query_embedding, rag_docs)

    # ---------------------------
    # PROMPT BUILD
    # ---------------------------
    context = ""
    if rag_docs:
        context = "\n".join(f"Document {i+1}: {d}" for i, d in enumerate(rag_docs))

    context_block = f"Relevant context:\n{context}\n" if context else ""

    final_prompt = (
        f"User question:\n{english_text}\n\n"
        f"{context_block}"
        "Answer clearly in simple English. Do not use code fences or Markdown. Respond with plain text only."
    )

    # ===========================
    # NON-STREAM MODE
    # ===========================
    if not stream:

        t0 = time.perf_counter()
        llm_output_en = llm_generate(final_prompt, max_new_tokens=max_tokens)
        llm_time = time.perf_counter() - t0

        llm_output_en = _clean_generation(llm_output_en)

        t0 = time.perf_counter()
        answer_native = translate(llm_output_en, "en", src_lang)
        translation_out_time = time.perf_counter() - t0

        total_time = time.perf_counter() - start_total

        return jsonify({
            "input": text,
            "english_in": english_text,
            "rag_used": rag_docs,
            "cache_hit": cache_hit,
            "cache_similarity": cache_similarity,
            "llm_prompt": final_prompt,
            "llm_output_en": llm_output_en,
            "final_output": answer_native,
            "metrics": {
                "embedding_time_sec": embedding_time,
                "rag_time_sec": rag_time,
                "llm_time_sec": llm_time,
                "translation_in_time_sec": translation_in_time,
                "translation_out_time_sec": translation_out_time,
                "total_time_sec": total_time
            }
        })

    # ===========================
    # STREAM MODE
    # ===========================

    sentence_end_re = re.compile(r"(.+?[.!?](?:\"|'|”)?)(\s+|$)", re.S)

    def event_stream():
        stream_start_total = time.perf_counter()

        meta = {
            "type": "meta",
            "english_in": english_text,
            "cache_hit": cache_hit,
            "cache_similarity": cache_similarity,
            "prompt": final_prompt,
            "metrics": {
                "embedding_time_sec": embedding_time,
                "rag_time_sec": rag_time,
                "translation_in_time_sec": translation_in_time,
            }
        }

        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

        buffer = ""
        llm_start = time.perf_counter()
        translation_out_time = 0.0

        try:
            for chunk in llm_generate_stream(final_prompt, max_new_tokens=max_tokens):
                buffer += chunk

                while True:
                    m = sentence_end_re.search(buffer)
                    if not m:
                        break

                    sent = m.group(1).strip()
                    buffer = buffer[m.end():]

                    sent_clean = _clean_generation(sent)
                    if not sent_clean:
                        continue

                    t0 = time.perf_counter()
                    translated = translate(sent_clean, "en", src_lang)
                    translation_out_time += time.perf_counter() - t0

                    payload = {
                        "type": "sentence",
                        "english": sent_clean,
                        "translated": translated
                    }

                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            llm_time = time.perf_counter() - llm_start
            total_time = time.perf_counter() - stream_start_total

            # ---- Send metrics event ----
            metrics_payload = {
                "type": "metrics",
                "llm_time_sec": llm_time,
                "translation_out_time_sec": translation_out_time,
                "total_time_sec": total_time,
            }

            yield "data: " + json.dumps(metrics_payload, ensure_ascii=False) + "\n\n"

            # ---- Done event ----
            done_payload = {"type": "done"}
            yield "data: " + json.dumps(done_payload, ensure_ascii=False) + "\n\n"

            # ---- Error handling ----
        except Exception as e:
            error_payload = {
                "type": "error",
                "message": str(e),
            }
            yield "data: " + json.dumps(error_payload, ensure_ascii=False) + "\n\n"


    headers = {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }

    return Response(stream_with_context(event_stream()), headers=headers)


@bp.post("/infer_raw")
def ep_infer_raw():
    import time

    start_total = time.perf_counter()

    body = request.get_json() or {}
    prompt = body.get("prompt")
    max_tokens = int(body.get("max_new_tokens", 128))
    use_rag = body.get("use_rag", True)

    if not prompt:
        return jsonify({"error": "prompt required"}), 400

    # ---------------------------
    # 1️⃣ Embed query properly
    # ---------------------------
    embed_model = get_embed_model()

    t0 = time.perf_counter()
    q_emb = embed_model.encode([prompt])[0].tolist()
    embedding_time = time.perf_counter() - t0

    # ---------------------------
    # 2️⃣ Check cache
    # ---------------------------
    qcache = get_query_cache()
    cached = None
    similarity = None

    if qcache is not None:
        cached = qcache.find_similar_query(q_emb)

    if cached:
        rag_docs, similarity = cached
        print(f"[Query Routing] CACHE HIT (similarity={similarity:.3f})")

        if rag_docs:
            rag_block = "\n\n".join(
                f"Document {i+1}: {d['text'] if isinstance(d, dict) else d}"
                for i, d in enumerate(rag_docs)
            )
            final_prompt = (
                f"Relevant context:\n{rag_block}\n\n"
                f"User question:\n{prompt}\n"
                "Answer clearly in simple words."
            )
        else:
            final_prompt = prompt

        # ---------- LLM ----------
        t0 = time.perf_counter()
        output = llm_generate(final_prompt, max_new_tokens=max_tokens)
        llm_time = time.perf_counter() - t0

        total_time = time.perf_counter() - start_total

        return jsonify({
            "prompt": prompt,
            "rag_used": rag_docs,
            "final_prompt": final_prompt,
            "output": output,
            "cache_hit": True,
            "cache_similarity": similarity,
            "metrics": {
                "embedding_time_sec": embedding_time,
                "retrieval_time_sec": 0.0,
                "rag_total_time_sec": 0.0,
                "llm_time_sec": llm_time,
                "total_time_sec": total_time
            }
        })

    # ---------------------------
    # 3️⃣ Fresh RAG retrieval
    # ---------------------------
    print("[Query Routing] CACHE MISS → Performing retrieval")

    t0 = time.perf_counter()
    rag_docs = rag_retrieve(prompt, top_k=3) if use_rag else []
    retrieval_time = time.perf_counter() - t0

    if rag_docs:
        rag_block = "\n\n".join(
            f"Document {i+1}: {d['text'] if isinstance(d, dict) else d}"
            for i, d in enumerate(rag_docs)
        )
        final_prompt = (
            f"Relevant context:\n{rag_block}\n\n"
            f"User question:\n{prompt}\n"
            "Answer clearly in simple words."
        )
    else:
        final_prompt = prompt

    # ---------- LLM ----------
    t0 = time.perf_counter()
    output = llm_generate(final_prompt, max_new_tokens=max_tokens)
    llm_time = time.perf_counter() - t0

    # ---------------------------
    # 4️⃣ Store in cache
    # ---------------------------
    if qcache is not None:
        qcache.add_query(prompt, q_emb, rag_docs)

    total_time = time.perf_counter() - start_total
    rag_total_time = embedding_time + retrieval_time

    return jsonify({
        "prompt": prompt,
        "rag_used": rag_docs,
        "final_prompt": final_prompt,
        "output": output,
        "cache_hit": False,
        "metrics": {
            "embedding_time_sec": embedding_time,
            "retrieval_time_sec": retrieval_time,
            "rag_total_time_sec": rag_total_time,
            "llm_time_sec": llm_time,
            "total_time_sec": total_time
        }
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
    docs = rag_list()
    return jsonify({"ok": True, "documents": docs, "count": len(docs)})

@bp.post("/translator_metrics v2")
def ep_translator_metricsv2():
    try:
        from app.services.translator_service import translate

        sample = "नमस्ते दुनिया"
        output = translate(sample, "hi", "en")

        return jsonify({
            "ok": True,
            "input": sample,
            "output": output,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@bp.post("/rag/search")
def ep_rag_search():
    body = request.get_json() or {}
    query = body.get("query")
    top_k = body.get("top_k", None)
    similarity_threshold = body.get("similarity_threshold", None)

    if not query:
        return jsonify({"error": "query required"}), 400

    try:
        if top_k is not None:
            top_k = int(top_k)
        if similarity_threshold is not None:
            similarity_threshold = float(similarity_threshold)

        results = rag_retrieve(query=query, top_k=top_k, similarity_threshold=similarity_threshold)
        return jsonify({"ok": True, "results": results, "count": len(results)})

    except Exception as e:
        return jsonify({"error": "rag search failed", "details": str(e)}), 500


@bp.post("/rag_metrics")
def ep_rag_metrics():
    body = request.get_json() or {}
    llm_name = body.get("llm_name", None)

    try:
        results = benchmark_rag_metrics(llm_name=llm_name)
        return jsonify({"ok": True, **results})
    except Exception as e:
        return jsonify({"ok": False, "error": "rag_metrics failed", "details": str(e)}), 500


@bp.get("/rag/backends")
def ep_rag_backends():
    try:
        return jsonify(
            {
                "ok": True,
                "available": available_backends(),
                "active": get_active_backend_name(),
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@bp.post("/rag/backend/load")
def ep_rag_backend_load():
    body = request.get_json() or {}
    name = body.get("name")

    if not name:
        return jsonify({"error": "name required"}), 400

    try:
        load_backend(name)
        return jsonify(
            {
                "ok": True,
                "active": get_active_backend_name(),
                "available": available_backends(),
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": "failed to load backend", "details": str(e)}), 500


@bp.post("/rag/pdf/add")
def ep_rag_pdf_add():
    body = request.get_json() or {}
    pdf_path = body.get("pdf_path")

    if not pdf_path:
        return jsonify({"error": "pdf_path required"}), 400

    try:
        out = add_pdf_to_rag(pdf_path)
        return jsonify({"ok": True, **out})
    except Exception as e:
        return jsonify({"ok": False, "error": "pdf ingest failed", "details": str(e)}), 500


@bp.post("/rag/pdf/upload")
def ep_rag_pdf_upload():
    if "file" not in request.files:
        return jsonify({"error": "file required"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "filename missing"}), 400

    filename = secure_filename(f.filename)
    if not filename.lower().endswith(".pdf"):
        return jsonify({"error": "only .pdf supported"}), 400

    try:
        upload_dir = Path("uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = upload_dir / filename
        f.save(str(pdf_path))

        out = add_pdf_to_rag(str(pdf_path))
        return jsonify({"ok": True, "saved_to": str(pdf_path), "result": out})
    except Exception as e:
        return jsonify({"ok": False, "error": "pdf upload failed", "details": str(e)}), 500


@bp.post("/rag/clear")
def ep_rag_clear():
    rag_clear()
    return jsonify({"ok": True, "message": "All RAG documents cleared"})


@bp.get("/query_cache/stats")
def ep_query_cache_stats():
    """Get query cache statistics."""
    qcache = get_query_cache()
    if qcache is None:
        return jsonify({"ok": True, "enabled": False, "message": "Query cache is disabled"}), 200
    
    return jsonify({"ok": True, "enabled": True, **qcache.stats()})


@bp.post("/query_cache/clear")
def ep_query_cache_clear():
    """Clear all cached queries."""
    qcache = get_query_cache()
    if qcache is None:
        return jsonify({"ok": True, "message": "Query cache is disabled"}), 200
    
    qcache.clear()
    return jsonify({"ok": True, "message": "Query cache cleared"})



@bp.post("/unload_llm")
def ep_unload_llm():
    unload_llm()
    return jsonify({"ok": True, "message": "LLM server stopped"})


@bp.get("/benchmark")
def ep_benchmark():
    text = request.args.get("text", "കേരളത്തിൽ മഴ കനത്തിരിക്കുന്നു.")
    lang = request.args.get("lang", "ml")

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
            return (
                jsonify(
                    {
                        "error": "translator initialization failed due to dependency conflict",
                        "details": str(e),
                        "suggestion": "Try: pip install --upgrade transformers torch torchvision",
                    }
                ),
                503,
            )
        raise


@bp.post("/benchmark/resource")
def ep_benchmark_resource():
    """
    Comprehensive resource usage benchmark endpoint.
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
            max_tokens=max_tokens,
        )
        return jsonify({"ok": True, "results": results})
    except Exception as e:
        return jsonify({"error": "benchmark failed", "details": str(e)}), 500
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


@bp.post("/translator_metrics")
def ep_translator_metrics():
    """
    Measure detailed translator performance metrics.
    
    Request body:
    {
        "src_lang": "hi",      // source language code
        "tgt_lang": "en"       // optional, target language (default: "en")
    }
    
    Returns:
    {
        "ok": true,
        "input": {
            "text": "...",
            "char_length": 250,
            "token_length_estimate": 45,
            "src_lang": "hi",
            "tgt_lang": "en"
        },
        "throughput": {
            "forward": {
                "chars_per_sec": 180.5,
                "tokens_per_sec": 32.1,
                "time_s": 1.385
            },
            "roundtrip": {
                "chars_per_sec": 195.2,
                "tokens_per_sec": 35.8,
                "time_s": 1.256
            }
        },
        "quality": {
            "bleu_score": 45.2,
            "chrf_score": 62.8,
            "char_length_similarity_pct": 92.5,
            "forward_output_chars": 245,
            "forward_output_tokens": 42,
            "roundtrip_output_chars": 248,
            "roundtrip_output_tokens": 44
        },
        "memory": {
            "baseline_rss_mb": 450.2,
            "after_forward_rss_mb": 650.8,
            "peak_rss_mb": 680.5,
            "translation_increase_mb": 200.6,
            "peak_increase_mb": 230.3
        },
        "vram": {
            "baseline_used_mb": 0,
            "after_forward_used_mb": 380.5,
            "peak_used_mb": 420.3,
            "total_mb": 8192
        },
        "end_to_end_time_s": 2.641,
        "outputs": {
            "forward_translation": "...",
            "roundtrip_translation": "..."
        }
    }
    """
    body = request.get_json() or {}
    
    src_lang = body.get("src_lang")
    tgt_lang = body.get("tgt_lang", "en")
    use_onnx = body.get("use_onnx", USE_ONNX_TRANSLATOR)
    
    if not src_lang:
        return jsonify({"error": "src_lang required"}), 400
    
    # Normalize language codes
    src_lang = LANG_ALIASES.get(src_lang, src_lang)
    tgt_lang = LANG_ALIASES.get(tgt_lang, tgt_lang)
    
    try:
        results = benchmark_translator_metrics(
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            use_onnx=use_onnx
        )
        
        if "error" in results:
            return jsonify(results), 400
        
        return jsonify({"ok": True, **results})
    except Exception as e:
        return jsonify({
            "error": "benchmark failed",
            "details": str(e)
        }), 500


@bp.post("/rag_metrics")
def ep_rag_metrics():
    """
    Measure detailed RAG performance metrics.
    
    Request body:
    {
        "llm_name": "Qwen2-500M-Instruct-GGUF",  // optional, for RAG impact analysis
        "n_ctx": 4096,                           // optional, default 4096
        "n_gpu_layers": -1                       // optional, default -1 (all)
    }
    
    Returns:
    {
        "ok": true,
        "documents_indexed": 10,
        "indexing_time_s": 0.234,
        "index_size": {
            "index_file_mb": 0.0012,
            "metadata_file_mb": 0.0008,
            "total_mb": 0.0020
        },
        "retrieval_performance": {
            "avg_query_time_ms": 2.5,
            "min_query_time_ms": 1.8,
            "max_query_time_ms": 3.2,
            "topk_avg_times_ms": {
                "1": 1.9,
                "3": 2.5,
                "5": 3.1
            }
        },
        "memory": {
            "baseline_rss_mb": 450.2,
            "after_indexing_rss_mb": 452.8,
            "indexing_increase_mb": 2.6
        },
        "relevance": {
            "avg_recall_at_3": 0.85,
            "queries_evaluated": 4,
            "perfect_recalls": 2
        },
        "rag_impact": {
            "query": "What is quantum computing and how does it work?",
            "answer_without_rag": "...",
            "answer_with_rag": "...",
            "answer_length_diff": 45,
            "inference_time_without_rag_s": 1.2,
            "inference_time_with_rag_s": 1.5,
            "rag_overhead_s": 0.3,
            "contexts_used": 3
        },
        "restoration": {
            "original_doc_count": 5,
            "restored_doc_count": 5
        }
    }
    
    Note: This endpoint temporarily clears RAG data, runs benchmarks with demo data,
    then restores the original RAG documents.
    """
    body = request.get_json() or {}
    
    llm_name = body.get("llm_name")
    n_ctx = int(body.get("n_ctx", 4096))
    n_gpu_layers = int(body.get("n_gpu_layers", -1))
    
    try:
        results = benchmark_rag_metrics(
            llm_name=llm_name,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers
        )
        
        if "error" in results:
            return jsonify(results), 400
        
        return jsonify({"ok": True, **results})
    except Exception as e:
        return jsonify({
            "error": "benchmark failed",
            "details": str(e)
        }), 500
