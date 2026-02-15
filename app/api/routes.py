from flask import Blueprint, request, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename

import json
import re
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


def _clean_generation(text: str) -> str:
    """Remove code fences and noisy prefixes from model output."""
    cleaned = text
    cleaned = re.sub(r"```.*?```", " ", cleaned, flags=re.S)  # drop fenced blocks
    cleaned = re.sub(r"`+", "", cleaned)  # drop stray backticks
    cleaned = re.sub(r"^\s*Answer\s*:\s*", "", cleaned, flags=re.I)  # drop leading Answer:
    return cleaned.strip()


bp = Blueprint("api", __name__)

# Initialize query cache (lazy-loaded on first access)
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

        p = psutil.Process(os.getpid())
        mem_info = p.memory_info()

        return jsonify(
            {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "ram_used_mb": round(mem_info.rss / (1024 * 1024), 2),
            }
        )
    except Exception as e:
        return jsonify({"error": "failed to fetch metrics", "details": str(e)}), 500


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
    body = request.get_json() or {}
    text = body.get("text")
    lang = body.get("lang")
    max_tokens = int(body.get("max_new_tokens", 128))

    if not text:
        return jsonify({"error": "text required"}), 400

    if not lang or lang not in LANG_MAP:
        return jsonify({"error": f"unsupported lang: {lang}"}), 400

    src_lang, _ = LANG_MAP[lang]

    try:
        english_text = translate(text, src_lang, "en")
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

    stream = bool(body.get("stream", True))

    # Query caching: check if similar query exists and reuse RAG docs
    cache_hit = False
    cache_similarity = None
    qcache = get_query_cache()
    
    if qcache is not None:
        try:
            embed_model = get_embed_model()
            query_embedding = embed_model.encode([english_text])[0].tolist()
            
            cached_result = qcache.find_similar_query(query_embedding)
            if cached_result is not None:
                rag_docs, cache_similarity = cached_result
                cache_hit = True
                print(f"[Query Cache] Hit! Similarity={cache_similarity:.3f}")
        except Exception as e:
            print(f"[Query Cache] Warning: Cache lookup failed: {e}")
    
    # If no cache hit, retrieve from RAG
    if not cache_hit:
        rag_docs = rag_retrieve(english_text, top_k=3)
        
        # Add to cache for future queries
        if qcache is not None:
            try:
                embed_model = get_embed_model()
                query_embedding = embed_model.encode([english_text])[0].tolist()
                qcache.add_query(english_text, query_embedding, rag_docs)
            except Exception as e:
                print(f"[Query Cache] Warning: Failed to cache query: {e}")
    
    context = ""
    if rag_docs:
        context = "\n".join(f"Document {i+1}: {d}" for i, d in enumerate(rag_docs))

    context_block = f"Relevant context:\n{context}\n" if context else ""
    final_prompt = (
        f"User question:\n{english_text}\n\n"
        f"{context_block}"
        "Answer clearly in simple English. Do not use code fences or Markdown. Respond with plain text only."
    )

    if not stream:
        llm_output_en = llm_generate(final_prompt, max_new_tokens=max_tokens)
        llm_output_en = _clean_generation(llm_output_en)
        answer_native = translate(llm_output_en, "en", src_lang)

        return jsonify(
            {
                "input": text,
                "english_in": english_text,
                "rag_used": rag_docs,
                "cache_hit": cache_hit,
                "cache_similarity": cache_similarity,
                "llm_prompt": final_prompt,
                "llm_output_en": llm_output_en,
                "final_output": answer_native,
            }
        )

    sentence_end_re = re.compile(r"(.+?[.!?](?:\"|'|”)?)(\s+|$)", re.S)

    def event_stream():
        meta = {
            "type": "meta",
            "english_in": english_text,
            "cache_hit": cache_hit,
            "cache_similarity": cache_similarity,
            "prompt": final_prompt,
        }
        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

        buffer = ""
        try:
            for chunk in llm_generate_stream(final_prompt, max_new_tokens=max_tokens):
                buffer += chunk

                while True:
                    m = sentence_end_re.search(buffer)
                    if not m:
                        break
                    sent = m.group(1).strip()
                    buffer = buffer[m.end() :]

                    if not sent:
                        continue

                    sent_clean = _clean_generation(sent)
                    if not sent_clean:
                        continue

                    translated = translate(sent_clean, "en", src_lang)
                    payload = {"type": "sentence", "english": sent_clean, "translated": translated}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            if buffer.strip():
                sent_clean = _clean_generation(buffer.strip())
                if sent_clean:
                    translated = translate(sent_clean, "en", src_lang)
                    payload = {"type": "sentence", "english": sent_clean, "translated": translated}
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


@bp.post("/infer_raw")
def ep_infer_raw():
    body = request.get_json() or {}
    prompt = body.get("prompt")
    max_tokens = int(body.get("max_new_tokens", 128))
    use_rag = body.get("use_rag", True)

    if not prompt:
        return jsonify({"error": "prompt required"}), 400

    rag_docs = rag_retrieve(prompt, top_k=3) if use_rag else []

    if rag_docs:
        rag_block = "\n\n".join(f"Document {i+1}: {d}" for i, d in enumerate(rag_docs))
        final_prompt = (
            f"Relevant context:\n{rag_block}\n\n"
            f"User question:\n{prompt}\n"
            "Answer the question clearly and factually. Return just the answer with no further explanation in simple words."
        )
    else:
        final_prompt = prompt

    output = llm_generate(final_prompt, max_new_tokens=max_tokens)
    return jsonify({"prompt": prompt, "rag_used": rag_docs, "final_prompt": final_prompt, "output": output})


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
