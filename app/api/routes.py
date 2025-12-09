from flask import Blueprint, request, jsonify, Response, stream_with_context
import json
import re
from app.config import LANG_MAP
from app.services.cache_service import model_cache
from app.services.llm_service import download_gguf, load_llm_from_gguf, llm_generate, llm_generate_stream, unload_llm, current_name, SERVER_URL
from app.services.translator_service import translate
from app.services.rag_service import rag_add, rag_remove, rag_retrieve, rag_list, rag_clear
from app.services.benchmark_service import benchmark_pipeline


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
        "loaded_llm": current_name,
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
            "final_output": answer_native
        })

    # Streaming response (SSE). We will yield JSON payloads per translated sentence.
    sentence_end_re = re.compile(r"(.+?[.!?](?:\"|'|”)?)(\s+|$)", re.S)

    def event_stream():
        # Meta event with initial English input
        meta = {"type": "meta", "english_in": english_text, "prompt": final_prompt}
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

@bp.post("/unload_llm")
def ep_unload_llm():
    """Stop the background llama-server."""
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
            return jsonify({
                "error": "translator initialization failed due to dependency conflict",
                "details": str(e),
                "suggestion": "Try: pip install --upgrade transformers torch torchvision"
            }), 503
        raise