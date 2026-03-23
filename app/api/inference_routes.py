import json
import time
import re
from flask import request, jsonify, Response, stream_with_context
from . import bp
from .utils import _clean_generation, _clean_rag_context, _ensure_complete_sentence
from .translate_routes import _get_effective_active_translator, get_active_onnx_family, _onnx_lang_map_for_family
from app.config import (
    LANG_MAP,
    LANG_ALIASES,
    LLM_DEFAULT_MAX_TOKENS,
    TRANSLATION_MAX_NEW_TOKENS,
    TRANSLATION_MIN_NEW_TOKENS,
    TRANSLATION_SHORT_TEXT_MAX_NEW_TOKENS,
    TRANSLATION_SHORT_TEXT_WORDS,
)
from app.services.llm_service import llm_generate, llm_generate_stream
from app.services.translator_service import translate, detect_supported_language
from app.services.onnx_translator_service import translate_onnx
from app.services.rag_service import rag_retrieve, get_embed_model
from .system_routes import get_query_cache

NO_DB_ANSWER_EN = "No answer found in the database for this question."
MAX_FINAL_SENTENCES = 10


def _sentence_signature(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _is_no_db_sentence(text: str) -> bool:
    cleaned = _clean_generation(text).strip().lower()
    cleaned = cleaned.strip('"\'“”‘’[](){}')
    cleaned = cleaned.rstrip(".!?। ")
    target = NO_DB_ANSWER_EN.lower().rstrip(".!?। ")
    return cleaned == target


def _translation_token_limit_for_text(text: str) -> int:
    words = max(1, len((text or "").split()))
    if words <= TRANSLATION_SHORT_TEXT_WORDS:
        adaptive = min(TRANSLATION_SHORT_TEXT_MAX_NEW_TOKENS, max(TRANSLATION_MIN_NEW_TOKENS, words * 2 + 6))
    else:
        adaptive = max(TRANSLATION_MIN_NEW_TOKENS, min(TRANSLATION_MAX_NEW_TOKENS, words * 2))
    return adaptive


def _build_grounded_prompt(question: str, rag_docs: list) -> str:
    top_docs = rag_docs[:3]
    context_lines = []
    for i, doc in enumerate(top_docs, start=1):
        text = doc.get("text") if isinstance(doc, dict) else str(doc)
        source = doc.get("source", "unknown") if isinstance(doc, dict) else "unknown"
        similarity = doc.get("similarity") if isinstance(doc, dict) else None
        text = _clean_generation(str(text or ""))
        if not text:
            continue
        if isinstance(similarity, (int, float)):
            context_lines.append(f"Document {i} (source: {source}, similarity: {similarity:.3f}): {text}")
        else:
            context_lines.append(f"Document {i} (source: {source}): {text}")

    context_block = "\n".join(context_lines)

    return (
        "You are a precise QA assistant.\n"
        "Use only the provided documents to answer the question.\n"
        "If the documents do not contain enough information, reply exactly with: "
        f"\"{NO_DB_ANSWER_EN}\"\n"
        "Do not ask follow-up questions.\n"
        "Do not include unrelated content, extra tasks, or meta commentary.\n"
        "Write a detailed answer in 6-10 complete, grammatically correct sentences.\n"
        "Avoid repeating the same sentence or idea verbatim.\n"
        "Ensure the final sentence is complete and ends with proper punctuation.\n\n"
        f"Documents:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _finalize_answer_en(text: str) -> str:
    cleaned = _clean_generation(text)
    if not cleaned:
        return NO_DB_ANSWER_EN

    parts = re.split(r"(?<=[.!?।])\s+", cleaned)
    seen_signatures = set()
    kept = []
    for part in parts:
        sentence = part.strip()
        if not sentence or sentence.endswith("?") or _is_no_db_sentence(sentence):
            continue

        signature = _sentence_signature(sentence)
        if len(signature) > 12 and signature in seen_signatures:
            continue

        if signature:
            seen_signatures.add(signature)
        kept.append(sentence)

        if len(kept) >= MAX_FINAL_SENTENCES:
            break

    if not kept:
        return NO_DB_ANSWER_EN

    finalized = " ".join(kept)
    return _ensure_complete_sentence(finalized)

@bp.post("/infer")
def ep_infer():
    body = request.get_json() or {}
    text = body.get("text")
    lang = body.get("lang", "auto").lower()
    requested_max = int(body.get("max_new_tokens", LLM_DEFAULT_MAX_TOKENS))
    max_tokens = max(128, min(requested_max, 900))
    stream = bool(body.get("stream", True))

    if not text:
        return jsonify({"error": "text required"}), 400

    detected_lang = detect_supported_language(text) if lang == "auto" else lang
    lang = LANG_ALIASES.get(detected_lang, detected_lang) or "en"
    
    src_lang, _ = LANG_MAP.get(lang, ("eng_Latn", "en"))
    is_source_english = (src_lang == "eng_Latn")
    
    backend = _get_effective_active_translator()
    requested_onnx_family = (body.get("onnx_family") or get_active_onnx_family() or "m2m").strip().lower()
    onnx_family = requested_onnx_family if requested_onnx_family in ("m2m", "nllb") else "m2m"
    onnx_lang_map = _onnx_lang_map_for_family(onnx_family)

    if backend == "onnx" and lang not in onnx_lang_map:
        backend = "nllb"

    def _translate_pipe(t, s, tg):
        token_limit = _translation_token_limit_for_text(t)
        return (
            translate_onnx(t, s, tg, max_tokens=token_limit, onnx_family=onnx_family)
            if backend == "onnx"
            else translate(t, s, tg, max_tokens=token_limit)
        )

    # 1. To English
    _t0 = time.perf_counter()
    english_text = text if is_source_english else _translate_pipe(text, lang if backend=="onnx" else src_lang, "en")
    translate_in_s = time.perf_counter() - _t0

    # 2. RAG & Cache
    qcache = get_query_cache()
    cache_hit, rag_docs = False, []
    cache_similarity = None
    if qcache:
        cached = qcache.find_similar_query(get_embed_model().encode([english_text])[0].tolist())
        if cached:
            cached_docs, cached_sim = cached
            if cached_docs:
                rag_docs = cached_docs
                cache_similarity = float(cached_sim)
                cache_hit = True

    if not cache_hit:
        try:
            rag_docs = rag_retrieve(english_text, top_k=3, strict=True)
        except Exception:
            rag_docs = []
        if qcache and rag_docs:
            qcache.add_query(english_text, get_embed_model().encode([english_text])[0].tolist(), rag_docs)

    context = _clean_rag_context(rag_docs)
    if not context:
        # Fallback Logic
        fallback_en = NO_DB_ANSWER_EN
        answer_native = fallback_en if is_source_english else _translate_pipe(fallback_en, "en", lang if backend=="onnx" else src_lang)
        if not stream:
            return jsonify({"final_output": answer_native, "out_of_bounds": True})

        def oob_stream():
            yield f"data: {json.dumps({'type': 'meta', 'english_in': english_text, 'cache_hit': cache_hit, 'cache_similarity': cache_similarity, 'out_of_bounds': True})}\n\n"
            yield f"data: {json.dumps({'type': 'sentence', 'translated': _ensure_complete_sentence(answer_native)})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return Response(stream_with_context(oob_stream()), content_type="text/event-stream")

    # 3. Prompt
    final_prompt = _build_grounded_prompt(english_text, rag_docs)

    if not stream:
        try:
            llm_out = _finalize_answer_en(llm_generate(final_prompt, max_new_tokens=max_tokens))
            final_out = llm_out if is_source_english else _translate_pipe(llm_out, "en", lang if backend=="onnx" else src_lang)
            return jsonify({"final_output": _ensure_complete_sentence(final_out)})
        except Exception as e:
            return jsonify({"error": str(e)}), 503

    def event_stream():
        sentence_end_re = re.compile(r"(.+?[.!?](?:\"|'|”)?)(\s+|$)", re.S)
        buffer = ""
        has_meaningful_output = False
        saw_no_db_output = False
        emitted_signatures = set()
        emitted_count = 0
        yield f"data: {json.dumps({'type': 'meta', 'english_in': english_text, 'cache_hit': cache_hit, 'cache_similarity': cache_similarity})}\n\n"
        try:
            for chunk in llm_generate_stream(final_prompt, max_new_tokens=max_tokens):
                buffer += chunk
                while True:
                    m = sentence_end_re.search(buffer)
                    if not m: break
                    sent_clean = _clean_generation(m.group(1))
                    buffer = buffer[m.end():]
                    if sent_clean.strip().endswith("?"):
                        continue
                    if _is_no_db_sentence(sent_clean):
                        saw_no_db_output = True
                        continue
                    if not sent_clean.strip():
                        continue

                    signature = _sentence_signature(sent_clean)
                    if len(signature) > 12 and signature in emitted_signatures:
                        continue

                    if signature:
                        emitted_signatures.add(signature)

                    has_meaningful_output = True
                    emitted_count += 1
                    translated = sent_clean if is_source_english else _translate_pipe(sent_clean, "en", lang if backend=="onnx" else src_lang)
                    yield f"data: {json.dumps({'type': 'sentence', 'translated': _ensure_complete_sentence(translated)})}\n\n"

                    if emitted_count >= MAX_FINAL_SENTENCES:
                        break

                if emitted_count >= MAX_FINAL_SENTENCES:
                    break

            trailing = _finalize_answer_en(buffer)
            if trailing and not _is_no_db_sentence(trailing):
                trailing_sentences = re.split(r"(?<=[.!?।])\s+", trailing)
                for trailing_sentence in trailing_sentences:
                    trailing_sentence = trailing_sentence.strip()
                    if not trailing_sentence or trailing_sentence.endswith("?"):
                        continue
                    signature = _sentence_signature(trailing_sentence)
                    if len(signature) > 12 and signature in emitted_signatures:
                        continue
                    has_meaningful_output = True
                    emitted_count += 1
                    if signature:
                        emitted_signatures.add(signature)
                    translated = trailing_sentence if is_source_english else _translate_pipe(trailing_sentence, "en", lang if backend=="onnx" else src_lang)
                    yield f"data: {json.dumps({'type': 'sentence', 'translated': _ensure_complete_sentence(translated)})}\n\n"
                    if emitted_count >= MAX_FINAL_SENTENCES:
                        break
            elif (not has_meaningful_output) and (saw_no_db_output or _is_no_db_sentence(trailing)):
                fallback_native = NO_DB_ANSWER_EN if is_source_english else _translate_pipe(NO_DB_ANSWER_EN, "en", lang if backend=="onnx" else src_lang)
                yield f"data: {json.dumps({'type': 'sentence', 'translated': _ensure_complete_sentence(fallback_native)})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(stream_with_context(event_stream()), content_type="text/event-stream")

@bp.post("/infer_raw")
def ep_infer_raw():
    body = request.get_json() or {}
    prompt = body.get("prompt")
    use_rag = body.get("use_rag", True)
    rag_docs = rag_retrieve(prompt, strict=True) if use_rag else []
    if use_rag and not rag_docs:
        return jsonify({"output": NO_DB_ANSWER_EN, "out_of_bounds": True})
    final_prompt = f"Context: {rag_docs}\n\nQuestion: {prompt}" if rag_docs else prompt
    return jsonify({"output": llm_generate(final_prompt)})