import re
from typing import List
import fitz  # PyMuPDF
import nltk


def _ensure_punkt():
    """Ensure NLTK punkt tokenizer is available."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


# Ensure tokenizer at import time
_ensure_punkt()


# ---------- PDF EXTRACTION ----------

def _extract_pdf_text(path: str) -> str:
    """Extract raw text from a text-based PDF."""
    doc = fitz.open(path)
    pages = []

    for page in doc:
        text = page.get_text("text")
        if text:
            pages.append(text)

    return "\n".join(pages)


# ---------- NORMALIZATION ----------

def _normalize_text(text: str) -> str:
    """Clean up PDF text noise."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------- CHUNKING ----------

def _chunk_text(
    text: str,
    max_words: int = 350,
    overlap: int = 75,
) -> List[str]:
    """Sentence-aware chunking with overlap."""

    from nltk.tokenize import sent_tokenize

    try:
        sentences = sent_tokenize(text)
    except Exception:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        words = sent.split()
        word_len = len(words)

        if current_len + word_len > max_words:
            chunks.append(" ".join(current))

            if overlap > 0:
                current = current[-overlap:]
                current_len = len(current)
            else:
                current = []
                current_len = 0

        current.extend(words)
        current_len += word_len

    if current:
        chunks.append(" ".join(current))

    return chunks


# ---------- PUBLIC API ----------

def ingest_pdf(
    path: str,
    *,
    max_words: int = 350,
    overlap: int = 75,
) -> List[str]:
    """PDF → cleaned text chunks."""

    raw_text = _extract_pdf_text(path)
    if not raw_text.strip():
        return []

    clean_text = _normalize_text(raw_text)

    chunks = _chunk_text(
        clean_text,
        max_words=max_words,
        overlap=overlap,
    )

    return [c for c in chunks if len(c.split()) > 20]
