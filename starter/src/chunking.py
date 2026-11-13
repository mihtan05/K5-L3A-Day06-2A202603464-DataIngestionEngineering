"""Implement fixed-size and structure-aware (paragraph) chunking."""
from __future__ import annotations

import re

from .models import Chunk, DocumentPage

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")


def _make_chunk(
    strategy: str,
    page: DocumentPage,
    chunk_index: int,
    text: str,
) -> Chunk:
    clean = text.strip()
    meta = dict(page.metadata)
    meta.update(
        {
            "document_id": page.document_id,
            "source": page.source,
            "page": page.page,
            "chunk_strategy": strategy,
            "char_length": len(clean),
        }
    )
    chunk_id = f"{strategy}__{page.document_id}__p{page.page:03d}__c{chunk_index:03d}"
    return Chunk(
        chunk_id=chunk_id,
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=clean,
        metadata=meta,
    )


def fixed_size(page: DocumentPage, size: int = 500, overlap: int = 50, strategy_label: str = "A") -> list[Chunk]:
    """Split normalized page text into overlapping fixed-character windows."""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")

    text = page.text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    step = size - overlap
    start = 0
    idx = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + size, text_len)
        window = text[start:end].strip()
        if window:
            chunks.append(_make_chunk(strategy_label, page, idx, window))
            idx += 1
        if end >= text_len:
            break
        start += step

    return chunks


def _split_oversized_paragraph(paragraph: str, max_size: int) -> list[str]:
    """Split an oversized paragraph on sentence boundaries first, then word boundaries."""
    if len(paragraph) <= max_size:
        return [paragraph]

    sentences = _SENTENCE_SPLIT.split(paragraph)
    pieces: list[str] = []
    buf = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) > max_size:
            if buf:
                pieces.append(buf)
                buf = ""
            # Word-level fallback for extraordinarily long sentences/tables
            words = sent.split()
            word_buf = ""
            for w in words:
                candidate = f"{word_buf} {w}".strip() if word_buf else w
                if len(candidate) <= max_size:
                    word_buf = candidate
                else:
                    if word_buf:
                        pieces.append(word_buf)
                    # Hard slice only if a single token exceeds max_size
                    while len(w) > max_size:
                        pieces.append(w[:max_size])
                        w = w[max_size:]
                    word_buf = w
            if word_buf:
                pieces.append(word_buf)
        else:
            candidate = f"{buf} {sent}".strip() if buf else sent
            if len(candidate) <= max_size:
                buf = candidate
            else:
                if buf:
                    pieces.append(buf)
                buf = sent

    if buf:
        pieces.append(buf)
    return pieces


def by_paragraph(page: DocumentPage, max_size: int = 800, strategy_label: str = "B") -> list[Chunk]:
    """Group consecutive paragraphs up to max_size and split oversized paragraphs cleanly."""
    if max_size <= 0:
        raise ValueError("max_size must be positive")

    raw_paragraphs = [p.strip() for p in page.text.split("\n\n") if p.strip()]
    if not raw_paragraphs:
        return []

    atomic_units: list[str] = []
    for para in raw_paragraphs:
        atomic_units.extend(_split_oversized_paragraph(para, max_size))

    chunks: list[Chunk] = []
    buf = ""
    idx = 0

    for unit in atomic_units:
        candidate = f"{buf}\n\n{unit}".strip() if buf else unit
        if len(candidate) <= max_size:
            buf = candidate
        else:
            if buf:
                chunks.append(_make_chunk(strategy_label, page, idx, buf))
                idx += 1
            buf = unit

    if buf:
        chunks.append(_make_chunk(strategy_label, page, idx, buf))

    return chunks
