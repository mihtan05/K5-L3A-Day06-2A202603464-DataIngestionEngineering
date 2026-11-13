"""Unit and integration tests for ingestion, chunking, pgvector store, and evaluation."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from src.chunking import by_paragraph, fixed_size
from src.evaluation import evaluate_query, hybrid_rrf_search, summarize
from src.ingestion import normalize_text
from src.models import Chunk, DocumentPage
from src.store import PgVectorStore


def test_normalize_text_preserves_paragraphs_and_strips_headers() -> None:
    raw = (
        "13\n"
        "TAKING STOCK AUGUST 2024\n"
        "REACHING NEW HEIGHTS IN CAPITAL MARKETS\n"
        "Growth accelerated in H1-2024\n"
        "Viet Nam\u2019s GDP grew by 6.4 percent in\n"
        "H1-2024 after moderate growth.\n"
        "Exports also rebounded strongly."
    )
    norm = normalize_text(raw)
    assert "TAKING STOCK AUGUST 2024" not in norm
    assert "Viet Nam's GDP grew by 6.4 percent in H1-2024 after moderate growth." in norm
    assert "\n\n" in norm


def test_fixed_size_and_by_paragraph_retain_metadata() -> None:
    page = DocumentPage(
        document_id="doc1",
        source="doc1.pdf",
        page=5,
        text=("First paragraph with enough words to test chunking behavior.\n\n"
              "Second paragraph discussing policy implications and trade flows.\n\n"
              "Third paragraph with final summary numbers."),
        metadata={"year": 2024, "section": "overview", "category": "macro"},
    )
    chunks_a = fixed_size(page, size=70, overlap=15, strategy_label="A")
    assert len(chunks_a) >= 2
    assert all(c.metadata["year"] == 2024 and c.page == 5 for c in chunks_a)

    chunks_b = by_paragraph(page, max_size=130, strategy_label="B")
    assert len(chunks_b) >= 2
    assert all(len(c.text) <= 130 for c in chunks_b)
    assert all(c.metadata["section"] == "overview" for c in chunks_b)


def test_evaluate_query_accepts_alternative_gold_and_checks_filter() -> None:
    question = {
        "id": "q_test",
        "split": "dev",
        "type": "metadata_filter",
        "question": "What was the forecast?",
        "expected_answer": "6.1 percent",
        "gold_source_id": "doc_2024",
        "gold_page": 16,
        "evidence_quote": "forecast to grow by 6.1 percent",
        "alternative_gold": [
            {"source_id": "doc_2024", "page": 33, "evidence_quote": "forecast to grow by 6.1 percent"}
        ],
        "filter": {"year": 2024},
    }
    hits = [
        {
            "chunk_id": "c1",
            "document_id": "doc_2024",
            "source": "a.pdf",
            "page": 33,
            "score": 0.91,
            "metadata": {"year": 2024},
            "text": "The economy is forecast to grow by 6.1 percent in 2024.",
        }
    ]
    trace = evaluate_query(question, hits, applied_filter={"year": 2024})
    assert trace["hit_at_1"] is True
    assert trace["evidence_hit_at_3"] is True
    assert trace["filter_compliant_top3"] is True
    agg = summarize([trace])
    assert agg["dev"]["hit_at_1_count"] == 1


def test_hybrid_rrf_boosts_lexical_match() -> None:
    pool = [
        {"chunk_id": "c1", "document_id": "d1", "source": "s", "page": 1, "metadata": {}, "text": "Macro outlook and GDP growth"},
        {"chunk_id": "c2", "document_id": "d1", "source": "s", "page": 2, "metadata": {}, "text": "VSDC sole central securities depositary clearing"},
        {"chunk_id": "c3", "document_id": "d1", "source": "s", "page": 3, "metadata": {}, "text": "Public investment infrastructure disbursement"},
        {"chunk_id": "c4", "document_id": "d1", "source": "s", "page": 4, "metadata": {}, "text": "Electric two-wheelers battery charging demand"},
    ]
    dense = [
        {**pool[0], "score": 0.85, "rank": 1},
        {**pool[1], "score": 0.84, "rank": 2},
    ]
    fused = hybrid_rrf_search(dense, pool, "VSDC central securities depositary", top_k=2, bm25_weight=1.5)
    assert fused[0]["chunk_id"] == "c2"


def test_pgvector_store_upsert_filter_and_cleanup() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    db_url = os.getenv("DATABASE_URL")
    assert db_url, "DATABASE_URL required for store integration test"

    store = PgVectorStore(database_url=db_url, dimensions=768)
    store.create_schema()
    strat = "_pytest_tmp"

    def vec(idx: int) -> list[float]:
        v = [0.0] * 768
        v[idx] = 1.0
        return v

    c1 = Chunk("p1", "doc_a", "a.pdf", 1, "Alpha 2024 policy", {"year": 2024, "section": "s1"})
    c2 = Chunk("p2", "doc_b", "b.pdf", 2, "Beta 2025 policy", {"year": 2025, "section": "s2"})
    try:
        store.upsert(strat, [c1, c2], [vec(0), vec(1)])
        assert store.count_chunks(strat) == 2

        # Re-ingest same data -> 0 duplicates
        store.upsert(strat, [c1, c2], [vec(0), vec(1)])
        assert store.count_chunks(strat) == 2

        # SQL metadata filter works and blocks unsupported filter keys
        res = store.search(strat, vec(0), top_k=3, filters={"year": 2025})
        assert len(res) == 1 and res[0]["chunk_id"] == "p2"
        with pytest.raises(ValueError):
            store.search(strat, vec(0), filters={"invalid_key": "x"})

        # Upserting updated doc_a with empty replacement removes obsolete p1? Test shrinking 2->1
        c1b = Chunk("p1b", "doc_a", "a.pdf", 1, "Alpha updated", {"year": 2024, "section": "s1"})
        res_upd = store.upsert(strat, [c1b], [vec(0)])
        assert res_upd["deleted_obsolete"] == 1
        assert store.count_chunks(strat, "doc_a") == 1
    finally:
        with store._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM document_chunks WHERE strategy = %s", (strat,))
            conn.commit()
