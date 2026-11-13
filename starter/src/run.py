"""End-to-end Day 06 Data Ingestion & Retrieval Benchmark Pipeline."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import build_embedder
from .evaluation import chunk_length_stats, evaluate_query, hybrid_rrf_search, normalize_for_match, summarize
from .ingestion import load_pdf
from .models import Chunk, DocumentPage, PageLog
from .store import PgVectorStore, _content_hash

REPO_ROOT = Path(__file__).resolve().parents[2]
STARTER_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.json"
BENCHMARK_PATH = REPO_ROOT / "benchmark" / "questions.json"
RESULTS_DIR = STARTER_ROOT / "results"


def load_corpus_from_manifest(manifest_path: Path) -> tuple[list[DocumentPage], list[PageLog], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_pages: list[DocumentPage] = []
    all_logs: list[PageLog] = []

    for doc in manifest["documents"]:
        excluded = {
            int(item["page"]): item["reason"]
            for item in doc["excluded_or_empty_pages"]
            if item["status"] == "excluded_front_back"
        }
        pdf_path = REPO_ROOT / doc["local_path"]
        pages, logs = load_pdf(
            path=pdf_path,
            document_id=doc["source_id"],
            source=doc["local_path"],
            base_metadata=doc["base_metadata"],
            excluded_pages=excluded,
            section_ranges=doc["section_ranges"],
        )
        all_pages.extend(pages)
        all_logs.extend(logs)

    if len(all_pages) < 30:
        raise RuntimeError(f"Corpus has only {len(all_pages)} content pages; minimum 30 required.")
    return all_pages, all_logs, manifest


def validate_benchmark(benchmark_path: Path, pages: list[DocumentPage]) -> list[dict[str, Any]]:
    bm = json.loads(benchmark_path.read_text(encoding="utf-8"))
    questions: list[dict[str, Any]] = bm["questions"]
    if len(questions) < 20:
        raise RuntimeError(f"Benchmark has {len(questions)} questions; minimum 20 required.")

    dev_count = sum(1 for q in questions if q["split"] == "dev")
    test_count = sum(1 for q in questions if q["split"] == "test")
    filter_total = sum(1 for q in questions if q.get("filter"))
    filter_test = sum(1 for q in questions if q.get("filter") and q["split"] == "test")

    if dev_count < 15 or test_count < 5:
        raise RuntimeError(f"Split invalid: dev={dev_count}, test={test_count} (need >=15 dev, >=5 test)")
    if filter_total < 3 or filter_test < 1:
        raise RuntimeError(f"Filter queries invalid: total={filter_total}, test={filter_test} (need >=3 total, >=1 test)")

    page_lookup = {(p.document_id, p.page): normalize_for_match(p.text) for p in pages}
    for q in questions:
        key = (q["gold_source_id"], int(q["gold_page"]))
        if key not in page_lookup:
            raise RuntimeError(f"Missing gold page {key} for {q['id']}")
        if normalize_for_match(q["evidence_quote"]) not in page_lookup[key]:
            raise RuntimeError(f"Gold evidence_quote not found on {key} for {q['id']}")
        for alt in q.get("alternative_gold", []):
            akey = (alt["source_id"], int(alt["page"]))
            if akey not in page_lookup or normalize_for_match(alt["evidence_quote"]) not in page_lookup[akey]:
                raise RuntimeError(f"Alternative gold quote not found on {akey} for {q['id']}")

    return questions


def verify_reingestion(store: PgVectorStore, b_chunks: list[Chunk], b_vectors: list[list[float]]) -> dict[str, Any]:
    """Verify idempotent re-ingestion (0 duplicates) and stale-chunk cleanup when a doc shrinks."""
    count_before = store.count_chunks("B")
    reingest_res = store.upsert("B", b_chunks, b_vectors)
    count_after = store.count_chunks("B")

    # Isolated test strategy for stale chunk removal on document update
    test_strat = "_reingest_test"
    dummy_c = [
        Chunk(f"t{i}", "dummy_doc", "dummy.pdf", 1, f"sample text {i}", {"year": 2024})
        for i in range(3)
    ]
    dummy_v = [[1.0] + [0.0] * (store.dimensions - 1) for _ in range(3)]
    store.upsert(test_strat, dummy_c, dummy_v)
    after_3 = store.count_chunks(test_strat, "dummy_doc")
    res_shrink = store.upsert(test_strat, dummy_c[:2], dummy_v[:2])
    after_2 = store.count_chunks(test_strat, "dummy_doc")
    # Clean up temporary test rows
    with store._connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM document_chunks WHERE strategy = %s", (test_strat,))
        conn.commit()

    return {
        "strategy_b_count_before": count_before,
        "strategy_b_count_after_reingest": count_after,
        "duplicates_created": count_after - count_before,
        "reingest_upsert_result": reingest_res,
        "stale_cleanup_test": {
            "initial_count": after_3,
            "after_shrink_count": after_2,
            "deleted_obsolete": res_shrink["deleted_obsolete"],
        },
    }


def _export_audit_templates(
    pages: list[DocumentPage],
    chunks_a: list[Chunk],
    chunks_b: list[Chunk],
    traces_by_strategy: dict[str, list[dict[str, Any]]],
    out_dir: Path,
) -> None:
    """Export deterministic pre-selected samples for manual Coherence, Metadata, and Grounding audits."""
    # 1. Coherence sample: first 5 sorted content pages per document (10 fixed positions total)
    by_doc_pages: dict[str, list[DocumentPage]] = {}
    for p in sorted(pages, key=lambda x: (x.document_id, x.page)):
        by_doc_pages.setdefault(p.document_id, []).append(p)
    selected_positions: list[DocumentPage] = []
    for doc_id in sorted(by_doc_pages):
        selected_positions.extend(by_doc_pages[doc_id][:5])

    a_by_pos: dict[tuple[str, int], list[Chunk]] = {}
    b_by_pos: dict[tuple[str, int], list[Chunk]] = {}
    for c in chunks_a:
        a_by_pos.setdefault((c.document_id, c.page), []).append(c)
    for c in chunks_b:
        b_by_pos.setdefault((c.document_id, c.page), []).append(c)

    with (out_dir / "audit_coherence_sample.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "position_id", "document_id", "page", "section", "strategy",
            "chunk_id", "char_length", "chunk_text", "coherence_score_0_2", "rationale"
        ])
        for idx, pos in enumerate(selected_positions, start=1):
            # Pick the first substantive chunk at each fixed page position (and B=C=D share B's chunks)
            for strat_label, lookup in [("A", a_by_pos), ("B_C_D", b_by_pos)]:
                c = lookup[(pos.document_id, pos.page)][0]
                writer.writerow([
                    f"pos_{idx:02d}", pos.document_id, pos.page, pos.metadata.get("section", ""),
                    strat_label, c.chunk_id, len(c.text), c.text, "", ""
                ])

    # 2. Metadata usefulness sample: 5 test questions x top-3 across A, B, C, D
    with (out_dir / "audit_metadata_sample.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "strategy", "question_id", "rank", "chunk_id", "document_id", "source", "page",
            "year", "section", "category", "question_filter", "filter_compliant",
            "metadata_complete_and_accurate", "notes"
        ])
        for strat in ("A", "B", "C", "D"):
            for trace in traces_by_strategy[strat]:
                if trace["split"] != "test":
                    continue
                for h in trace["hits"]:
                    m = h["metadata"]
                    writer.writerow([
                        strat, trace["question_id"], h["rank"], h["chunk_id"],
                        h["document_id"], h["source"], h["page"],
                        m.get("year"), m.get("section"), m.get("category"),
                        json.dumps(trace["question_filter"]) if trace["question_filter"] else "none",
                        h["filter_compliant"] if h["filter_compliant"] is not None else "N/A",
                        "", ""
                    ])

    # 3. Grounding quality sample: 5 test questions per strategy
    with (out_dir / "audit_grounding_sample.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "strategy", "question_id", "type", "question", "expected_answer",
            "gold_source_page", "evidence_quote", "evidence_hit_at_3",
            "top3_summary", "grounding_score_0_2", "rationale"
        ])
        for strat in ("A", "B", "C", "D"):
            for trace in traces_by_strategy[strat]:
                if trace["split"] != "test":
                    continue
                top3_txt = " || ".join(
                    f"[{h['rank']}:{h['document_id']} p{h['page']}] {h['text'][:240]}"
                    for h in trace["hits"]
                )
                writer.writerow([
                    strat, trace["question_id"], trace["type"], trace["question"],
                    trace["expected_answer"],
                    f"{trace['gold_source_id']} p{trace['gold_page']}",
                    trace["evidence_quote"], trace["evidence_hit_at_3"],
                    top3_txt, "", ""
                ])


def _ingest_strategy_incremental(
    store: PgVectorStore,
    embedder: Any,
    strategy: str,
    chunks: list[Chunk],
    batch_size: int = 45,
) -> tuple[list[list[float]], dict[str, int]]:
    """Embed only missing/changed chunks (matched by content_hash) and upsert incrementally."""
    existing = store.get_existing_hashes_and_vectors(strategy)
    vectors_by_id: dict[str, list[float]] = {}
    to_embed: list[Chunk] = []

    for c in chunks:
        chash = _content_hash(c)
        if c.chunk_id in existing and existing[c.chunk_id][0] == chash:
            vectors_by_id[c.chunk_id] = existing[c.chunk_id][1]
        else:
            to_embed.append(c)

    reused_count = len(chunks) - len(to_embed)
    newly_embedded = 0
    for start in range(0, len(to_embed), batch_size):
        batch = to_embed[start : start + batch_size]
        vecs = embedder.embed_documents([c.text for c in batch])
        for c, v in zip(batch, vecs):
            vectors_by_id[c.chunk_id] = v
        # Partial batch upsert without deleting remaining docs yet — just insert/update batch
        store.upsert_batch_no_delete(strategy, batch, vecs)
        newly_embedded += len(batch)
        print(f"[{strategy}] Embedded & persisted {newly_embedded}/{len(to_embed)} new chunks (reused {reused_count})")

    ordered_vectors = [vectors_by_id[c.chunk_id] for c in chunks]
    # Full transactional upsert reconciles obsolete chunks while keeping exact row set
    final_res = store.upsert(strategy, chunks, ordered_vectors)
    final_res["reused_cached"] = reused_count
    final_res["newly_embedded"] = newly_embedded
    return ordered_vectors, final_res


def run_pipeline(
    split_filter: str = "all",
    skip_ingest: bool = False,
    k_rrf: int = 60,
    bm25_weight: float = 0.8,
) -> dict[str, Any]:
    load_dotenv(STARTER_ROOT / ".env")
    database_url = os.getenv("DATABASE_URL")
    provider = os.getenv("EMBEDDING_PROVIDER", "local")
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load & validate corpus + benchmark
    t_ingest_start = time.perf_counter()
    pages, page_logs, manifest = load_corpus_from_manifest(MANIFEST_PATH)
    questions = validate_benchmark(BENCHMARK_PATH, pages)
    if split_filter in ("dev", "test"):
        active_questions = [q for q in questions if q["split"] == split_filter]
    else:
        active_questions = questions

    # 2. Chunking A (fixed-size) and B (by_paragraph; shared with C & D)
    chunks_a = [c for p in pages for c in fixed_size(p, size=500, overlap=50, strategy_label="A")]
    chunks_b = [c for p in pages for c in by_paragraph(p, max_size=800, strategy_label="B")]

    embedder = build_embedder(provider=provider, model_name=model_name)
    store = PgVectorStore(database_url=database_url or "", dimensions=embedder.dimensions)
    store.create_schema()

    # Ensure we never mix embeddings from a different model in PostgreSQL
    active_model_file = RESULTS_DIR / "active_embedding_model.json"
    prev_model = None
    if active_model_file.exists():
        prev_model = json.loads(active_model_file.read_text(encoding="utf-8")).get("model_name")
    if not skip_ingest and prev_model != embedder.model_name:
        with store._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM document_chunks WHERE strategy IN ('A', 'B')")
            conn.commit()
        active_model_file.write_text(
            json.dumps({"provider": provider, "model_name": embedder.model_name, "dimensions": embedder.dimensions}, indent=2),
            encoding="utf-8",
        )

    ingest_timings: dict[str, float] = {}
    reingest_report: dict[str, Any] = {}

    if not skip_ingest:
        t0 = time.perf_counter()
        _, res_a = _ingest_strategy_incremental(store, embedder, "A", chunks_a)
        ingest_timings["A_seconds"] = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        vecs_b, res_b = _ingest_strategy_incremental(store, embedder, "B", chunks_b)
        ingest_timings["B_C_D_seconds"] = round(time.perf_counter() - t0, 3)

        reingest_report = verify_reingestion(store, chunks_b, vecs_b)
        reingest_report["incremental_stats"] = {"A": res_a, "B": res_b}
    else:
        ingest_timings["A_seconds"] = 0.0
        ingest_timings["B_C_D_seconds"] = 0.0

    ingest_usage = asdict(embedder.usage)
    total_ingest_elapsed = round(time.perf_counter() - t_ingest_start, 3)

    # Reset usage counter to measure query embedding separately
    embedder.usage = type(embedder.usage)()

    # 3. Embed queries once (and cache locally by question id + text hash) so A/B/C/D share identical vectors
    qcache_path = RESULTS_DIR / "query_vectors_cache.json"
    qcache: dict[str, Any] = json.loads(qcache_path.read_text(encoding="utf-8")) if qcache_path.exists() else {}
    t_q_start = time.perf_counter()
    query_vectors: dict[str, list[float]] = {}
    for q in active_questions:
        qid = q["id"]
        cached = qcache.get(qid)
        if cached and cached.get("question") == q["question"] and cached.get("model") == model_name:
            query_vectors[qid] = cached["vector"]
        else:
            vec = embedder.embed_query(q["question"])
            query_vectors[qid] = vec
            qcache[qid] = {"question": q["question"], "model": model_name, "vector": vec}
    qcache_path.write_text(json.dumps(qcache, ensure_ascii=False), encoding="utf-8")
    query_embed_seconds = round(time.perf_counter() - t_q_start, 3)
    query_usage = asdict(embedder.usage)

    # 4. Run retrieval for A, B, C, and D (Hybrid RRF)
    traces_by_strategy: dict[str, list[dict[str, Any]]] = {s: [] for s in ("A", "B", "C", "D")}
    search_timings: dict[str, float] = {}

    for strat in ("A", "B", "C"):
        t0 = time.perf_counter()
        store_strat = "A" if strat == "A" else "B"
        for q in active_questions:
            applied_filter = q["filter"] if (strat == "C" and q.get("filter")) else None
            hits = store.search(
                strategy=store_strat,
                query_vector=query_vectors[q["id"]],
                top_k=3,
                filters=applied_filter,
            )
            traces_by_strategy[strat].append(evaluate_query(q, hits, applied_filter=applied_filter))
        search_timings[strat] = round(time.perf_counter() - t0, 3)

    # Strategy D: Hybrid Dense (top-15 from B index with C's filter) + BM25 via RRF
    t0 = time.perf_counter()
    for q in active_questions:
        applied_filter = q["filter"] if q.get("filter") else None
        dense_candidates = store.search(
            strategy="B",
            query_vector=query_vectors[q["id"]],
            top_k=15,
            filters=applied_filter,
        )
        pool = store.fetch_chunks(strategy="B", filters=applied_filter)
        hybrid_hits = hybrid_rrf_search(
            dense_hits=dense_candidates,
            candidate_pool=pool,
            query_text=q["question"],
            top_k=3,
            k_rrf=k_rrf,
            dense_weight=1.0,
            bm25_weight=bm25_weight,
        )
        traces_by_strategy["D"].append(evaluate_query(q, hybrid_hits, applied_filter=applied_filter))
    search_timings["D"] = round(time.perf_counter() - t0, 3)

    # 5. Save ingestion log, reingest check, traces, metrics summary, and audit templates
    ingestion_log = {
        "corpus_total_content_pages": len(pages),
        "manifest_total_pages": manifest["total_counted_content_pages"],
        "page_logs": [asdict(l) for l in page_logs],
        "skipped_or_empty_pages": [asdict(l) for l in page_logs if l.status != "included"],
        "chunk_stats": {
            "A": chunk_length_stats([len(c.text) for c in chunks_a]),
            "B_C_D": chunk_length_stats([len(c.text) for c in chunks_b]),
        },
        "embedding_provider": provider,
        "embedding_model": embedder.model_name,
        "embedding_dimensions": embedder.dimensions,
        "ingest_timings_seconds": ingest_timings,
        "total_ingest_elapsed_seconds": total_ingest_elapsed,
        "ingest_api_usage": ingest_usage,
        "estimated_ingest_tokens_approx": round(ingest_usage["total_chars"] / 4),
        "estimated_cost_usd": 0.0,
    }
    (RESULTS_DIR / "ingestion_log.json").write_text(
        json.dumps(ingestion_log, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if reingest_report:
        (RESULTS_DIR / "reingest_check.json").write_text(
            json.dumps(reingest_report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    metrics_summary: dict[str, Any] = {
        "split_evaluated": split_filter,
        "embedding_model": model_name,
        "embedding_dimensions": embedder.dimensions,
        "vector_store": "PostgreSQL 18 + pgvector 0.8.1 (exact cosine similarity)",
        "hybrid_params": {"k_rrf": k_rrf, "dense_weight": 1.0, "bm25_weight": bm25_weight},
        "query_embed_seconds": query_embed_seconds,
        "query_api_usage": query_usage,
        "search_timings_seconds": search_timings,
        "strategies": {},
    }
    for strat, trs in traces_by_strategy.items():
        (RESULTS_DIR / f"traces_{strat}.json").write_text(
            json.dumps(trs, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        metrics_summary["strategies"][strat] = summarize(trs)

    (RESULTS_DIR / "metrics_summary.json").write_text(
        json.dumps(metrics_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if split_filter in ("all", "test"):
        _export_audit_templates(pages, chunks_a, chunks_b, traces_by_strategy, RESULTS_DIR)

    return metrics_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Day 06 Data Ingestion & Retrieval pipeline")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--k-rrf", type=int, default=60)
    parser.add_argument("--bm25-weight", type=float, default=0.8)
    args = parser.parse_args()

    summary = run_pipeline(
        split_filter=args.split,
        skip_ingest=args.skip_ingest,
        k_rrf=args.k_rrf,
        bm25_weight=args.bm25_weight,
    )
    print(json.dumps(summary["strategies"], indent=2))


if __name__ == "__main__":
    main()
