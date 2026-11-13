"""Evaluate retrieval accuracy (Hit@1, Evidence Hit@3), filter compliance, and Hybrid RRF."""
from __future__ import annotations

import re
import statistics
from typing import Any

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", re.IGNORECASE)


def normalize_for_match(text: str) -> str:
    """Normalize whitespace and casing for substring evidence verification."""
    return re.sub(r"\s+", " ", text).strip().lower()


def tokenize_for_bm25(text: str) -> list[str]:
    """Simple alphanumeric tokenizer preserving hyphenated/decimal terms like H1-2024 or 7.1."""
    return _TOKEN_RE.findall(text.lower())


def _is_filter_compliant(hit_meta: dict[str, Any], filters: dict[str, Any] | None) -> bool | None:
    if not filters:
        return None
    for k, expected in filters.items():
        if hit_meta.get(k) != expected:
            return False
    return True


def _matches_gold_evidence(hit: dict[str, Any], question: dict[str, Any]) -> bool:
    """Check whether hit belongs to a valid gold source/page and contains the verified evidence quote."""
    candidates = [
        {
            "source_id": question["gold_source_id"],
            "page": int(question["gold_page"]),
            "evidence_quote": question["evidence_quote"],
        }
    ]
    for alt in question.get("alternative_gold", []):
        candidates.append(
            {
                "source_id": alt["source_id"],
                "page": int(alt["page"]),
                "evidence_quote": alt["evidence_quote"],
            }
        )

    hit_text = normalize_for_match(hit["text"])
    hit_doc = hit.get("document_id") or hit.get("metadata", {}).get("document_id")
    hit_page = int(hit["page"])

    for cand in candidates:
        if hit_doc == cand["source_id"] and hit_page == cand["page"]:
            if normalize_for_match(cand["evidence_quote"]) in hit_text:
                return True
    return False


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]], applied_filter: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute Hit@1, Evidence Hit@3, and filter compliance while preserving full top-3 traces."""
    annotated_hits: list[dict[str, Any]] = []
    expected_filter = question.get("filter")

    for idx, hit in enumerate(hits, start=1):
        is_hit = _matches_gold_evidence(hit, question)
        compliant = _is_filter_compliant(hit.get("metadata", {}), expected_filter)
        annotated_hits.append(
            {
                "rank": idx,
                "chunk_id": hit["chunk_id"],
                "document_id": hit["document_id"],
                "source": hit["source"],
                "page": hit["page"],
                "score": round(float(hit["score"]), 6),
                "metadata": hit["metadata"],
                "text": hit["text"],
                "is_hit": is_hit,
                "filter_compliant": compliant,
            }
        )

    hit_at_1 = bool(annotated_hits and annotated_hits[0]["is_hit"])
    evidence_hit_at_3 = any(h["is_hit"] for h in annotated_hits[:3])

    if expected_filter:
        all_compliant = bool(annotated_hits) and all(
            h["filter_compliant"] is True for h in annotated_hits[:3]
        )
    else:
        all_compliant = None

    return {
        "question_id": question["id"],
        "split": question["split"],
        "type": question.get("type", ""),
        "question": question["question"],
        "expected_answer": question["expected_answer"],
        "gold_source_id": question["gold_source_id"],
        "gold_page": question["gold_page"],
        "evidence_quote": question["evidence_quote"],
        "alternative_gold": question.get("alternative_gold", []),
        "question_filter": expected_filter,
        "applied_filter": applied_filter,
        "hit_at_1": hit_at_1,
        "evidence_hit_at_3": evidence_hit_at_3,
        "filter_compliant_top3": all_compliant,
        "hits": annotated_hits,
    }


def chunk_length_stats(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        return {"count": 0, "total_chars": 0, "min": 0, "median": 0, "p95": 0, "max": 0}
    sorted_lens = sorted(lengths)
    p95_idx = min(len(sorted_lens) - 1, int(round(0.95 * (len(sorted_lens) - 1))))
    return {
        "count": len(sorted_lens),
        "total_chars": sum(sorted_lens),
        "min": sorted_lens[0],
        "median": int(statistics.median(sorted_lens)),
        "p95": sorted_lens[p95_idx],
        "max": sorted_lens[-1],
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1 and Evidence Hit@3 by split (dev, test, all) plus filter compliance."""
    summary: dict[str, Any] = {}
    for split_name in ("dev", "test", "all"):
        subset = traces if split_name == "all" else [t for t in traces if t["split"] == split_name]
        n = len(subset)
        h1 = sum(1 for t in subset if t["hit_at_1"])
        h3 = sum(1 for t in subset if t["evidence_hit_at_3"])
        filter_qs = [t for t in subset if t.get("question_filter")]
        f_ok = sum(1 for t in filter_qs if t.get("filter_compliant_top3") is True)
        summary[split_name] = {
            "n": n,
            "hit_at_1_count": h1,
            "hit_at_1_rate": round(h1 / n, 4) if n else 0.0,
            "evidence_hit_at_3_count": h3,
            "evidence_hit_at_3_rate": round(h3 / n, 4) if n else 0.0,
            "filter_queries_count": len(filter_qs),
            "filter_compliant_count": f_ok if filter_qs else None,
        }
    return summary


def hybrid_rrf_search(
    dense_hits: list[dict[str, Any]],
    candidate_pool: list[dict[str, Any]],
    query_text: str,
    top_k: int = 3,
    k_rrf: int = 60,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> list[dict[str, Any]]:
    """Combine dense vector hits and BM25 lexical hits over the same candidate pool via RRF."""
    if not candidate_pool:
        return []

    tokenized_corpus = [tokenize_for_bm25(c["text"]) for c in candidate_pool]
    bm25 = BM25Okapi(tokenized_corpus)
    q_tokens = tokenize_for_bm25(query_text)
    bm25_scores = bm25.get_scores(q_tokens) if q_tokens else [0.0] * len(candidate_pool)

    bm25_ranked = sorted(
        range(len(candidate_pool)),
        key=lambda i: (-float(bm25_scores[i]), candidate_pool[i]["chunk_id"]),
    )
    bm25_rank_map = {
        candidate_pool[idx]["chunk_id"]: rank
        for rank, idx in enumerate(bm25_ranked, start=1)
        if bm25_scores[idx] > 0
    }
    dense_rank_map = {hit["chunk_id"]: rank for rank, hit in enumerate(dense_hits, start=1)}
    by_id = {c["chunk_id"]: c for c in candidate_pool}

    fused_scores: list[tuple[str, float]] = []
    relevant_ids = set(dense_rank_map) | set(bm25_rank_map)
    for cid in relevant_ids:
        score = 0.0
        if cid in dense_rank_map:
            score += dense_weight / (k_rrf + dense_rank_map[cid])
        if cid in bm25_rank_map:
            score += bm25_weight / (k_rrf + bm25_rank_map[cid])
        fused_scores.append((cid, score))

    fused_scores.sort(key=lambda item: (-item[1], item[0]))

    results: list[dict[str, Any]] = []
    for rank, (cid, rrf_score) in enumerate(fused_scores[:top_k], start=1):
        rec = dict(by_id[cid])
        rec["rank"] = rank
        rec["score"] = rrf_score
        results.append(rec)
    return results
