"""Persistent PostgreSQL + pgvector store with safe parameterized metadata filtering."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from .models import Chunk

ALLOWED_FILTER_KEYS = {"year", "category", "section", "document_id", "language"}


def _vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{float(x):.8f}" for x in vec) + "]"


def _content_hash(chunk: Chunk) -> str:
    payload = json.dumps(
        {
            "text": chunk.text,
            "metadata": chunk.metadata,
            "page": chunk.page,
            "source": chunk.source,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PgVectorStore:
    """PostgreSQL + pgvector persistent storage using exact cosine similarity."""

    def __init__(self, database_url: str, dimensions: int) -> None:
        if not database_url:
            raise ValueError("database_url must be provided")
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.database_url = database_url
        self.dimensions = dimensions

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url)

    def create_schema(self) -> None:
        """Create pgvector extension, document_chunks table, and metadata/strategy indexes."""
        ddl = f"""
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS document_chunks (
            strategy TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            source TEXT NOT NULL,
            page INTEGER NOT NULL,
            text TEXT NOT NULL,
            metadata JSONB NOT NULL,
            content_hash TEXT NOT NULL,
            embedding vector({self.dimensions}) NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (strategy, chunk_id)
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_strategy_doc
            ON document_chunks (strategy, document_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_metadata_gin
            ON document_chunks USING GIN (metadata);
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()

    def get_existing_hashes_and_vectors(self, strategy: str) -> dict[str, tuple[str, list[float]]]:
        """Load stored (content_hash, vector) keyed by chunk_id for incremental re-ingestion."""
        sql = """
        SELECT chunk_id, content_hash, embedding::text
        FROM document_chunks
        WHERE strategy = %s;
        """
        out: dict[str, tuple[str, list[float]]] = {}
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (strategy,))
                for cid, chash, vec_txt in cur.fetchall():
                    vec = [float(x) for x in vec_txt.strip("[]").split(",")]
                    out[cid] = (chash, vec)
        return out

    def count_chunks(self, strategy: str, document_id: str | None = None) -> int:
        sql = "SELECT COUNT(*) FROM document_chunks WHERE strategy = %s"
        params: list[Any] = [strategy]
        if document_id is not None:
            sql += " AND document_id = %s"
            params.append(document_id)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
                return int(row[0]) if row else 0

    def upsert_batch_no_delete(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        """Persist a batch of chunks/vectors without deleting other chunks yet."""
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks ({len(chunks)}) and vectors ({len(vectors)}) length mismatch")
        upsert_sql = """
        INSERT INTO document_chunks (
            strategy, chunk_id, document_id, source, page, text, metadata, content_hash, embedding, updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector, NOW())
        ON CONFLICT (strategy, chunk_id) DO UPDATE SET
            document_id = EXCLUDED.document_id,
            source = EXCLUDED.source,
            page = EXCLUDED.page,
            text = EXCLUDED.text,
            metadata = EXCLUDED.metadata,
            content_hash = EXCLUDED.content_hash,
            embedding = EXCLUDED.embedding,
            updated_at = NOW();
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                for chunk, vec in zip(chunks, vectors):
                    if len(vec) != self.dimensions:
                        raise ValueError(f"Vector has {len(vec)} dims, expected {self.dimensions}")
                    cur.execute(
                        upsert_sql,
                        (
                            strategy,
                            chunk.chunk_id,
                            chunk.document_id,
                            chunk.source,
                            chunk.page,
                            chunk.text,
                            Jsonb(chunk.metadata),
                            _content_hash(chunk),
                            _vector_literal(vec),
                        ),
                    )
            conn.commit()
        return len(chunks)

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> dict[str, int]:
        """Idempotently upsert chunks & vectors, removing obsolete chunks for updated documents."""
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks ({len(chunks)}) and vectors ({len(vectors)}) length mismatch")
        for idx, vec in enumerate(vectors):
            if len(vec) != self.dimensions:
                raise ValueError(f"Vector {idx} has {len(vec)} dims, expected {self.dimensions}")

        doc_chunk_ids: dict[str, list[str]] = defaultdict(list)
        for chunk in chunks:
            doc_chunk_ids[chunk.document_id].append(chunk.chunk_id)

        deleted_obsolete = 0
        upserted = 0

        upsert_sql = """
        INSERT INTO document_chunks (
            strategy, chunk_id, document_id, source, page, text, metadata, content_hash, embedding, updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector, NOW())
        ON CONFLICT (strategy, chunk_id) DO UPDATE SET
            document_id = EXCLUDED.document_id,
            source = EXCLUDED.source,
            page = EXCLUDED.page,
            text = EXCLUDED.text,
            metadata = EXCLUDED.metadata,
            content_hash = EXCLUDED.content_hash,
            embedding = EXCLUDED.embedding,
            updated_at = NOW();
        """

        with self._connect() as conn:
            with conn.cursor() as cur:
                # Remove chunks from earlier versions of any document being ingested under this strategy
                for doc_id, active_ids in doc_chunk_ids.items():
                    cur.execute(
                        """
                        DELETE FROM document_chunks
                        WHERE strategy = %s
                          AND document_id = %s
                          AND NOT (chunk_id = ANY(%s))
                        """,
                        (strategy, doc_id, active_ids),
                    )
                    deleted_obsolete += cur.rowcount

                for chunk, vec in zip(chunks, vectors):
                    cur.execute(
                        upsert_sql,
                        (
                            strategy,
                            chunk.chunk_id,
                            chunk.document_id,
                            chunk.source,
                            chunk.page,
                            chunk.text,
                            Jsonb(chunk.metadata),
                            _content_hash(chunk),
                            _vector_literal(vec),
                        ),
                    )
                    upserted += 1
            conn.commit()

        return {"upserted": upserted, "deleted_obsolete": deleted_obsolete}

    def _build_filter_clause(self, filters: dict[str, Any] | None) -> tuple[str, list[Any]]:
        if not filters:
            return "", []
        for key in filters:
            if key not in ALLOWED_FILTER_KEYS:
                raise ValueError(f"Unsupported metadata filter key: {key!r}")
        # JSONB containment @> cleanly matches exact int/str/bool types without SQL injection
        return " AND metadata @> %s::jsonb", [json.dumps(filters, ensure_ascii=False)]

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Filter permitted metadata in SQL, rank by exact cosine similarity, and return top-k."""
        if len(query_vector) != self.dimensions:
            raise ValueError(f"Query vector has {len(query_vector)} dims, expected {self.dimensions}")
        if top_k <= 0:
            return []

        q_lit = _vector_literal(query_vector)
        filter_sql, filter_params = self._build_filter_clause(filters)

        sql = f"""
        SELECT
            chunk_id,
            document_id,
            source,
            page,
            text,
            metadata,
            1 - (embedding <=> %s::vector) AS score
        FROM document_chunks
        WHERE strategy = %s
        {filter_sql}
        ORDER BY embedding <=> %s::vector ASC, chunk_id ASC
        LIMIT %s;
        """
        params: list[Any] = [q_lit, strategy, *filter_params, q_lit, top_k]

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for rank, row in enumerate(rows, start=1):
            results.append(
                {
                    "rank": rank,
                    "chunk_id": row[0],
                    "document_id": row[1],
                    "source": row[2],
                    "page": int(row[3]),
                    "text": row[4],
                    "metadata": dict(row[5]),
                    "score": float(row[6]),
                }
            )
        return results

    def fetch_chunks(
        self,
        strategy: str,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch all stored chunks matching strategy and optional metadata filters (for BM25)."""
        filter_sql, filter_params = self._build_filter_clause(filters)
        sql = f"""
        SELECT chunk_id, document_id, source, page, text, metadata
        FROM document_chunks
        WHERE strategy = %s
        {filter_sql}
        ORDER BY document_id, page, chunk_id;
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, [strategy, *filter_params])
                rows = cur.fetchall()
        return [
            {
                "chunk_id": row[0],
                "document_id": row[1],
                "source": row[2],
                "page": int(row[3]),
                "text": row[4],
                "metadata": dict(row[5]),
            }
            for row in rows
        ]
