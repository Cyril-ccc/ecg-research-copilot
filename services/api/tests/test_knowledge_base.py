from __future__ import annotations

from contextlib import contextmanager

import pytest

import app.agent.knowledge_base as kb


class _FakeEmbeddingClient:
    def embed_text(self, text: str) -> list[float]:
        assert text
        return [0.1, 0.2, 0.3]


class _FakeCursor:
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.last_sql = ""
        self.last_params = []

    def execute(self, sql: str, params=None) -> None:
        self.last_sql = sql
        self.last_params = params or []

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeConn:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def cursor(self):
        return _FakeCursor(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@contextmanager
def _fake_get_meta_conn(rows: list[dict]):
    yield _FakeConn(rows)


def test_retriever_applies_dedup_and_adds_non_executable_declaration(monkeypatch):
    rows = [
        {
            "doc_name": "feature_defs.md",
            "doc_type": "feature",
            "version": "v1",
            "updated_at": "2026-03-06T00:00:00Z",
            "chunk_idx": 0,
            "content": "qtc_ms = qt_ms / sqrt(rr_sec)",
            "score": 0.93,
        },
        {
            "doc_name": "feature_defs.md",
            "doc_type": "feature",
            "version": "v1",
            "updated_at": "2026-03-06T00:00:00Z",
            "chunk_idx": 1,
            "content": "qtc_ms = qt_ms / sqrt(rr_sec)",
            "score": 0.92,
        },
    ]

    monkeypatch.setattr(kb, "get_meta_conn", lambda: _fake_get_meta_conn(rows))
    retriever = kb.KnowledgeBaseRetriever(embedding_client=_FakeEmbeddingClient())

    hits = retriever.retrieve(query="QTc 是怎么算的？", top_k=3, doc_types=["feature"])

    assert len(hits) == 1
    assert hits[0].doc_name == "feature_defs.md"
    assert kb.NON_EXECUTABLE_DECLARATION in hits[0].declared_content


def test_retriever_rejects_illegal_doc_type_filter():
    retriever = kb.KnowledgeBaseRetriever(embedding_client=_FakeEmbeddingClient())

    with pytest.raises(ValueError, match="unsupported doc_type filter"):
        retriever.retrieve(query="test", top_k=3, doc_types=["template", "unknown"])


def test_priority_doc_names_for_mapping_queries():
    icd_docs = kb._priority_doc_names_for_query("AF对应的ICD是什么？")
    drug_docs = kb._priority_doc_names_for_query("胺碘酮在处方表里用什么英文名？")

    assert "disease_icd_map.md" in icd_docs
    assert "drug_alias_map.md" in drug_docs
