from __future__ import annotations

import argparse
import json

from app.agent.knowledge_base import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    KnowledgeBaseRetriever,
    OllamaEmbeddingClient,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query local pgvector knowledge base")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--doc-types", default="template,qc,feature,report,security")
    parser.add_argument("--version", default="")
    parser.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    doc_types = [x.strip() for x in str(args.doc_types).split(",") if x.strip()]

    embed_client = OllamaEmbeddingClient(
        model=args.embedding_model,
        base_url=args.ollama_base_url,
        timeout_sec=args.timeout_sec,
    )
    retriever = KnowledgeBaseRetriever(embedding_client=embed_client)
    hits = retriever.retrieve(
        query=args.query,
        top_k=args.top_k,
        doc_types=doc_types,
        version=args.version or None,
    )

    out = [
        {
            "doc_name": h.doc_name,
            "doc_type": h.doc_type,
            "version": h.version,
            "updated_at": h.updated_at,
            "chunk_idx": h.chunk_idx,
            "score": h.score,
            "declared_content": h.declared_content,
        }
        for h in hits
    ]
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
