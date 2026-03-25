from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.agent.knowledge_base import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_KB_DIR,
    DEFAULT_OLLAMA_BASE_URL,
    KnowledgeBaseIndexer,
    OllamaEmbeddingClient,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index local knowledge_base docs into pgvector")
    parser.add_argument("--kb-dir", default=str(DEFAULT_KB_DIR))
    parser.add_argument("--version", default="v1")
    parser.add_argument("--replace-version", action="store_true")
    parser.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    embed_client = OllamaEmbeddingClient(
        model=args.embedding_model,
        base_url=args.ollama_base_url,
        timeout_sec=args.timeout_sec,
    )
    indexer = KnowledgeBaseIndexer(
        embedding_client=embed_client,
        kb_dir=Path(args.kb_dir).resolve(),
    )
    summary = indexer.index(version=args.version, replace_version=bool(args.replace_version))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
