from __future__ import annotations

import json

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from .chunking import Chunk
from .config import CHROMA_DIR, CHUNKS_FILE, COLLECTION_NAME, EMBEDDING_MODEL


def _load_chunks() -> list[Chunk]:
    data = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    return [Chunk(**c) for c in data]


def _get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def run_embedding() -> None:
    # Full rebuild each run — cheap since embedding is local, and avoids stale
    # entries if chunk_ids shift after re-chunking.
    chunks = _load_chunks()
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = _get_client()

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # nothing to delete on first run
    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)

    collection.add(
        ids=[c.chunk_id for c in chunks],
        documents=[c.content_md for c in chunks],
        metadatas=[c.model_dump(exclude={"chunk_id", "content_md"}) for c in chunks],
    )
    print(f"Embedded {len(chunks)} chunks into collection '{COLLECTION_NAME}' at {CHROMA_DIR}")
