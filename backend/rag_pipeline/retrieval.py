from __future__ import annotations

import re

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi

from .config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL

RRF_K = 50  # standard Reciprocal Rank Fusion constant


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class Retriever:
    # Loads the embedding model, collection, and BM25 index once so repeated
    # retrieve() calls (e.g. one per eval question) don't pay the model-load
    # cost every time.
    def __init__(self) -> None:
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)

        # Build the BM25 index from whatever's actually in the collection, so it
        # always matches the vector index rather than a separately-read file.
        all_items = self._collection.get(include=["documents", "metadatas"])
        self._chunk_ids = all_items["ids"]
        self._documents = all_items["documents"]
        self._metadatas = all_items["metadatas"]
        self._bm25 = BM25Okapi([_tokenize(doc) for doc in self._documents])

    def _fuse(self, text: str, pool: int) -> dict[str, float]:
        vector_results = self._collection.query(query_texts=[text], n_results=pool)
        vector_rank = {cid: rank for rank, cid in enumerate(vector_results["ids"][0], start=1)}

        bm25_scores = self._bm25.get_scores(_tokenize(text))
        bm25_top = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])[:pool]
        bm25_rank = {self._chunk_ids[i]: rank for rank, i in enumerate(bm25_top, start=1)}

        # Reciprocal Rank Fusion — combines ranks (not raw scores) since cosine
        # distance and BM25 scores aren't on comparable scales.
        candidate_ids = set(vector_rank) | set(bm25_rank)
        return {
            cid: (1 / (RRF_K + vector_rank[cid]) if cid in vector_rank else 0)
            + (1 / (RRF_K + bm25_rank[cid]) if cid in bm25_rank else 0)
            for cid in candidate_ids
        }

    def retrieve(self, text: str, k: int = 5) -> list[dict]:
        pool = max(k * 4, 20)
        fused_scores = self._fuse(text, pool)
        # Break ties on chunk_id so results are reproducible across processes —
        # sorting a set alone left tied scores ordered by hash-randomized iteration.
        top_ids = sorted(fused_scores, key=lambda cid: (-fused_scores[cid], cid))[:k]

        id_to_idx = {cid: i for i, cid in enumerate(self._chunk_ids)}
        return [
            {
                "chunk_id": cid,
                "document": self._documents[id_to_idx[cid]],
                "metadata": self._metadatas[id_to_idx[cid]],
                "score": fused_scores[cid],
            }
            for cid in top_ids
        ]
