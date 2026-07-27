from __future__ import annotations

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from .config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL


class Retriever:
    # Loads the embedding model and collection once so repeated retrieve() calls
    # (e.g. one per eval question) don't pay the model-load cost every time.
    def __init__(self) -> None:
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)

    def retrieve(self, text: str, k: int = 5) -> list[dict]:
        results = self._collection.query(query_texts=[text], n_results=k)
        hits = []
        for i, chunk_id in enumerate(results["ids"][0]):
            hits.append({
                "chunk_id": chunk_id,
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return hits
