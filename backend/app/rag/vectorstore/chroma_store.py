import json
import math
from typing import Any

from app.db.chromadb import get_chroma_collection
from langsmith import traceable


class ChromaVectorStore:
    def __init__(self) -> None:
        self.collection = get_chroma_collection()

    def _normalize_metadata_value(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def _to_chroma_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in metadata.items():
            if key == "content" or value is None:
                continue
            normalized_value = self._normalize_metadata_value(value)
            if normalized_value is not None:
                normalized[key] = normalized_value
        return normalized

    def add_chunks(self, chunks: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        ids = [chunk["id"] for chunk in chunks]
        documents = [chunk["content"] for chunk in chunks]
        metadatas = [self._to_chroma_metadata(chunk["metadata"]) for chunk in chunks]
        self.collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if len(left) == 0 or len(right) == 0:
            return 0.0
        dot_product = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot_product / (left_norm * right_norm)

    def search(self, query_embedding: list[float], where_filter: dict[str, Any] | None, top_k: int = 5) -> list[dict[str, Any]]:
        return self._search_traced(query_embedding, where_filter, top_k)

    @traceable(name="chroma_vector_search")
    def _search_traced(self, query_embedding: list[float], where_filter: dict[str, Any] | None, top_k: int = 5) -> list[dict[str, Any]]:
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        items: list[dict[str, Any]] = []
        for idx, chunk_id in enumerate(result["ids"][0] if result.get("ids") else []):
            chunk_embedding = None
            embeddings = result.get("embeddings")
            if embeddings is not None and len(embeddings) > 0 and len(embeddings[0]) > idx:
                chunk_embedding = embeddings[0][idx]
            similarity = self._cosine_similarity(query_embedding, list(chunk_embedding)) if chunk_embedding is not None else None
            items.append(
                {
                    "id": chunk_id,
                    "content": result["documents"][0][idx],
                    "metadata": result["metadatas"][0][idx],
                    "distance": result["distances"][0][idx] if result.get("distances") else None,
                    "similarity": similarity,
                }
            )
        items.sort(key=lambda item: item.get("similarity") if item.get("similarity") is not None else float("-inf"), reverse=True)
        return items

    def delete_by_document_id(self, document_id: str) -> None:
        self.collection.delete(where={"document_id": document_id})
