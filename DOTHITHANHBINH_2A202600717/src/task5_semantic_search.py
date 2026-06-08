"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

from functools import lru_cache
from pathlib import Path

from sentence_transformers import SentenceTransformer

try:
    from .task4_chunking_indexing import EMBEDDING_MODEL
except ImportError:
    from task4_chunking_indexing import EMBEDDING_MODEL


CHROMA_DIR = Path(__file__).parent.parent / "data" / "chromadb"
COLLECTION_NAME = "DrugLawDocs"


@lru_cache(maxsize=1)
def _get_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _get_collection():
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if top_k <= 0:
        return []

    query_embedding = _get_model().encode(query).tolist()
    results = _get_collection().query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    hits = [
        {
            "content": content,
            "score": 1.0 - float(distance),
            "metadata": metadata or {},
        }
        for content, metadata, distance in zip(documents, metadatas, distances)
    ]
    hits.sort(key=lambda item: item["score"], reverse=True)
    return hits



if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
