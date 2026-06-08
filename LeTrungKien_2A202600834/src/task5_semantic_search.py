"""
Task 5 — Semantic Search.

Query vector store (ChromaDB) bằng cosine similarity.
Dùng cùng embedding model với Task 4: text-embedding-3-small (OpenAI, 1536 dim).

ChromaDB lưu với hnsw:space=cosine nên trả về distance ∈ [0, 2].
Score = 1 - distance → cosine similarity ∈ [-1, 1], càng cao càng liên quan.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_VECTORSTORE_DIR = Path(__file__).parent.parent / "data" / "vectorstore" / "chroma"
_EMBEDDING_MODEL = "text-embedding-3-small"

# Module-level cache để không khởi tạo lại mỗi lần gọi
_openai_client = None
_chroma_collection = None


def _get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def _get_collection():
    global _chroma_collection
    if _chroma_collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=str(_VECTORSTORE_DIR))
        _chroma_collection = client.get_collection("drug_law_docs")
    return _chroma_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng cosine similarity trên ChromaDB.

    Args:
        query: Câu truy vấn tiếng Việt
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # cosine similarity, sorted descending
            'metadata': dict     # source, type, url, chunk_index
        }
    """
    # Embed query
    response = _get_openai().embeddings.create(
        model=_EMBEDDING_MODEL,
        input=[query],
    )
    query_embedding = response.data[0].embedding

    # Query ChromaDB
    results = _get_collection().query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # ChromaDB cosine distance = 1 - cosine_similarity → invert back
    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "content": doc,
            "score": float(1.0 - dist),
            "metadata": meta,
        })

    # Already sorted ascending by distance → descending by score
    return output


if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] ({r['metadata'].get('type')}) {r['content'][:120]}")
