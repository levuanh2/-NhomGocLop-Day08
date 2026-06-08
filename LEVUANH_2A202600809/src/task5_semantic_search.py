"""
Task 5 — Semantic Search Module.

Tim kiem ngu nghia (dense retrieval) tren vector store, tuong thich voi Task 4.
"""

import json
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from typing import Any

# Path detection - tuong tu Task 4
_BASE = Path(__file__).resolve().parent.parent
if not (_BASE / "src").exists() and (_BASE / "LEVUANH").exists():
    _BASE = _BASE / "LEVUANH"

INDEX_DIR = _BASE / "data" / "index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
METADATA_PATH = INDEX_DIR / "index_metadata.json"
FAISS_PATH = INDEX_DIR / "vector_store.faiss"
NUMPY_PATH = INDEX_DIR / "embeddings.npy"


def _is_noise_chunk(text: str) -> bool:
    """Kiem tra chunk co phai noise khong."""
    if len(text) < 80:
        return True
    lowered = text.lower()
    if "javascript:" in lowered:
        return True
    if "chia sẻ bài viết" in text:
        return True
    if "bình luận" in text:
        return True
    if "copy link" in lowered:
        return True
    return False


def _load_index_and_metadata() -> tuple[list[dict], dict, Any, str]:
    """Load chunks, metadata, vector store, va store type."""
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"chunks.json not found: {CHUNKS_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"index_metadata.json not found: {METADATA_PATH}")

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    store_type = metadata.get("vector_store", "none")
    faiss_index = None
    embeddings_matrix = None

    if store_type == "faiss":
        try:
            import faiss
            if not FAISS_PATH.exists():
                raise FileNotFoundError(f"vector_store.faiss not found: {FAISS_PATH}")
            faiss_index = faiss.read_index(str(FAISS_PATH))
        except ImportError:
            raise ImportError("FAISS not installed. Install with: pip install faiss-cpu")
    elif store_type == "numpy":
        import numpy as np
        if not NUMPY_PATH.exists():
            raise FileNotFoundError(f"embeddings.npy not found: {NUMPY_PATH}")
        embeddings_matrix = np.load(str(NUMPY_PATH))
    else:
        raise ValueError(f"Unsupported vector_store type: {store_type}")

    return chunks, metadata, faiss_index if store_type == "faiss" else embeddings_matrix, store_type


def _faiss_search(
    query_embedding: Any,
    faiss_index: Any,
    chunks: list[dict],
    top_k: int = 10,
) -> list[dict]:
    """Tim kiem su dung FAISS index."""
    import faiss
    import numpy as np

    # Normalize query embedding
    query_embedding = np.array(query_embedding, dtype="float32")
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    norms = np.linalg.norm(query_embedding, axis=1, keepdims=True)
    norms[norms == 0] = 1
    query_embedding = query_embedding / norms

    # Tim kiem top_k * 3 de co du lieu cho filter
    search_k = min(top_k * 3, len(chunks))
    scores_matrix, indices = faiss_index.search(query_embedding, search_k)

    results = []
    seen_ids = set()
    for score, idx in zip(scores_matrix[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        chunk_id = chunk.get("chunk_id", "")
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)

        text = chunk.get("text", "")
        if _is_noise_chunk(text):
            continue

        results.append({
            "content": text,
            "score": float(score),
            "metadata": {
                "chunk_id": chunk_id,
                "doc_id": chunk.get("doc_id", ""),
                "source_file": chunk.get("source_file", ""),
                "title": chunk.get("title", ""),
                "url": chunk.get("url", ""),
                "source": chunk.get("source", ""),
                "doc_type": chunk.get("doc_type", ""),
                "chunk_index": chunk.get("chunk_index", 0),
            },
        })

        if len(results) >= top_k:
            break

    return results


def _numpy_search(
    query_embedding: Any,
    embeddings_matrix: Any,
    chunks: list[dict],
    top_k: int = 10,
) -> list[dict]:
    """Tim kiem su dung numpy embeddings."""
    import numpy as np

    query_embedding = np.array(query_embedding, dtype="float32")
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # Normalize query embedding
    norms = np.linalg.norm(query_embedding, axis=1, keepdims=True)
    norms[norms == 0] = 1
    query_embedding = query_embedding / norms

    embeddings_matrix = np.array(embeddings_matrix, dtype="float32")
    if embeddings_matrix.ndim == 1:
        embeddings_matrix = embeddings_matrix.reshape(1, -1)

    # Cosine similarity: da normalized thi chi can dot product
    scores = embeddings_matrix @ query_embedding.T
    scores = scores.flatten()

    # Sort descending
    sorted_indices = np.argsort(scores)[::-1]

    results = []
    seen_ids = set()
    for idx in sorted_indices:
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        chunk_id = chunk.get("chunk_id", "")
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)

        text = chunk.get("text", "")
        if _is_noise_chunk(text):
            continue

        score = float(scores[idx])
        results.append({
            "content": text,
            "score": score,
            "metadata": {
                "chunk_id": chunk_id,
                "doc_id": chunk.get("doc_id", ""),
                "source_file": chunk.get("source_file", ""),
                "title": chunk.get("title", ""),
                "url": chunk.get("url", ""),
                "source": chunk.get("source", ""),
                "doc_type": chunk.get("doc_type", ""),
                "chunk_index": chunk.get("chunk_index", 0),
            },
        })

        if len(results) >= top_k:
            break

    return results


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tim kiem ngu nghia su dung vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            "content": str,
            "score": float,
            "metadata": dict
        }
        Sorted by score descending.
    """
    chunks, metadata, vector_store, store_type = _load_index_and_metadata()

    model_name = metadata.get("embedding_model", "BAAI/bge-m3")
    print(f"[INFO] Loading embedding model: {model_name}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    print(f"[INFO] Encoding query: {query}")
    query_embedding = model.encode([query], show_progress_bar=False)

    if store_type == "faiss":
        results = _faiss_search(query_embedding, vector_store, chunks, top_k)
    else:
        results = _numpy_search(query_embedding, vector_store, chunks, top_k)

    return results


def _print_results(results: list[dict], query: str) -> None:
    """In ket qua tim kiem de doc."""
    if not results:
        print(f"\n[NO RESULTS] Khong tim thay ket qua cho: {query}")
        return

    print(f"\n{'='*70}")
    print(f"QUERY: {query}")
    print(f"RESULTS: {len(results)}")
    print(f"{'='*70}")

    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        title = meta.get("title", "Unknown")
        url = meta.get("url", "")
        score = r.get("score", 0.0)
        content = r.get("content", "")
        preview = content[:500].replace("\n", " ")

        print(f"\n[{i}] Score: {score:.4f}")
        print(f"    Title: {title}")
        if url:
            print(f"    URL:   {url}")
        print(f"    Preview: {preview}...")

    print(f"\n{'='*70}")


def main() -> None:
    """CLI test cho semantic search."""
    import argparse

    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    parser.add_argument("query", nargs="?", default="Những nghệ sĩ nào liên quan đến ma túy?", help="Search query")
    parser.add_argument("--top_k", type=int, default=10, help="Number of results")
    args = parser.parse_args()

    query = args.query
    top_k = args.top_k

    print(f"[INFO] Semantic search initialized")
    print(f"[INFO] Index: {INDEX_DIR}")

    results = semantic_search(query, top_k=top_k)
    _print_results(results, query)


if __name__ == "__main__":
    main()
