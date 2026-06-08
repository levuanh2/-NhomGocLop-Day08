"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DATA_DIR = Path(os.getenv("DATA_DIR", str(_PROJECT_ROOT / "LeTrungKien_2A202600834" / "data")))
_STANDARDIZED_DIR = _DATA_DIR / "standardized"
_VECTORSTORE_DIR = _DATA_DIR / "vectorstore" / "chroma"

CORPUS: list[dict] = []
_bm25 = None


def _load_corpus_from_chroma() -> list[dict]:
    """Load chunks từ ChromaDB (cùng data với semantic search)."""
    import chromadb
    client = chromadb.PersistentClient(path=str(_VECTORSTORE_DIR))
    collection = client.get_collection("drug_law_docs")
    results = collection.get(include=["documents", "metadatas"])
    return [
        {"content": doc, "metadata": meta}
        for doc, meta in zip(results["documents"], results["metadatas"])
    ]


def _load_corpus_from_files() -> list[dict]:
    """Fallback: load toàn bộ .md files từ data/standardized/."""
    search_dir = _STANDARDIZED_DIR
    corpus = []
    for md_file in sorted(search_dir.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        corpus.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type},
        })
    return corpus


def _ensure_corpus():
    global CORPUS, _bm25
    if CORPUS:
        return
    try:
        CORPUS = _load_corpus_from_chroma()
    except Exception:
        CORPUS = _load_corpus_from_files()
    if CORPUS:
        _bm25 = build_bm25_index(CORPUS)


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi

    # Tokenize — simple whitespace split đủ dùng cho tiếng Việt
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    import numpy as np

    _ensure_corpus()

    if _bm25 is None:
        return []

    tokenized_query = query.lower().split()
    scores = _bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": CORPUS[idx]["metadata"],
            })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
