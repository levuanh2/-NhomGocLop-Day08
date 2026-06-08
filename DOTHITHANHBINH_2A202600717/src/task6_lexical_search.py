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

from functools import lru_cache
from pathlib import Path

CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chromadb"
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _load_corpus() -> list[dict]:
    try:
        import chromadb

        collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(
            "DrugLawDocs"
        )
        data = collection.get(include=["documents", "metadatas"])
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        if documents:
            return [
                {"content": content, "metadata": metadata or {}}
                for content, metadata in zip(documents, metadatas)
            ]
    except Exception:
        pass

    corpus = []
    base_path = str(STANDARDIZED_DIR)
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        path = str(md_file)
        doc_type = "legal" if "legal" in path[len(base_path):] else "news"
        corpus.append({
            "content": content,
            "metadata": {"source": md_file.name, "doc_type": doc_type},
        })
    return corpus


@lru_cache(maxsize=1)
def _get_bm25_index():
    global CORPUS

    if not CORPUS:
        CORPUS = _load_corpus()
    return build_bm25_index(CORPUS)


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi

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
    if top_k <= 0:
        return []

    bm25 = _get_bm25_index()
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(
        range(len(scores)),
        key=lambda idx: scores[idx],
        reverse=True,
    )[:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": score,
                "metadata": CORPUS[idx]["metadata"],
            })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
