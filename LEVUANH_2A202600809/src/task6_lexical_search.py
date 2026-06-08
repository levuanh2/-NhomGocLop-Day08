"""
Task 6 — Lexical Search Module (BM25).

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import json
import re
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Path detection - tương tự Task 4/5
_BASE = Path(__file__).resolve().parent.parent
if not (_BASE / "src").exists() and (_BASE / "LEVUANH").exists():
    _BASE = _BASE / "LEVUANH"

INDEX_DIR = _BASE / "data" / "index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"

CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
_bm25_index = None


def _tokenize(text: str) -> list[str]:
    """Tokenize tiếng Việt đơn giản."""
    text = text.lower()
    # Giữ chữ cái (Unicode), số, dấu gạch dưới; bỏ punctuation còn lại
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return text.split()


def _is_noise(text: str) -> bool:
    """Kiểm tra content có phải noise không."""
    if len(text) < 80:
        return True
    lowered = text.lower()
    if "javascript:" in lowered:
        return True
    if "chia sẻ bài viết" in text:
        return True
    if "bình luận" in text:
        return True
    return False


def _load_corpus() -> list[dict]:
    """Load corpus từ data/index/chunks.json (đồng bộ Task 4)."""
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"chunks.json not found: {CHUNKS_PATH}")

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    corpus = []
    for chunk in chunks:
        text = chunk.get("text", "")
        if _is_noise(text):
            continue
        corpus.append({
            "content": text,
            "metadata": {
                "chunk_id": chunk.get("chunk_id", ""),
                "doc_id": chunk.get("doc_id", ""),
                "source_file": chunk.get("source_file", ""),
                "title": chunk.get("title", ""),
                "url": chunk.get("url", ""),
                "source": chunk.get("source", ""),
                "doc_type": chunk.get("doc_type", ""),
                "chunk_index": chunk.get("chunk_index", 0),
            },
        })
    return corpus


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    global _bm25_index

    if not corpus:
        return None

    try:
        from rank_bm25 import BM25Okapi

        tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
        _bm25_index = BM25Okapi(tokenized_corpus)
        return _bm25_index
    except ImportError:
        return None


def _simple_keyword_search(query: str, corpus: list[dict], top_k: int = 10) -> list[dict]:
    """Fallback simple keyword scoring nếu không có rank_bm25."""
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return []

    scored = []
    for doc in corpus:
        doc_tokens = set(_tokenize(doc["content"]))
        overlap = len(query_tokens & doc_tokens)
        if overlap > 0:
            scored.append((overlap, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, doc in scored[:top_k]:
        results.append({
            "content": doc["content"],
            "score": float(score),
            "metadata": doc["metadata"],
        })
    return results


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
    global CORPUS, _bm25_index

    if not CORPUS:
        CORPUS = _load_corpus()

    if not CORPUS:
        return []

    if _bm25_index is None:
        _bm25_index = build_bm25_index(CORPUS)

    if _bm25_index is not None:
        tokenized_query = _tokenize(query)
        import numpy as np

        scores = _bm25_index.get_scores(tokenized_query)
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
    else:
        return _simple_keyword_search(query, CORPUS, top_k=top_k)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lexical Search CLI")
    parser.add_argument("query", nargs="?", default="Điều 248 tàng trữ trái phép chất ma tuý", help="Search query")
    parser.add_argument("--top_k", type=int, default=10, help="Number of results")
    args = parser.parse_args()

    print(f"[INFO] Loading corpus from {CHUNKS_PATH}")
    CORPUS = _load_corpus()
    print(f"[INFO] Corpus size: {len(CORPUS)}")

    _bm25_index = build_bm25_index(CORPUS)
    if _bm25_index is not None:
        print("[INFO] BM25 index loaded (rank_bm25)")
    else:
        print("[WARN] rank_bm25 not installed, using simple keyword fallback")

    print(f"\nQuery: {args.query}")
    print("-" * 60)
    results = lexical_search(args.query, top_k=args.top_k)
    for i, r in enumerate(results, 1):
        title = r.get("metadata", {}).get("title", "Unknown")
        url = r.get("metadata", {}).get("url", "")
        preview = r["content"][:300].replace("\n", " ")
        print(f"\n[{i}] Score: {r['score']:.3f}")
        print(f"    Title: {title}")
        if url:
            print(f"    URL:   {url}")
        print(f"    Preview: {preview}...")
