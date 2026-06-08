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

from pathlib import Path
import unicodedata


def _resolve_data_dir() -> Path:
    import os

    env_dir = os.getenv("RAG_DATA_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)

    repo_root = Path(__file__).resolve().parents[2]
    candidates = [Path(__file__).parent.parent / "data"]
    candidates.extend(sorted(repo_root.glob("*/data")))
    for candidate in candidates:
        if (candidate / "standardized").exists() and (candidate / "vectorstore" / "chroma").exists():
            return candidate
    for candidate in candidates:
        if (candidate / "standardized").exists():
            return candidate
    return candidates[0]


_DATA_DIR = _resolve_data_dir()
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
    """Fallback: load .md files từ data/standardized/ và chia thành chunk nhỏ."""
    search_dir = _STANDARDIZED_DIR
    corpus = []
    for md_file in sorted(search_dir.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        for chunk_index, chunk_text in enumerate(_chunk_markdown_text(content)):
            corpus.append({
                "content": chunk_text,
                "metadata": {
                    "source": md_file.name,
                    "type": doc_type,
                    "chunk_index": chunk_index,
                    "title": md_file.stem,
                },
            })
    return corpus


def _chunk_markdown_text(text: str, max_chars: int = 1800, overlap: int = 180) -> list[str]:
    """Chia fallback markdown thành chunk vừa đủ để không làm nổ context window."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            start = 0
            while start < len(paragraph):
                end = start + max_chars
                chunks.append(paragraph[start:end].strip())
                if end >= len(paragraph):
                    break
                start = max(0, end - overlap)
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current.strip())
            prefix = current[-overlap:].strip() if overlap and current else ""
            current = f"{prefix}\n\n{paragraph}" if prefix else paragraph

    if current:
        chunks.append(current.strip())
    return chunks


def _ensure_corpus():
    global CORPUS, _bm25
    if CORPUS:
        return
    try:
        CORPUS = _load_corpus_from_chroma()
    except Exception:
        CORPUS = _load_corpus_from_files()
    _bm25 = build_bm25_index(CORPUS)


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi

    # Tokenize — normalize dấu để câu hỏi không dấu vẫn khớp văn bản tiếng Việt.
    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn").replace("đ", "d")


def _tokenize(text: str) -> list[str]:
    return _normalize_text(text).split()


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

    tokenized_query = _tokenize(query)
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
