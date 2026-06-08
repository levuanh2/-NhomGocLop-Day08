"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")

STANDARDIZED_DIR = _DATA_DIR / "standardized"


def upload_documents():
    """
    Upload toàn bộ PDF/DOCX documents lên PageIndex.
    PageIndexClient.submit_document nhận file_path (PDF), không nhận raw text.
    """
    from pageindex import PageIndexClient

    pi = PageIndexClient(api_key=PAGEINDEX_API_KEY)

    for pdf_file in STANDARDIZED_DIR.rglob("*.pdf"):
        resp = pi.submit_document(file_path=str(pdf_file))
        print(f"  ✓ Submitted: {pdf_file.name} -> doc_id={resp.get('doc_id')}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if not PAGEINDEX_API_KEY:
        raise ValueError("PAGEINDEX_API_KEY chưa được set trong .env")

    import time
    from pageindex import PageIndexClient

    pi = PageIndexClient(api_key=PAGEINDEX_API_KEY)

    # Lấy danh sách tất cả documents đã upload
    docs_resp = pi.list_documents(limit=50)
    documents = docs_resp.get("documents", [])
    if not documents:
        return []

    # Submit query cho từng document, thu thập retrieval_id
    retrieval_ids = []
    for doc in documents:
        doc_id = doc.get("id") or doc.get("doc_id")
        if not doc_id:
            continue
        try:
            r = pi.submit_query(doc_id=doc_id, query=query)
            retrieval_ids.append((doc_id, r.get("retrieval_id"), doc))
        except Exception:
            continue

    # Poll results
    output = []
    for doc_id, retrieval_id, doc_meta in retrieval_ids:
        if not retrieval_id:
            continue
        for _ in range(10):
            ret = pi.get_retrieval(retrieval_id=retrieval_id)
            status = ret.get("status", "")
            if status in ("completed", "done", "success"):
                chunks = ret.get("results") or ret.get("chunks") or []
                for i, chunk in enumerate(chunks[:top_k]):
                    text = chunk.get("content") or chunk.get("text") or str(chunk)
                    score = float(chunk.get("score", 1.0 - i * 0.05))
                    output.append({
                        "content": text,
                        "score": score,
                        "metadata": {**doc_meta, **chunk.get("metadata", {})},
                        "source": "pageindex",
                    })
                break
            elif status in ("failed", "error"):
                break
            time.sleep(1)

    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
