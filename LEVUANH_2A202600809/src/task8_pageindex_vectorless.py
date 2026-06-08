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
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")

# Path detection
_BASE = Path(__file__).resolve().parent.parent
if not (_BASE / "src").exists() and (_BASE / "LEVUANH").exists():
    _BASE = _BASE / "LEVUANH"

STANDARDIZED_DIR = _BASE / "data" / "standardized"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.

    Returns:
        int: Số lượng file đã upload
    """
    if not PAGEINDEX_API_KEY:
        print("[WARN] PAGEINDEX_API_KEY not set. Skipping upload.")
        return 0

    try:
        from pageindex import PageIndex
    except ImportError:
        print("[WARN] pageindex package not installed. Skipping upload.")
        print("  Install with: pip install pageindex")
        return 0

    if not STANDARDIZED_DIR.exists():
        print(f"[WARN] Standardized directory not found: {STANDARDIZED_DIR}")
        return 0

    md_files = sorted(STANDARDIZED_DIR.rglob("*.md"))
    if not md_files:
        print(f"[INFO] No markdown files found in {STANDARDIZED_DIR}")
        return 0

    try:
        pi = PageIndex(api_key=PAGEINDEX_API_KEY)
        uploaded = 0

        for md_file in md_files:
            try:
                content = md_file.read_text(encoding="utf-8")
                # Thử upload với metadata
                pi.upload(
                    content=content,
                    metadata={
                        "filename": md_file.name,
                        "type": md_file.parent.name,
                        "path": str(md_file.relative_to(_BASE)),
                    },
                )
                uploaded += 1
                print(f"  [UPLOADED] {md_file.name}")
            except Exception as e:
                print(f"  [WARN] Cannot upload {md_file.name}: {type(e).__name__}: {e}")

        print(f"[INFO] Uploaded {uploaded}/{len(md_files)} files")
        return uploaded

    except Exception as e:
        print(f"[WARN] Cannot initialize PageIndex: {type(e).__name__}: {e}")
        return 0


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
        print("[WARN] PAGEINDEX_API_KEY not set. PageIndex search skipped.")
        return []

    try:
        from pageindex import PageIndex
    except ImportError:
        print("[WARN] pageindex package not installed. PageIndex search skipped.")
        print("  Install with: pip install pageindex")
        return []

    try:
        pi = PageIndex(api_key=PAGEINDEX_API_KEY)
        results = pi.query(query=query, top_k=top_k)

        formatted = []
        for r in results:
            formatted.append({
                "content": getattr(r, "text", str(r)),
                "score": float(getattr(r, "score", 0.0)),
                "metadata": getattr(r, "metadata", {}),
                "source": "pageindex",
            })
        return formatted

    except Exception as e:
        print(f"[WARN] PageIndex search failed: {type(e).__name__}: {e}")
        return []


if __name__ == "__main__":
    print(f"[INFO] PAGEINDEX_API_KEY: {'set' if PAGEINDEX_API_KEY else 'not set'}")
    print(f"[INFO] Standardized dir: {STANDARDIZED_DIR}")

    if not PAGEINDEX_API_KEY:
        print("\n[WARN] Missing PAGEINDEX_API_KEY.")
        print("  Set in .env file or export PAGEINDEX_API_KEY=<key>")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("\n[INFO] Uploading documents...")
        upload_documents()

        print("\n[INFO] Test query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            preview = r["content"][:100].replace("\n", " ")
            print(f"  [{r['score']:.3f}] {preview}...")
