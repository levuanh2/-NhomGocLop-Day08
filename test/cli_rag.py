"""
CLI test cho RAG Pipeline — dùng data từ LeTrungKien/data/vectorstore.

Chạy (dùng conda env ai20k):
    python test/cli_rag.py            # chế độ full (retrieve + generate)
    python test/cli_rag.py --retrieve # chỉ xem kết quả retrieve, không gọi LLM
"""

import sys
import io
import json
import argparse
from datetime import datetime
from pathlib import Path

# Fix Windows console encoding (tránh crash khi pipeline print emoji)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Setup paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

VECTORSTORE = ROOT / "data" / "vectorstore" / "chroma"

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import rag_pipeline.semantic_search as _sem
import rag_pipeline.lexical_search as _lex


# ── Session Memory ────────────────────────────────────────────────────────────

SESSIONS_DIR = ROOT / "sessions"


class Session:
    """Quản lý memory hội thoại trong một session, lưu ra file JSON."""

    def __init__(self):
        SESSIONS_DIR.mkdir(exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = SESSIONS_DIR / f"{self.session_id}.json"
        self.messages: list[dict] = []  # {"role": "user"|"assistant", "content": str}
        self._save()  # tạo file ngay khi khởi tạo

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self._save()

    def _save(self) -> None:
        data = {
            "session_id": self.session_id,
            "created_at": self.session_id,
            "message_count": len(self.messages),
            "messages": self.messages,
        }
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Helpers hiển thị ───────────────────────────────────────────────────────────

DIVIDER = "─" * 70
BOLD_DIVIDER = "═" * 70


def _type_badge(doc_type: str) -> str:
    return "📜 Luật" if doc_type == "legal" else "📰 Báo"


def print_chunks(chunks: list[dict]) -> None:
    if not chunks:
        print("  (Không tìm thấy kết quả)\n")
        return

    for i, r in enumerate(chunks, 1):
        meta = r.get("metadata", {})
        source = meta.get("source", "?")
        doc_type = meta.get("type", "?")
        score = r.get("score", 0.0)
        via = r.get("source", "hybrid")
        url = meta.get("url", "")
        article_id = meta.get("article_id", "")

        # Header của chunk
        badge = _type_badge(doc_type)
        print(f"\n  [{i}] {badge}  {source}  |  score={score:.4f}  |  via={via}")
        if article_id:
            print(f"       Điều khoản: {article_id}")
        if url:
            print(f"       URL: {url}")

        # Nội dung preview (250 ký tự)
        content_preview = r["content"].replace("\n", " ")[:250]
        suffix = "…" if len(r["content"]) > 250 else ""
        print(f"       \"{content_preview}{suffix}\"")

    print()


def print_answer(result: dict) -> None:
    print(f"\n{BOLD_DIVIDER}")
    print("  TRẢ LỜI")
    print(BOLD_DIVIDER)
    print(result["answer"])
    print(f"\n  [Dùng {len(result['chunks'])} chunks]")
    print(BOLD_DIVIDER)


# ── Chế độ retrieve-only ───────────────────────────────────────────────────────

def run_retrieve_mode() -> None:
    from rag_pipeline.retrieval_pipeline import retrieve

    print(f"\n{BOLD_DIVIDER}")
    print("  LEGALBOT — Retrieve-Only Mode")
    print(f"  Vectorstore: {VECTORSTORE}")
    print(f"{BOLD_DIVIDER}")
    print("  Nhập câu hỏi để xem các chunks được retrieve.")
    print("  Gõ 'q' hoặc Enter trống để thoát.\n")

    while True:
        try:
            query = input("Câu hỏi > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nThoát.")
            break

        if not query or query.lower() == "q":
            break

        print(f"\n{DIVIDER}")
        print(f"  Đang retrieve cho: \"{query}\"")
        print(DIVIDER)

        try:
            chunks = retrieve(query, top_k=5)
            print_chunks(chunks)
        except Exception as e:
            print(f"  Lỗi: {e}\n")


# ── Chế độ full RAG ────────────────────────────────────────────────────────────

def run_full_mode() -> None:
    from rag_pipeline.generation import generate_with_citation

    session = Session()

    print(f"\n{BOLD_DIVIDER}")
    print("  LEGALBOT — Full RAG Mode (Retrieve + Generate)")
    print(f"  Vectorstore: {VECTORSTORE}")
    print(f"  Model: gpt-4o-mini")
    print(f"  Session: {session.session_id}  →  {session.path}")
    print(f"{BOLD_DIVIDER}")
    print("  Chatbot chỉ trả lời về pháp luật và các vụ vi phạm pháp luật.")
    print("  Gõ 'q' hoặc Enter trống để thoát.\n")

    while True:
        try:
            query = input("Câu hỏi > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nThoát.")
            break

        if not query or query.lower() == "q":
            break

        print(f"\n{DIVIDER}")
        print("  Đang retrieve + generate…")
        print(DIVIDER)

        try:
            result = generate_with_citation(query, history=session.messages)

            if result["out_of_scope"]:
                print(f"\n  [Ngoài phạm vi] {result['answer']}\n")
            else:
                print("\n  SOURCES (chunks đã retrieve):")
                print_chunks(result["chunks"])
                print_answer(result)

            # Lưu lượt hội thoại vào session
            session.add("user", query)
            session.add("assistant", result["answer"])

        except Exception as e:
            print(f"  Lỗi: {e}")
            print("  (Thử chạy lại với --retrieve nếu chưa có OPENAI_API_KEY)\n")

        print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLI test cho RAG Pipeline pháp luật"
    )
    parser.add_argument(
        "--retrieve",
        action="store_true",
        help="Chỉ chạy bước retrieve, không gọi LLM (không cần OPENAI_API_KEY cho generation)",
    )
    args = parser.parse_args()

    if args.retrieve:
        run_retrieve_mode()
    else:
        run_full_mode()


if __name__ == "__main__":
    main()
