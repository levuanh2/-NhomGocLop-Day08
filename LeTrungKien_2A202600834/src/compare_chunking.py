"""
So sánh các chiến lược chunking trên tập dữ liệu thực tế.

Chạy: conda run -n ai20k python src/compare_chunking.py

Ba chiến lược:
  1. recursive        — RecursiveCharacterTextSplitter (baseline)
  2. markdown_header  — MarkdownHeaderTextSplitter (tận dụng cấu trúc heading)
  3. semantic         — SemanticChunker (OpenAI API, tốn tiền, chạy cuối)

Mỗi chiến lược được đánh giá qua:
  - Tổng số chunk
  - Phân phối kích thước (min / trung bình / max / median)
  - % chunk trong giới hạn 800 ký tự
  - Thống kê theo loại tài liệu (legal vs news)
  - 2 mẫu chunk để xem chất lượng phân đoạn
"""

import statistics
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

load_dotenv()

CLEANED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHUNK_SIZE   = 800
CHUNK_OVERLAP = 100


# =============================================================================
# Load documents
# =============================================================================

def load_docs() -> list[dict]:
    docs = []
    for fp in sorted(CLEANED_DIR.rglob("*.md")):
        doc_type = "legal" if "legal" in str(fp) else "news"
        docs.append({"content": fp.read_text(encoding="utf-8"),
                     "name": fp.name, "type": doc_type})
    return docs


# =============================================================================
# Strategy 1 — RecursiveCharacterTextSplitter
# =============================================================================

def chunk_recursive(docs: list[dict]) -> list[dict]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in docs:
        for i, text in enumerate(splitter.split_text(doc["content"])):
            chunks.append({"content": text, "source": doc["name"],
                           "type": doc["type"], "index": i})
    return chunks


# =============================================================================
# Strategy 2 — MarkdownHeaderTextSplitter + fallback RecursiveCharacter
# =============================================================================

def chunk_markdown_header(docs: list[dict]) -> list[dict]:
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    # Tách theo heading trước, sau đó fallback nếu section quá dài
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    fallback = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in docs:
        sections = header_splitter.split_text(doc["content"])
        chunk_index = 0
        for section in sections:
            text = section.page_content
            if len(text) <= CHUNK_SIZE:
                chunks.append({"content": text, "source": doc["name"],
                                "type": doc["type"], "index": chunk_index})
                chunk_index += 1
            else:
                for sub in fallback.split_text(text):
                    chunks.append({"content": sub, "source": doc["name"],
                                   "type": doc["type"], "index": chunk_index})
                    chunk_index += 1
    return chunks


# =============================================================================
# Strategy 3 — SemanticChunker (OpenAI, tốn API)
# =============================================================================

def chunk_semantic(docs: list[dict]) -> list[dict]:
    import os
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    semantic_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85,
    )
    fallback = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in docs:
        print(f"    semantic chunking: {doc['name']}")
        chunk_index = 0
        for sc in semantic_splitter.split_text(doc["content"]):
            if len(sc) <= CHUNK_SIZE:
                chunks.append({"content": sc, "source": doc["name"],
                                "type": doc["type"], "index": chunk_index})
                chunk_index += 1
            else:
                for sub in fallback.split_text(sc):
                    chunks.append({"content": sub, "source": doc["name"],
                                   "type": doc["type"], "index": chunk_index})
                    chunk_index += 1
    return chunks


# =============================================================================
# Đánh giá
# =============================================================================

def evaluate(chunks: list[dict]) -> dict:
    sizes = [len(c["content"]) for c in chunks]
    by_type = {"legal": [], "news": []}
    for c in chunks:
        by_type[c["type"]].append(len(c["content"]))

    return {
        "total":        len(chunks),
        "min":          min(sizes),
        "max":          max(sizes),
        "mean":         round(statistics.mean(sizes)),
        "median":       round(statistics.median(sizes)),
        "within_limit": sum(1 for s in sizes if s <= CHUNK_SIZE),
        "within_pct":   round(sum(1 for s in sizes if s <= CHUNK_SIZE) / len(sizes) * 100, 1),
        "legal_count":  len(by_type["legal"]),
        "news_count":   len(by_type["news"]),
        "legal_mean":   round(statistics.mean(by_type["legal"])) if by_type["legal"] else 0,
        "news_mean":    round(statistics.mean(by_type["news"]))  if by_type["news"]  else 0,
        "size_buckets": {
            "<200":    sum(1 for s in sizes if s < 200),
            "200-400": sum(1 for s in sizes if 200 <= s < 400),
            "400-600": sum(1 for s in sizes if 400 <= s < 600),
            "600-800": sum(1 for s in sizes if 600 <= s <= 800),
            ">800":    sum(1 for s in sizes if s > 800),
        },
    }


def print_report(name: str, chunks: list[dict], stats: dict):
    W = 62
    print(f"\n{'=' * W}")
    print(f"  Chiến lược: {name}")
    print(f"{'=' * W}")
    print(f"  Tổng chunk    : {stats['total']:>6}  (legal {stats['legal_count']} / news {stats['news_count']})")
    print(f"  Kích thước    : min={stats['min']}  avg={stats['mean']}  median={stats['median']}  max={stats['max']} chars")
    print(f"  Trong 800 ký  : {stats['within_limit']}/{stats['total']}  ({stats['within_pct']}%)")
    print(f"  Trung bình    : legal={stats['legal_mean']} chars  |  news={stats['news_mean']} chars")
    print(f"\n  Phân phối kích thước:")
    total = stats["total"]
    for label, count in stats["size_buckets"].items():
        bar = "█" * int(count / total * 40)
        print(f"    {label:>8} chars: {bar:<40} {count:>4} ({count/total*100:4.1f}%)")

    # 2 mẫu chunk: 1 từ legal, 1 từ news
    print(f"\n  Mẫu chunk (legal):")
    legal_samples = [c for c in chunks if c["type"] == "legal"]
    if legal_samples:
        sample = legal_samples[len(legal_samples) // 3]  # lấy đoạn giữa để tránh header/footer
        preview = sample["content"].replace("\n", "↵ ")[:300]
        print(f"    [{len(sample['content'])} chars] {sample['source']}")
        print(f"    {preview}{'…' if len(sample['content']) > 300 else ''}")

    print(f"\n  Mẫu chunk (news):")
    news_samples = [c for c in chunks if c["type"] == "news"]
    if news_samples:
        sample = news_samples[len(news_samples) // 3]
        preview = sample["content"].replace("\n", "↵ ")[:300]
        print(f"    [{len(sample['content'])} chars] {sample['source']}")
        print(f"    {preview}{'…' if len(sample['content']) > 300 else ''}")


def print_summary(results: dict[str, dict]):
    print(f"\n\n{'=' * 62}")
    print("  TỔNG KẾT SO SÁNH")
    print(f"{'=' * 62}")
    header = f"  {'Chiến lược':<20} {'Chunks':>7} {'Avg':>6} {'Median':>7} {'≤800':>6}"
    print(header)
    print(f"  {'-'*20} {'-'*7} {'-'*6} {'-'*7} {'-'*6}")
    for name, s in results.items():
        print(f"  {name:<20} {s['total']:>7} {s['mean']:>6} {s['median']:>7} {s['within_pct']:>5}%")

    print("""
  Nhận xét:
  ┌─ recursive       : Đơn giản, nhanh, đảm bảo 100% chunk ≤ 800 chars.
  │                    Tuy nhiên có thể cắt giữa câu/đoạn văn pháp luật.
  │
  ├─ markdown_header : Giữ nguyên cấu trúc Điều/Khoản cho văn bản pháp lý.
  │                    Tốt nhất khi legal docs có heading markdown rõ ràng.
  │                    Nếu legal dùng plain text (win32com), kém hiệu quả hơn.
  │
  └─ semantic        : Tách tại ranh giới ngữ nghĩa tự nhiên, chunk có context
                       ngữ nghĩa tốt nhất. Chi phí API cao hơn, chậm hơn.
                       Phù hợp nhất cho retrieval chính xác.

  Khuyến nghị cho bộ dữ liệu này:
    • Legal docs  → markdown_header (nếu có heading) hoặc recursive (an toàn)
    • News docs   → semantic (câu báo chí ngắn, ranh giới sự kiện rõ)
    • Tổng thể    → semantic + fallback recursive (đang dùng) là lựa chọn tốt nhất
    """)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import sys

    print("Đang tải documents từ data/cleaned/...")
    docs = load_docs()
    print(f"→ {len(docs)} documents ({sum(1 for d in docs if d['type']=='legal')} legal, "
          f"{sum(1 for d in docs if d['type']=='news')} news)\n")

    run_semantic = "--semantic" in sys.argv or "--all" in sys.argv
    results = {}

    strategies: list[tuple[str, Callable]] = [
        ("recursive",       chunk_recursive),
        ("markdown_header", chunk_markdown_header),
    ]
    if run_semantic:
        strategies.append(("semantic", chunk_semantic))

    for name, fn in strategies:
        print(f"Đang chunk với '{name}'...")
        chunks = fn(docs)
        stats  = evaluate(chunks)
        print_report(name, chunks, stats)
        results[name] = stats

    if not run_semantic:
        print(f"\n  [semantic] Bỏ qua (gọi API tốn tiền).")
        print(f"  Để chạy semantic: python src/compare_chunking.py --semantic")

    print_summary(results)
