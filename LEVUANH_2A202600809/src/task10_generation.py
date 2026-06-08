"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

# Flexible import hỗ trợ cả relative và direct script
try:
    from .task9_retrieval_pipeline import retrieve
except ImportError:
    from task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return chunks

    reordered = []
    n = len(chunks)

    # Odd positions: 0, 2, 4, ... → đầu
    for i in range(0, n, 2):
        reordered.append(chunks[i])

    # Even positions reversed: n-1, n-3, ... → cuối
    for i in range(n - 1, 0, -2):
        reordered.append(chunks[i])

    return reordered


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        title = meta.get("title", f"Document {i}")
        source = meta.get("source", "unknown")
        url = meta.get("url", "")
        doc_type = meta.get("doc_type", "unknown")
        score = chunk.get("score", 0)

        header = f"[Document {i} | Title: {title} | Source: {source}"
        if url:
            header += f" | URL: {url}"
        header += f" | Type: {doc_type} | Score: {score:.4f}]"

        context_parts.append(f"{header}\n{chunk['content']}")

    return "\n---\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def _extract_fallback_answer(query: str, chunks: list[dict]) -> dict:
    """Tạo extractive answer fallback khi không có OpenAI API key."""
    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    reordered = reorder_for_llm(chunks)
    lines = []
    sources = []

    for i, chunk in enumerate(reordered[:TOP_K], 1):
        meta = chunk.get("metadata", {})
        title = meta.get("title", "Unknown")
        source = meta.get("source", "unknown")
        url = meta.get("url", "")

        content = chunk.get("content", "")
        preview = content[:400].replace("\n", " ")

        citation = f"[{title} | {source}]"
        if url:
            citation = f"[{title} | {source} | {url}]"

        lines.append(f"{i}. {preview}... {citation}")
        sources.append({
            "title": title,
            "source": source,
            "url": url,
            "content": content[:500],
        })

    answer = (
        f"Dựa trên các nguồn tìm được, tôi có thể trả lời câu hỏi như sau:\n\n"
        + "\n\n".join(lines)
        + "\n\n(Lưu ý: Đây là câu trả lời extractive, được tạo tự động do không có OpenAI API key.)"
    )

    return {
        "answer": answer,
        "sources": sources,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
    }


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k)

    # Step 2: Check empty
    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    # Step 3: Reorder
    reordered = reorder_for_llm(chunks)

    # Step 4: Format context
    context = format_context(reordered)

    # Step 5: Call LLM nếu có API key
    if not OPENAI_API_KEY:
        return _extract_fallback_answer(query, chunks)

    # Call OpenAI
    try:
        client_kwargs = {"api_key": OPENAI_API_KEY}
        if OPENAI_BASE_URL:
            client_kwargs["base_url"] = OPENAI_BASE_URL

        from openai import OpenAI

        client = OpenAI(**client_kwargs)

        user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )

        answer = response.choices[0].message.content

        sources = []
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            sources.append({
                "title": meta.get("title", ""),
                "source": meta.get("source", ""),
                "url": meta.get("url", ""),
                "content": chunk.get("content", "")[:500],
            })

        return {
            "answer": answer,
            "sources": sources,
            "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
        }

    except Exception as e:
        print(f"[WARN] LLM call failed: {type(e).__name__}: {e}")
        print("[INFO] Falling back to extractive answer")
        return _extract_fallback_answer(query, chunks)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generation with Citation CLI")
    parser.add_argument("query", nargs="?", default="Những nghệ sĩ nào liên quan tới ma túy?", help="Query")
    parser.add_argument("--top_k", type=int, default=TOP_K, help="Number of chunks")
    args = parser.parse_args()

    print(f"[INFO] OPENAI_API_KEY: {'set' if OPENAI_API_KEY else 'not set'}")
    if not OPENAI_API_KEY:
        print("[INFO] Will use extractive fallback answer")

    print(f"\nQuery: {args.query}")
    print("=" * 70)

    result = generate_with_citation(args.query, top_k=args.top_k)

    print(f"\nAnswer:\n{result['answer']}")
    print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")

    if result["sources"]:
        print("\n--- Sources ---")
        for i, s in enumerate(result["sources"], 1):
            print(f"  {i}. {s.get('title', 'Unknown')} | {s.get('source', '?')}")
