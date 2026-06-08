"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

import sys

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Flexible import hỗ trợ cả relative và direct script
try:
    from .task5_semantic_search import semantic_search
    from .task6_lexical_search import lexical_search
    from .task7_reranking import rerank, rerank_rrf
    from .task8_pageindex_vectorless import pageindex_search
except ImportError:
    from task5_semantic_search import semantic_search
    from task6_lexical_search import lexical_search
    from task7_reranking import rerank, rerank_rrf
    from task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.01  # RRF score thường nhỏ nên dùng threshold thấp
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"  # "rrf" làm mặc định, fallback cross_encoder nếu cần


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold OR empty:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: Chạy semantic + lexical song song (thực thi tuần tự, nhưng tách biệt)
    dense_results = semantic_search(query, top_k=top_k * 3)
    sparse_results = lexical_search(query, top_k=top_k * 3)

    # Step 2: Merge bằng RRF
    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)

    # Gắn source = hybrid
    for item in merged:
        item["source"] = "hybrid"
        item["metadata"]["retrieval_source"] = "hybrid"

    # Step 3: Rerank
    if use_reranking and merged:
        if RERANK_METHOD == "rrf":
            # RRF đã merge ở trên, sort theo score desc
            final_results = merged[:top_k]
        else:
            # Fallback sang cross_encoder hoặc mmr
            final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
            for item in final_results:
                item["source"] = "hybrid"
    else:
        final_results = merged[:top_k]

    # Step 4: Check threshold → fallback PageIndex nếu rỗng hoặc score thấp
    if not final_results:
        print(f"  [INFO] Hybrid results empty, falling back to PageIndex")
        fallback = pageindex_search(query, top_k=top_k)
        return fallback

    if final_results[0]["score"] < score_threshold:
        print(f"  [INFO] Best hybrid score ({final_results[0]['score']:.4f}) < threshold ({score_threshold})")
        print(f"  [INFO] Falling back to PageIndex")
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback

    return final_results[:top_k]


if __name__ == "__main__":
    print("[INFO] Retrieval Pipeline")
    print("=" * 70)

    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Miu Lê bị phát hiện sử dụng chất gì?",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            title = r.get("metadata", {}).get("title", "Unknown")
            url = r.get("metadata", {}).get("url", "")
            preview = r["content"][:80].replace("\n", " ")
            print(f"  {i}. Score: {r['score']:.4f} | Source: {r.get('source', '?')}")
            print(f"     Title: {title}")
            if url:
                print(f"     URL:   {url}")
            print(f"     Preview: {preview}...")
