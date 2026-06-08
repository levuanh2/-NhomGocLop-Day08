"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. (Optional) HyDE: sinh hypothetical document từ query → dùng làm search query
    2. Chạy semantic_search + lexical_search song song (top_k * 4 candidates mỗi bên)
    3. Merge kết quả (RRF)
    4. Diversify: đảm bảo kết quả đến từ nhiều nguồn khác nhau
    5. Rerank (cross-encoder nếu có Jina API key, else sort by score)
    6. Nếu top result score < threshold → fallback sang PageIndex
    7. Return top_k results

Improvements vs v1:
  - SCORE_THRESHOLD = 0.008: phù hợp với RRF score range (max ~0.016)
  - Search pool nhân 4x thay vì 2x → nhiều candidates → tăng recall
  - Source diversity: tối đa MAX_CHUNKS_PER_SOURCE chunks / file → tránh
    toàn bộ kết quả từ 1 article
  - Query expansion: queries về định nghĩa tự động thêm "giải thích từ ngữ"
    → fix lỗi ngữ nghĩa khi tìm điều luật định nghĩa
  - HyDE (Hypothetical Document Embeddings): thay vì embed câu hỏi trực tiếp,
    sinh ra một đoạn văn bản giả định có thể trả lời câu hỏi, rồi embed đoạn đó
    → embedding gần hơn với embedding của document thật trong vector space
"""

import os
from .semantic_search import semantic_search
from .lexical_search import lexical_search, _ensure_corpus
from . import lexical_search as _t6  # Module-level access so CORPUS ref stays live after reassign
from .reranking import rerank, rerank_rrf
from .pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

# RRF score = 1/(60+rank) ≈ 0.016 for rank-1 result.
# Threshold 0.008 → fallback chỉ khi cả hai ranker đều không có signal.
SCORE_THRESHOLD = 0.008

DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # "cross_encoder" | "mmr" | "rrf"

# Số chunks tối đa lấy từ một file nguồn (tránh 5/5 từ cùng 1 article)
# = 3 → đủ để cover multi-chunk answers, nhưng vẫn đảm bảo diversity
MAX_CHUNKS_PER_SOURCE = 3


# =============================================================================
# HyDE — Hypothetical Document Embeddings
# =============================================================================

_HYDE_PROMPT = """Bạn là chuyên gia pháp luật Việt Nam.
Viết MỘT đoạn văn bản khoảng 150 từ như thể đó là trích dẫn từ văn bản luật hoặc bài báo pháp lý
trả lời trực tiếp cho câu hỏi sau. Chỉ viết đoạn văn, không giải thích thêm.

Câu hỏi: {query}"""


def _hyde_expand(query: str) -> str:
    """
    HyDE: thay vì embed câu hỏi → sinh hypothetical document → embed document đó.

    Tại sao hiệu quả hơn:
      - Câu hỏi thường ngắn, mơ hồ về mặt ngữ nghĩa (vector thưa).
      - Hypothetical document có cùng vocabulary với documents thật → embedding
        gần hơn trong vector space → cosine similarity cao hơn với đúng chunk.
      - Đặc biệt hữu ích với câu hỏi "là gì", câu hỏi về định nghĩa pháp lý.

    Fallback: trả về query gốc nếu OpenAI không khả dụng.
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": _HYDE_PROMPT.format(query=query)}],
            temperature=0,
            max_tokens=250,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠ HyDE failed ({e}), falling back to original query")
        return query


# =============================================================================
# HELPERS
# =============================================================================

_DEFINITION_KEYWORDS = ("định nghĩa", "là gì", "được hiểu", "có nghĩa là", "giải thích")
_DEFINITION_SPLITTERS = ("được định nghĩa", "định nghĩa", "là gì", "được hiểu", "có nghĩa là")


def _expand_query(query: str) -> list[str]:
    """
    Query expansion cho 2 trường hợp:
      1. Queries về định nghĩa: thêm "giải thích từ ngữ X" VÀ "X là"
         (BM25 match trực tiếp với pattern định nghĩa trong luật, ví dụ:
          "12. Cai nghiện ma túy là quá trình...")
      2. Queries liệt kê: không expand (đủ candidate từ pool lớn)
    """
    q_lower = query.lower()
    queries = [query]
    if any(kw in q_lower for kw in _DEFINITION_KEYWORDS):
        subject = query
        for splitter in _DEFINITION_SPLITTERS:
            if splitter in q_lower:
                subject = query[: q_lower.index(splitter)].strip()
                break
        if subject != query:
            # "giải thích từ ngữ X" → khớp với đầu Điều 2 của luật
            queries.append(f"giải thích từ ngữ {subject}")
            # "X là" → khớp trực tiếp với dòng định nghĩa ("X là quá trình...")
            queries.append(f"{subject} là")
    return queries


def _add_continuations(retrieved: list[dict], depth: int = 3) -> list[dict]:
    """
    Với mỗi chunk được retrieve, thêm tối đa `depth` chunk kế tiếp từ cùng source.

    depth=3 vì:
    - Điều 5 (11 hành vi nghiêm cấm) bị split qua ≥5 chunks → depth=3 lấy thêm 3 chunks → behaviors 1-4+
    - Điều 4 (4 nguồn tài chính) bị split qua chunks 12-15 → depth=3 lấy đủ tất cả 4 nguồn
    - Article bắt giữ: tên người trong chunk_index=1 → depth=1 là đủ

    Uses _t6.CORPUS (module attribute) instead of imported CORPUS to get the live reference
    after _ensure_corpus() reassigns the list.
    """
    _t6._ensure_corpus()
    corpus = _t6.CORPUS
    if not corpus:
        return retrieved

    # Build lookup: (source, chunk_index) → corpus entry
    corpus_map: dict[tuple, dict] = {}
    for entry in corpus:
        src = entry.get("metadata", {}).get("source", "")
        ci = entry.get("metadata", {}).get("chunk_index")
        if ci is not None:
            corpus_map[(src, int(ci))] = entry

    existing_keys = {
        (r.get("metadata", {}).get("source", ""), r.get("metadata", {}).get("chunk_index"))
        for r in retrieved
    }

    additions = []
    for item in retrieved:
        src = item.get("metadata", {}).get("source", "")
        ci = item.get("metadata", {}).get("chunk_index")
        if ci is None:
            continue
        score = item["score"]
        for step in range(1, depth + 1):
            next_key = (src, int(ci) + step)
            if next_key not in existing_keys and next_key in corpus_map:
                next_chunk = corpus_map[next_key].copy()
                next_chunk["source"] = "hybrid"
                next_chunk["score"] = score * (0.9 ** step)
                additions.append(next_chunk)
                existing_keys.add(next_key)
            else:
                # Stop if chunk is already present or doesn't exist
                break

    return retrieved + additions


def _diversify(candidates: list[dict], top_k: int, max_per_source: int) -> list[dict]:
    """
    Chọn top_k candidates đảm bảo diversity nguồn.
    Mỗi file nguồn tối đa max_per_source chunks, trừ khi không đủ docs khác.
    """
    source_count: dict[str, int] = {}
    selected = []
    deferred = []

    for item in candidates:
        src = item.get("metadata", {}).get("source", "unknown")
        if source_count.get(src, 0) < max_per_source:
            selected.append(item)
            source_count[src] = source_count.get(src, 0) + 1
            if len(selected) >= top_k:
                break
        else:
            deferred.append(item)

    # Nếu chưa đủ top_k, bổ sung từ deferred
    if len(selected) < top_k:
        selected.extend(deferred[: top_k - len(selected)])

    return selected


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
    use_hyde: bool = False,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ (Optional) HyDE: sinh hypothetical document → dùng làm search query
          ├→ Query expansion
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Diversify (max MAX_CHUNKS_PER_SOURCE per source file)
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results (RRF scale)
        use_reranking: Có áp dụng reranking hay không
        use_hyde: Dùng HyDE (Hypothetical Document Embeddings) cho semantic search

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 0 (Optional): HyDE — sinh hypothetical document để embed thay cho query
    semantic_query = _hyde_expand(query) if use_hyde else query

    # Step 1: Query expansion (dùng query gốc để BM25 khớp từ khoá chính xác)
    queries = _expand_query(query)
    search_k = top_k * (4 if len(queries) > 1 else 2)

    # Semantic search: dùng semantic_query (HyDE doc hoặc query gốc)
    # HyDE doc có cùng vocabulary với documents thật → cosine similarity cao hơn
    dense_results = []
    try:
        r = semantic_search(semantic_query, top_k=search_k)
        dense_results.extend(r)
    except Exception as e:
        print(f"  ⚠ Semantic search failed: {e}")

    # Lexical search: dùng expanded queries (keyword match, không cần HyDE)
    sparse_results = []
    for q in queries:
        try:
            r = lexical_search(q, top_k=search_k)
            sparse_results.extend(r)
        except Exception as e:
            print(f"  ⚠ Lexical search failed: {e}")

    # Step 2: Merge bằng RRF
    ranked_lists = [r for r in [dense_results, sparse_results] if r]
    if ranked_lists:
        merged = rerank_rrf(ranked_lists, top_k=search_k)
    else:
        merged = []

    for item in merged:
        item["source"] = "hybrid"

    # Step 3: Diversify sources — tối đa MAX_CHUNKS_PER_SOURCE / file
    diverse = _diversify(merged, top_k=search_k, max_per_source=MAX_CHUNKS_PER_SOURCE)

    # Step 4: Rerank
    if use_reranking and diverse:
        try:
            final_results = rerank(query, diverse, top_k=top_k, method=RERANK_METHOD)
        except Exception as e:
            print(f"  ⚠ Reranking failed: {e}")
            final_results = diverse[:top_k]
    else:
        final_results = diverse[:top_k]

    # Ensure source field is set
    for item in final_results:
        item.setdefault("source", "hybrid")

    # Step 5: Check threshold → fallback PageIndex
    best_score = final_results[0]["score"] if final_results else 0.0
    if not final_results or best_score < score_threshold:
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback
        except Exception:
            pass  # PageIndex unavailable — silently continue

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
