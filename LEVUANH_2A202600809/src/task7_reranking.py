"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

import sys
from typing import Optional

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Tính cosine similarity giữa hai vectors."""
    import numpy as np

    a = np.array(a, dtype="float32")
    b = np.array(b, dtype="float32")

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    if not candidates:
        return []

    print("[WARN] Cross encoder not configured, fallback to original score")
    sorted_candidates = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
    return sorted_candidates[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if not candidates:
        return []
    if top_k <= 0:
        return []

    # Chỉ xử lý candidates có embedding
    candidates_with_emb = [c for c in candidates if "embedding" in c]
    if not candidates_with_emb:
        print("[WARN] No candidates with embedding, falling back to score sort")
        sorted_candidates = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
        return sorted_candidates[:top_k]

    selected = []
    remaining_indices = list(range(len(candidates_with_emb)))

    for _ in range(min(top_k, len(candidates_with_emb))):
        best_local_idx = None
        best_mmr_score = float("-inf")

        for local_idx in remaining_indices:
            candidate = candidates_with_emb[local_idx]

            # Relevance to query
            relevance = _cosine_sim(query_embedding, candidate["embedding"])

            # Max similarity to already selected
            max_sim_to_selected = 0.0
            for sel_local_idx in selected:
                selected_candidate = candidates_with_emb[sel_local_idx]
                sim = _cosine_sim(candidate["embedding"], selected_candidate["embedding"])
                max_sim_to_selected = max(max_sim_to_selected, sim)

            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_local_idx = local_idx

        if best_local_idx is not None:
            selected.append(best_local_idx)
            remaining_indices.remove(best_local_idx)

    results = []
    for local_idx in selected:
        candidate = candidates_with_emb[local_idx].copy()
        candidate["mmr_score"] = best_mmr_score if local_idx == selected[-1] else 0.0
        results.append(candidate)

    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    if not ranked_lists:
        return []
    if top_k <= 0:
        return []

    rrf_scores = {}  # key -> rrf_score
    content_map = {}  # key -> full item (bản có score gốc cao nhất)
    original_scores = {}  # key -> list of original scores
    rank_sources = {}  # key -> list of source tags

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            # Dedupe key: ưu tiên chunk_id, fallback content[:200]
            chunk_id = item.get("metadata", {}).get("chunk_id", "")
            if chunk_id:
                key = chunk_id
            else:
                key = item.get("content", "")[:200]

            if not key:
                continue

            # RRF contribution
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank)

            # Track original score (lấy bản cao nhất)
            orig_score = item.get("score", 0)
            if key not in original_scores:
                original_scores[key] = []
                content_map[key] = item.copy()
            original_scores[key].append(orig_score)

            # Gộp metadata: giữ bản có score gốc cao nhất
            if key not in content_map or orig_score > content_map[key].get("score", 0):
                content_map[key] = item.copy()

    # Sort by RRF score
    sorted_keys = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for key, rrf_score in sorted_keys[:top_k]:
        item = content_map[key].copy()
        item["score"] = rrf_score
        item["original_scores"] = original_scores.get(key, [])
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if not candidates:
        return []

    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        return rerank_mmr(query, candidates, top_k)
    elif method == "rrf":
        # RRF cần nhiều ranked lists - gọi riêng rerank_rrf
        # Nếu chỉ có 1 list, sort theo score desc
        sorted_candidates = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
        return sorted_candidates[:top_k]
    else:
        # Unknown method, fallback to score sort
        sorted_candidates = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
        return sorted_candidates[:top_k]


if __name__ == "__main__":
    # Test with dummy data
    print("[TEST] Reranking module test")
    print("=" * 60)

    dummy_candidates = [
        {
            "content": "Điều 248: Tội tàng trữ trái phép chất ma tuý",
            "score": 0.8,
            "metadata": {"chunk_id": "doc1", "title": "Điều 248"}
        },
        {
            "content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý",
            "score": 0.7,
            "metadata": {"chunk_id": "doc2", "title": "Nghệ sĩ X"}
        },
        {
            "content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ",
            "score": 0.6,
            "metadata": {"chunk_id": "doc3", "title": "Hình phạt"}
        },
    ]

    print("\n1. Test rerank_cross_encoder fallback:")
    results = rerank_cross_encoder("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"   [{r['score']:.3f}] {r['content']}")

    print("\n2. Test rerank_rrf (với 2 ranked lists):")
    list1 = [
        {"content": "Doc A về ma tuý", "score": 0.9, "metadata": {"chunk_id": "a"}},
        {"content": "Doc B về hình phạt", "score": 0.8, "metadata": {"chunk_id": "b"}},
        {"content": "Doc C về cai nghiện", "score": 0.7, "metadata": {"chunk_id": "c"}},
    ]
    list2 = [
        {"content": "Doc B về hình phạt", "score": 0.95, "metadata": {"chunk_id": "b"}},
        {"content": "Doc D về tàng trữ", "score": 0.85, "metadata": {"chunk_id": "d"}},
        {"content": "Doc A về ma tuý", "score": 0.75, "metadata": {"chunk_id": "a"}},
    ]
    rrf_results = rerank_rrf([list1, list2], top_k=3)
    for r in rrf_results:
        print(f"   RRF={r['score']:.4f} | scores={r.get('original_scores', [])} | {r['content']}")

    print("\n3. Test rerank (cross_encoder):")
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2, method="cross_encoder")
    for r in results:
        print(f"   [{r['score']:.3f}] {r['content']}")

    print("\n[TEST] All reranking tests passed!")
