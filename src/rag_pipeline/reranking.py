"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: flashrank (ONNX local, không cần torch/torchvision)
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Lựa chọn:
    - flashrank + ms-marco-MiniLM-L-12-v2 (~34 MB ONNX, CPU-only, không cần GPU)
      → pip install flashrank
    - Fallback về sort by score nếu flashrank chưa cài
    - rerank_rrf() dùng để merge nhiều ranked lists trong Task 9
"""

from pathlib import Path

# Cache model tại thư mục gốc project để tránh re-download
_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "models" / "flashrank"

# Model nhẹ nhất có chất lượng tốt (~34 MB ONNX)
# Đổi thành "ms-marco-MultiBERT-L-12" để hỗ trợ tiếng Việt tốt hơn (~400 MB)
_FLASHRANK_MODEL = "ms-marco-MiniLM-L-12-v2"

_ranker = None  # lazy init — chỉ load khi lần đầu gọi rerank


def _get_ranker():
    global _ranker
    if _ranker is None:
        from flashrank import Ranker
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _ranker = Ranker(model_name=_FLASHRANK_MODEL, cache_dir=str(_CACHE_DIR))
    return _ranker


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates dùng flashrank (ONNX cross-encoder, chạy local CPU).

    Model ms-marco-MiniLM-L-12-v2: ~34 MB, không cần torch/GPU/API key.
    Fallback: sort by original score nếu flashrank chưa được cài.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    try:
        from flashrank import RerankRequest
        ranker = _get_ranker()
        passages = [{"id": i, "text": c["content"]} for i, c in enumerate(candidates)]
        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)
        return [
            {**candidates[r["id"]], "score": float(r["score"])}
            for r in results[:top_k]
        ]
    except ImportError:
        print("  [!] flashrank not installed (pip install flashrank), falling back to score sort")
    except Exception as e:
        print(f"  [!] flashrank reranker failed ({e}), falling back to score sort")

    return sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k]


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
    import numpy as np

    def cosine_sim(a, b):
        a, b = np.array(a), np.array(b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    selected = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])
            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sim = cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
                max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]


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
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = score
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
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        raise NotImplementedError("Call rerank_mmr with query_embedding")
    elif method == "rrf":
        # Wrap single list as a list-of-lists for RRF
        return rerank_rrf([candidates], top_k=top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
