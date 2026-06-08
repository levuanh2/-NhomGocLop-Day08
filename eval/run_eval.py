"""
Evaluation pipeline cho LegalBot RAG — dùng custom LLM-based evaluation.

Metrics (tương đương RAGAS, không cần langchain_community):
    - faithfulness:        Câu trả lời có bịa đặt ngoài context không?
    - answer_relevancy:    Câu trả lời có đúng với câu hỏi không?
    - context_precision:   Chunks retrieve có chứa thông tin cần thiết không?
    - context_recall:      Có bỏ sót chunks quan trọng không?

Chạy:
    cd e:\\AI_in_action\\-NhomGocLop-Day08
    set PYTHONPATH=src
    python eval/run_eval.py
    python eval/run_eval.py --config A   # chỉ chạy config cụ thể
    python eval/run_eval.py --output eval/results_full.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")


# =============================================================================
# RETRIEVE + GENERATE cho một câu hỏi
# =============================================================================

def run_pipeline(question: str, config: dict) -> dict:
    """
    Chạy pipeline RAG với config cho trước, trả về answer + contexts.

    Returns:
        {
            "answer": str,
            "contexts": list[str],   # nội dung các chunks đã retrieve
            "sources": list[str],    # tên file nguồn
        }
    """
    from rag_pipeline.retrieval_pipeline import retrieve
    from rag_pipeline.generation import generate_with_citation

    top_k = config.get("top_k", 5)
    use_reranking = config.get("use_reranking", True)

    try:
        chunks = retrieve(question, top_k=top_k, use_reranking=use_reranking)
        result = generate_with_citation(question, top_k=top_k)
        return {
            "answer": result.get("answer", ""),
            "contexts": [c["content"] for c in chunks],
            "sources": [c.get("metadata", {}).get("source", "") for c in chunks],
        }
    except Exception as e:
        print(f"  [WARN] Pipeline failed for question '{question[:50]}': {e}")
        return {"answer": "", "contexts": [], "sources": []}


# =============================================================================
# MAIN
# =============================================================================

def main(config_name: str = "B", output_path: str | None = None) -> None:
    from custom_eval import evaluate_samples

    configs = {
        "A": {
            "name": "Config A — Baseline",
            "top_k": 3,
            "use_reranking": False,
            "description": "top_k=3, không reranking, không query expansion",
        },
        "B": {
            "name": "Config B — Optimized",
            "top_k": 5,
            "use_reranking": True,
            "description": "top_k=5, reranking Jina, query expansion bật",
        },
    }

    dataset_path = ROOT / "eval" / "golden_dataset.json"
    golden = json.loads(dataset_path.read_text(encoding="utf-8"))

    # Lọc bỏ out_of_scope — không có ground truth để evaluate
    eval_items = [q for q in golden if q["category"] != "out_of_scope"]
    print(f"Loaded {len(eval_items)} evaluation questions (excluding out-of-scope).")

    config = configs.get(config_name.upper(), configs["B"])
    print(f"\nRunning evaluation with: {config['name']}")
    print(f"  {config['description']}\n")

    samples = []
    for i, item in enumerate(eval_items, 1):
        q = item["question"]
        print(f"  [{i}/{len(eval_items)}] {q[:70]}...")
        pipeline_result = run_pipeline(q, config)
        samples.append({
            "question": q,
            "answer": pipeline_result["answer"],
            "contexts": pipeline_result["contexts"] or ["(no context retrieved)"],
            "ground_truth": item["ground_truth"],
        })
        time.sleep(0.5)  # tránh rate limit

    print("\nRunning LLM-based evaluation...")
    scores = evaluate_samples(samples)

    print("\n" + "=" * 60)
    print(f"  Evaluation Results — {config['name']}")
    print("=" * 60)
    for metric, score in scores.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {metric:<25} {bar}  {score:.4f}")
    print("=" * 60)

    # Ghi kết quả ra CSV
    if output_path is None:
        output_path = ROOT / "eval" / f"results_config_{config_name.upper()}.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)

    import csv
    rows = []
    for item, sample in zip(eval_items, samples):
        rows.append({
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "answer": sample["answer"][:300],
            "ground_truth": item["ground_truth"][:300],
            "contexts_count": len(sample["contexts"]),
            "config": config_name.upper(),
        })

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    scores_path = output_path.with_name(output_path.stem + "_scores.json")
    scores_path.write_text(
        json.dumps({"config": config_name.upper(), "metrics": scores}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nResults saved to: {output_path}")
    print(f"Scores saved to:  {scores_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-based evaluation for LegalBot")
    parser.add_argument("--config", default="B", choices=["A", "B"],
                        help="Config to evaluate: A (baseline) or B (optimized)")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()
    main(config_name=args.config, output_path=args.output)
