"""
A/B comparison — So sánh 2 configs của LegalBot RAG pipeline.

Config A (Baseline):  top_k=3, không reranking, không query expansion
Config B (Optimized): top_k=5, reranking Jina, query expansion bật

Kết quả ghi ra:
    eval/results_config_A_scores.json
    eval/results_config_B_scores.json
    eval/ab_comparison.csv        ← bảng so sánh từng câu hỏi
    eval/ab_summary.json          ← tóm tắt A vs B theo metric

Chạy:
    cd e:\\AI_in_action\\-NhomGocLop-Day08
    set PYTHONPATH=src
    python eval/ab_compare.py
    python eval/ab_compare.py --skip-eval   # chỉ tạo báo cáo từ kết quả đã có
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
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

CONFIGS = {
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

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


# =============================================================================
# HELPERS
# =============================================================================

def load_golden_dataset() -> list[dict]:
    path = ROOT / "eval" / "golden_dataset.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [q for q in data if q["category"] != "out_of_scope"]


def run_pipeline_for_config(question: str, config: dict) -> dict:
    from rag_pipeline.retrieval_pipeline import retrieve
    from rag_pipeline.generation import generate_with_citation

    top_k = config["top_k"]
    use_reranking = config["use_reranking"]
    try:
        chunks = retrieve(question, top_k=top_k, use_reranking=use_reranking)
        result = generate_with_citation(question, top_k=top_k)
        return {
            "answer": result.get("answer", ""),
            "contexts": [c["content"] for c in chunks],
        }
    except Exception as e:
        print(f"    [WARN] {e}")
        return {"answer": "", "contexts": []}


def evaluate_config(config_name: str, golden: list[dict]) -> tuple[list[dict], dict]:
    """Chạy pipeline + custom eval cho một config, trả về (samples, scores)."""
    sys.path.insert(0, str(ROOT / "eval"))
    from custom_eval import evaluate_samples

    config = CONFIGS[config_name]
    print(f"\n{'='*60}")
    print(f"  Running {config['name']}")
    print(f"  {config['description']}")
    print(f"{'='*60}")

    samples = []
    for i, item in enumerate(golden, 1):
        q = item["question"]
        print(f"  [{i}/{len(golden)}] {q[:65]}...")
        res = run_pipeline_for_config(q, config)
        samples.append({
            "id": item["id"],
            "category": item["category"],
            "question": q,
            "answer": res["answer"],
            "contexts": res["contexts"] or ["(no context)"],
            "ground_truth": item["ground_truth"],
        })
        time.sleep(0.5)

    print(f"\n  Evaluating {len(samples)} samples with LLM judge...")
    eval_samples = [
        {k: v for k, v in s.items() if k in ("question", "answer", "contexts", "ground_truth")}
        for s in samples
    ]
    scores = evaluate_samples(eval_samples)

    # Lưu scores
    scores_path = ROOT / "eval" / f"results_config_{config_name}_scores.json"
    scores_path.write_text(
        json.dumps({"config": config_name, "metrics": scores}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  Scores saved → {scores_path.name}")
    for m, v in scores.items():
        bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
        print(f"    {m:<25} {bar}  {v:.4f}")

    return samples, scores


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_ab_report(
    samples_a: list[dict],
    samples_b: list[dict],
    scores_a: dict,
    scores_b: dict,
) -> None:
    import csv

    # ── ab_comparison.csv ──────────────────────────────────────────────────────
    rows = []
    for sa, sb in zip(samples_a, samples_b):
        rows.append({
            "id": sa["id"],
            "category": sa["category"],
            "question": sa["question"][:120],
            "answer_A": sa["answer"][:200],
            "answer_B": sb["answer"][:200],
            "contexts_A": len(sa["contexts"]),
            "contexts_B": len(sb["contexts"]),
        })

    csv_path = ROOT / "eval" / "ab_comparison.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nComparison CSV → {csv_path.name}")

    # ── ab_summary.json ────────────────────────────────────────────────────────
    summary = {
        "configs": {k: v["description"] for k, v in CONFIGS.items()},
        "metrics": {},
        "winner": {},
    }
    for metric in METRICS:
        a_val = scores_a.get(metric, 0.0)
        b_val = scores_b.get(metric, 0.0)
        diff = b_val - a_val
        summary["metrics"][metric] = {
            "A": round(a_val, 4),
            "B": round(b_val, 4),
            "diff_B_minus_A": round(diff, 4),
            "improvement_pct": round(diff / max(a_val, 0.001) * 100, 1),
        }
        summary["winner"][metric] = "B" if b_val > a_val else ("A" if a_val > b_val else "tie")

    overall_a = sum(scores_a.get(m, 0) for m in METRICS) / len(METRICS)
    overall_b = sum(scores_b.get(m, 0) for m in METRICS) / len(METRICS)
    summary["overall"] = {
        "A": round(overall_a, 4),
        "B": round(overall_b, 4),
        "winner": "B" if overall_b > overall_a else "A",
    }

    summary_path = ROOT / "eval" / "ab_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Summary JSON    → {summary_path.name}")

    # ── Print summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  A/B COMPARISON SUMMARY")
    print("=" * 70)
    print(f"  {'Metric':<25} {'Config A':>10} {'Config B':>10} {'Δ (B-A)':>10}  Winner")
    print("  " + "-" * 65)
    for metric in METRICS:
        info = summary["metrics"][metric]
        winner = "✓ B" if info["winner"] == "B" else ("✓ A" if info["winner"] == "A" else "tie")
        print(f"  {metric:<25} {info['A']:>10.4f} {info['B']:>10.4f} {info['diff_B_minus_A']:>+10.4f}  {winner}")
    print("  " + "-" * 65)
    print(f"  {'OVERALL':<25} {overall_a:>10.4f} {overall_b:>10.4f} {overall_b-overall_a:>+10.4f}  {'✓ B' if overall_b > overall_a else '✓ A'}")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main(skip_eval: bool = False) -> None:
    golden = load_golden_dataset()
    print(f"Loaded {len(golden)} evaluation questions.")

    scores_a_path = ROOT / "eval" / "results_config_A_scores.json"
    scores_b_path = ROOT / "eval" / "results_config_B_scores.json"

    if skip_eval and scores_a_path.exists() and scores_b_path.exists():
        print("Using existing score files (--skip-eval).")
        scores_a = json.loads(scores_a_path.read_text(encoding="utf-8"))["metrics"]
        scores_b = json.loads(scores_b_path.read_text(encoding="utf-8"))["metrics"]
        samples_a = [{"id": q["id"], "category": q["category"], "question": q["question"],
                      "answer": "", "contexts": [], "ground_truth": q["ground_truth"]} for q in golden]
        samples_b = samples_a[:]
    else:
        samples_a, scores_a = evaluate_config("A", golden)
        samples_b, scores_b = evaluate_config("B", golden)

    generate_ab_report(samples_a, samples_b, scores_a, scores_b)
    print("\nA/B comparison complete. Run generate_report.py to create the full report.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-eval", action="store_true",
                        help="Bỏ qua bước evaluate, dùng kết quả đã có")
    args = parser.parse_args()
    main(skip_eval=args.skip_eval)
