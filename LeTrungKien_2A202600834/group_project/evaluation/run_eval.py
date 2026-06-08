"""
Diagnostic evaluation on new_test.json (10 test cases).

Chạy:
    python group_project/evaluation/run_eval.py

Đánh giá 3 chiều theo ragas_dimension:
  - context_recall   : retrieved chunks có chứa source_doc đúng không?
  - faithfulness     : answer có chứa thông tin từ ground_truth không?
  - answer_relevance : answer có trả lời đúng câu hỏi không?
"""

import json
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import generate_with_citation, format_context, reorder_for_llm

TEST_FILE = Path(__file__).parent / "new_test.json"
RESULTS_FILE = Path(__file__).parent / "eval_results.json"


def check_context_recall(retrieved: list[dict], source_docs: list[str]) -> dict:
    """Kiểm tra xem source_doc đúng có trong retrieved context không."""
    retrieved_sources = [r.get("metadata", {}).get("source", "") for r in retrieved]
    retrieved_content = " ".join(r["content"] for r in retrieved)

    hits = []
    for doc in source_docs:
        # Match by filename (without path)
        doc_stem = doc.replace(".md", "").lower()
        found_in_meta = any(doc_stem in s.lower() for s in retrieved_sources)
        found_in_content = doc_stem[:20] in retrieved_content.lower()
        hits.append(found_in_meta or found_in_content)

    recall = sum(hits) / len(hits) if hits else 0.0
    return {
        "recall": recall,
        "source_hits": {doc: hit for doc, hit in zip(source_docs, hits)},
        "retrieved_sources": retrieved_sources,
    }


def check_answer_coverage(answer: str, ground_truth: str) -> float:
    """
    Heuristic: tính % key terms từ ground_truth xuất hiện trong answer.
    Đây là proxy đơn giản cho faithfulness / relevance — không phải LLM judge.
    """
    # Lấy các token quan trọng (>3 ký tự) từ ground_truth
    gt_tokens = set(
        t.lower().strip(",.;:()[]\"'")
        for t in ground_truth.split()
        if len(t) > 3
    )
    if not gt_tokens:
        return 0.0
    answer_lower = answer.lower()
    covered = sum(1 for t in gt_tokens if t in answer_lower)
    return covered / len(gt_tokens)


def run_evaluation():
    test_cases = json.loads(TEST_FILE.read_text(encoding="utf-8"))
    print(f"Loaded {len(test_cases)} test cases\n")

    results = []
    dimension_scores = {"context_recall": [], "faithfulness": [], "answer_relevance": []}

    for tc in test_cases:
        qid = tc["id"]
        question = tc["question"]
        ground_truth = tc["ground_truth"]
        source_docs = [s.strip() for s in tc["source_doc"].split(",")]
        dimension = tc["ragas_dimension"]

        print(f"{'='*70}")
        print(f"[{qid:02d}] [{dimension}] {question}")
        print(f"{'='*70}")

        # Step 1: Retrieve
        try:
            retrieved = retrieve(question, top_k=5)
        except Exception as e:
            print(f"  ⚠ Retrieval failed: {e}")
            retrieved = []

        recall_info = check_context_recall(retrieved, source_docs)
        print(f"  Context recall : {recall_info['recall']:.2f}")
        print(f"  Retrieved from : {recall_info['retrieved_sources']}")
        for doc, hit in recall_info["source_hits"].items():
            mark = "✓" if hit else "✗"
            print(f"  {mark} {doc}")

        # Step 2: Generate
        try:
            gen = generate_with_citation(question, top_k=5)
            answer = gen["answer"]
        except Exception as e:
            print(f"  ⚠ Generation failed: {e}")
            answer = ""

        coverage = check_answer_coverage(answer, ground_truth)
        print(f"  GT coverage    : {coverage:.2f}")
        print(f"\n  Answer (first 300 chars):\n  {answer[:300]}")
        print(f"\n  Ground truth   :\n  {ground_truth[:300]}")

        score = {
            "context_recall": recall_info["recall"],
            "gt_coverage": coverage,
        }
        dimension_scores[dimension].append(coverage)

        results.append({
            "id": qid,
            "dimension": dimension,
            "question": question,
            "source_docs": source_docs,
            "context_recall": recall_info["recall"],
            "retrieved_sources": recall_info["retrieved_sources"],
            "source_hits": recall_info["source_hits"],
            "gt_coverage": coverage,
            "answer": answer,
            "ground_truth": ground_truth,
        })
        print()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    overall_recall = sum(r["context_recall"] for r in results) / len(results)
    overall_coverage = sum(r["gt_coverage"] for r in results) / len(results)
    print(f"  Avg context recall (source found) : {overall_recall:.3f}")
    print(f"  Avg ground-truth token coverage   : {overall_coverage:.3f}")

    print("\n  Per-dimension GT coverage:")
    for dim, scores in dimension_scores.items():
        if scores:
            avg = sum(scores) / len(scores)
            print(f"    {dim:<20} : {avg:.3f}  ({len(scores)} cases)")

    print("\n  Worst cases (by GT coverage):")
    worst = sorted(results, key=lambda x: x["gt_coverage"])[:3]
    for r in worst:
        print(f"    [{r['id']:02d}] [{r['dimension']}] {r['gt_coverage']:.3f} — {r['question'][:60]}")

    print("\n  Best cases (by GT coverage):")
    best = sorted(results, key=lambda x: x["gt_coverage"], reverse=True)[:3]
    for r in best:
        print(f"    [{r['id']:02d}] [{r['dimension']}] {r['gt_coverage']:.3f} — {r['question'][:60]}")

    # Save
    RESULTS_FILE.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  Full results saved → {RESULTS_FILE}")

    return results


if __name__ == "__main__":
    run_evaluation()
