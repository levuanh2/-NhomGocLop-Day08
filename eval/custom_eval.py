"""
Custom evaluation metrics — thay thế RAGAS để tránh conflict langchain_community.

Cài đặt 4 metrics tương đương RAGAS bằng OpenAI trực tiếp:
  - faithfulness       : câu trả lời có được hỗ trợ bởi context không?
  - answer_relevancy   : câu trả lời có liên quan đến câu hỏi không?
  - context_precision  : bao nhiêu % context được retrieve thực sự hữu ích?
  - context_recall     : bao nhiêu % ground truth được bao phủ bởi context?

Mỗi metric trả về float [0, 1].
"""

from __future__ import annotations
import os
import re
import time
from typing import Any

_OPENAI_CLIENT = None


def _client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        from openai import OpenAI
        _OPENAI_CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _OPENAI_CLIENT


def _ask(prompt: str, temperature: float = 0) -> str:
    resp = _client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=10,
    )
    return resp.choices[0].message.content.strip()


def _parse_score(text: str) -> float:
    """Trích số đầu tiên trong [0,1] từ text trả về."""
    m = re.search(r"[01](?:\.\d+)?|\d+\.\d+", text)
    if m:
        return max(0.0, min(1.0, float(m.group())))
    return 0.5


# =============================================================================
# METRIC FUNCTIONS
# =============================================================================

def faithfulness(question: str, answer: str, contexts: list[str]) -> float:
    """Câu trả lời có được hỗ trợ bởi contexts không? (0 = hoàn toàn bịa, 1 = hoàn toàn từ context)"""
    if not answer or not contexts:
        return 0.0
    ctx_text = "\n---\n".join(contexts[:5])
    prompt = f"""Đánh giá mức độ câu trả lời được hỗ trợ bởi các đoạn văn dưới đây.
Chỉ trả về một số thực duy nhất từ 0 đến 1 (0=hoàn toàn bịa đặt, 1=hoàn toàn có cơ sở).

Câu hỏi: {question}

Câu trả lời: {answer}

Các đoạn văn:
{ctx_text}

Điểm (chỉ một số từ 0 đến 1):"""
    try:
        return _parse_score(_ask(prompt))
    except Exception:
        return 0.5


def answer_relevancy(question: str, answer: str) -> float:
    """Câu trả lời có liên quan đến câu hỏi không?"""
    if not answer:
        return 0.0
    prompt = f"""Đánh giá mức độ câu trả lời giải quyết đúng câu hỏi.
Chỉ trả về một số thực duy nhất từ 0 đến 1 (0=hoàn toàn không liên quan, 1=trả lời đúng trọng tâm).

Câu hỏi: {question}
Câu trả lời: {answer}

Điểm (chỉ một số từ 0 đến 1):"""
    try:
        return _parse_score(_ask(prompt))
    except Exception:
        return 0.5


def context_precision(question: str, contexts: list[str]) -> float:
    """Tỷ lệ contexts được retrieve thực sự hữu ích để trả lời câu hỏi."""
    if not contexts:
        return 0.0
    useful = 0
    for ctx in contexts[:5]:
        prompt = f"""Đoạn văn sau có hữu ích để trả lời câu hỏi không?
Chỉ trả về 1 (có) hoặc 0 (không).

Câu hỏi: {question}
Đoạn văn: {ctx[:400]}

Trả lời (0 hoặc 1):"""
        try:
            score = _parse_score(_ask(prompt))
            useful += 1 if score >= 0.5 else 0
        except Exception:
            useful += 0.5
        time.sleep(0.2)
    return useful / len(contexts[:5])


def context_recall(question: str, ground_truth: str, contexts: list[str]) -> float:
    """Bao nhiêu % thông tin trong ground_truth được bao phủ bởi contexts?"""
    if not contexts or not ground_truth:
        return 0.0
    ctx_text = "\n---\n".join(contexts[:5])
    prompt = f"""Cho ground truth và các đoạn văn đã retrieve, hỏi xem các đoạn văn đó có chứa đủ thông tin để tái tạo ground truth không.
Chỉ trả về một số thực duy nhất từ 0 đến 1 (0=không có gì, 1=bao phủ toàn bộ).

Ground truth: {ground_truth}

Các đoạn văn:
{ctx_text}

Điểm (chỉ một số từ 0 đến 1):"""
    try:
        return _parse_score(_ask(prompt))
    except Exception:
        return 0.5


# =============================================================================
# BATCH EVALUATE
# =============================================================================

def evaluate_samples(samples: list[dict]) -> dict[str, float]:
    """
    Tính 4 metrics trung bình trên một list samples.

    Mỗi sample phải có: question, answer, contexts (list[str]), ground_truth
    Trả về dict {metric_name: avg_score}.
    """
    totals = {m: 0.0 for m in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")}
    n = len(samples)
    if n == 0:
        return totals

    for i, s in enumerate(samples, 1):
        q  = s["question"]
        a  = s.get("answer", "")
        cs = s.get("contexts", [])
        gt = s.get("ground_truth", "")

        print(f"    [{i}/{n}] evaluating: {q[:55]}...")

        totals["faithfulness"]     += faithfulness(q, a, cs)
        totals["answer_relevancy"] += answer_relevancy(q, a)
        totals["context_precision"]+= context_precision(q, cs)
        totals["context_recall"]   += context_recall(q, gt, cs)
        time.sleep(0.3)  # rate-limit

    return {m: round(v / n, 4) for m, v in totals.items()}
