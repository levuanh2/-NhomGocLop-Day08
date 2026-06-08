"""Public chat entry point used by the Streamlit interface."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

def _clean_source_name(source: str) -> str:
    if not source:
        return "Nguồn tham khảo"
    name = Path(source).name
    return name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").strip() or source


def _infer_article_id(metadata: dict[str, Any], content: str) -> str:
    for key in ("article_id", "article", "dieu", "điều", "section", "demuc"):
        value = metadata.get(key)
        if value:
            return str(value)

    match = re.search(r"\b(Điều|Dieu)\s+\d+[a-zA-Z]?\b", content or "", flags=re.IGNORECASE)
    if match:
        return match.group(0)
    return ""


def _normalize_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(chunk.get("metadata") or {})
    content = chunk.get("content") or chunk.get("text") or ""
    source_file = str(metadata.get("source") or metadata.get("file") or metadata.get("filename") or "")
    source_type = str(metadata.get("type") or metadata.get("doc_type") or "legal")

    metadata.setdefault("source", source_file)
    metadata.setdefault("type", source_type)
    metadata.setdefault("chunk_index", metadata.get("chunk_index", 0))
    metadata.setdefault("url", metadata.get("url", ""))
    metadata.setdefault("title", metadata.get("title") or _clean_source_name(source_file))
    metadata.setdefault("date", metadata.get("date") or metadata.get("published_at") or "")
    metadata.setdefault("article_id", _infer_article_id(metadata, content))

    return {
        "content": content,
        "score": float(chunk.get("score", 0.0) or 0.0),
        "metadata": metadata,
        "source": chunk.get("source") or "hybrid",
    }


def _normalize_response(response: dict[str, Any], query: str) -> dict[str, Any]:
    chunks = response.get("chunks")
    if chunks is None:
        chunks = response.get("sources", [])

    out_of_scope = bool(response.get("out_of_scope", False))
    answer = response.get("answer") or (
        "Xin lỗi, hệ thống chưa tạo được câu trả lời. "
        "Bạn vui lòng thử lại với câu hỏi cụ thể hơn."
    )

    return {
        "answer": answer,
        "chunks": [] if out_of_scope else [_normalize_chunk(chunk) for chunk in chunks],
        "out_of_scope": out_of_scope,
        "query": response.get("query") or query,
    }


def chat(query: str, history: list[dict] | None = None, top_k: int = 5) -> dict:
    """Run the full RAG flow and return the BotResponse schema expected by UI."""
    from .generation import generate_with_citation

    clean_history = [
        {"role": item.get("role", ""), "content": item.get("content", "")}
        for item in (history or [])
        if item.get("role") in {"user", "assistant"} and item.get("content")
    ]
    response = generate_with_citation(query=query, history=clean_history, top_k=top_k)
    return _normalize_response(response, query)
