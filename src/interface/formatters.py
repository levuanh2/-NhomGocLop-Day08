"""Formatting helpers for the LegalBot Streamlit interface."""

from __future__ import annotations

import re
from pathlib import Path


def format_source_type(source_type: str | None) -> str:
    mapping = {
        "legal": "Văn bản luật",
        "news": "Báo chí",
    }
    return mapping.get((source_type or "").lower(), "Nguồn tham khảo")


def format_score(score: float | int | None) -> str:
    if score is None:
        return "N/A"
    value = float(score)
    if value <= 1:
        value *= 100
    value = max(0, min(value, 100))
    return f"{value:.0f}%"


def score_percent(score: float | int | None) -> int:
    if score is None:
        return 0
    value = float(score)
    if value <= 1:
        value *= 100
    return int(max(0, min(value, 100)))


def preview_text(text: str | None, limit: int = 200) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def chunk_metadata(chunk: dict) -> dict:
    return chunk.get("metadata") or {}


def _clean_source_name(source: str | None) -> str:
    if not source:
        return "Nguồn không có tiêu đề"
    name = Path(str(source)).name
    return name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").strip() or str(source)


def get_chunk_title(chunk: dict) -> str:
    metadata = chunk_metadata(chunk)
    return (
        metadata.get("title")
        or metadata.get("document_title")
        or metadata.get("doc_title")
        or _clean_source_name(metadata.get("source") or metadata.get("file") or metadata.get("filename"))
    )


def get_chunk_date(chunk: dict) -> str:
    metadata = chunk_metadata(chunk)
    return metadata.get("date") or metadata.get("published_at") or metadata.get("created_at") or "Không rõ ngày"


def get_article_id(chunk: dict) -> str:
    metadata = chunk_metadata(chunk)
    for key in ("article_id", "article", "dieu", "điều", "section", "demuc"):
        value = metadata.get(key)
        if value:
            return str(value)
    content = chunk.get("content") or chunk.get("text") or ""
    match = re.search(r"\b(Điều|Dieu)\s+\d+[a-zA-Z]?\b", content, flags=re.IGNORECASE)
    if match:
        return match.group(0)
    return "Không rõ điều khoản"
