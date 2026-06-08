"""Formatting helpers for the LegalBot Streamlit interface."""

from __future__ import annotations


def format_source_type(source_type: str | None) -> str:
    mapping = {
        "legal": "Van ban luat",
        "news": "Bao chi",
    }
    return mapping.get((source_type or "").lower(), "Nguon tham khao")


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


def get_chunk_title(chunk: dict) -> str:
    return chunk_metadata(chunk).get("title") or "Nguon khong co tieu de"


def get_chunk_date(chunk: dict) -> str:
    return chunk_metadata(chunk).get("date") or "Khong ro ngay"


def get_article_id(chunk: dict) -> str:
    return chunk_metadata(chunk).get("article_id") or "Khong ro dieu khoan"
