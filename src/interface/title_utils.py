"""Conversation title helpers."""

from __future__ import annotations

import re


REMOVE_PHRASES = [
    "cho toi hoi",
    "cho tôi hỏi",
    "toi muon hoi",
    "tôi muốn hỏi",
    "hay giai thich",
    "hãy giải thích",
    "giai thich",
    "giải thích",
    "la gi",
    "là gì",
    "nhu the nao",
    "như thế nào",
    "bao nhieu nam",
    "bao nhiêu năm",
]


def normalize_title(title: str, max_length: int = 50) -> str:
    title = " ".join((title or "").strip().split())
    title = title.strip(" ?.!,:;-")
    if not title:
        return "Hoi thoai phap luat"
    if len(title) > max_length:
        title = title[: max_length - 3].rstrip() + "..."
    return title[:1].upper() + title[1:]


def fallback_title_from_query(query: str) -> str:
    title = " ".join((query or "").split())
    lowered = title.lower()
    for phrase in REMOVE_PHRASES:
        lowered = lowered.replace(phrase, "")
    lowered = re.sub(r"\s+", " ", lowered)
    return normalize_title(lowered)


def generate_chat_title(messages: list[dict]) -> str:
    first_user_message = next(
        (message for message in messages if message.get("role") == "user"),
        None,
    )
    if not first_user_message:
        return "Hội thoại mới"
    return fallback_title_from_query(first_user_message.get("content", ""))
