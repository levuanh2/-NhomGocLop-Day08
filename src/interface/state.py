"""Session state helpers for LegalBot."""

from __future__ import annotations

from uuid import uuid4

import streamlit as st

from chat_store import create_new_chat, now_iso


DEFAULT_SETTINGS = {
    "model": "claude-sonnet-4-6",
    "top_k": 5,
    "show_debug": False,
    "show_scores": True,
    "show_source_file": False,
    "use_mock": True,
}


def init_session_state() -> None:
    defaults = {
        "chat_id": None,
        "current_chat_title": "Hội thoại mới",
        "messages": [],
        "last_chunks": [],
        "last_response": None,
        "selected_message_id": None,
        "is_generating": False,
        "settings": DEFAULT_SETTINGS.copy(),
        "pending_example": None,
        "pending_stream_response": None,
        "pending_action": None,
        "new_name": "",
        "last_action": None,
        "debug_error": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_current_chat() -> None:
    st.session_state.chat_id = None
    st.session_state.current_chat_title = "Hội thoại mới"
    st.session_state.messages = []
    st.session_state.last_chunks = []
    st.session_state.last_response = None
    st.session_state.selected_message_id = None
    st.session_state.debug_error = ""


def ensure_active_chat() -> str:
    if not st.session_state.chat_id:
        st.session_state.chat_id = create_new_chat()
    return st.session_state.chat_id


def create_message(
    role: str,
    content: str,
    chunks: list[dict] | None = None,
    out_of_scope: bool = False,
) -> dict:
    message = {
        "id": f"msg_{uuid4().hex[:10]}",
        "role": role,
        "content": content,
        "created_at": now_iso(),
    }
    if role == "assistant":
        message["chunks"] = chunks or []
        message["out_of_scope"] = out_of_scope
    return message


def build_history_for_pipeline(messages: list[dict], limit: int = 10) -> list[dict]:
    return [
        {"role": message.get("role", ""), "content": message.get("content", "")}
        for message in messages[-limit:]
        if message.get("role") in {"user", "assistant"}
    ]


def find_last_assistant_message(messages: list[dict]) -> dict | None:
    return next(
        (message for message in reversed(messages) if message.get("role") == "assistant"),
        None,
    )
