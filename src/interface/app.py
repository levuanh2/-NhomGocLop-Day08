"""Streamlit entry point for the LegalBot interface."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
for path in (CURRENT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chat_store import load_chats
from layout import render_main_layout
from mock_data import mock_chat
from state import (
    build_history_for_pipeline,
    create_message,
    ensure_active_chat,
    init_session_state,
)
from title_utils import generate_chat_title


def load_css() -> None:
    css_path = CURRENT_DIR / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def validate_prompt(prompt: str) -> tuple[bool, str]:
    prompt = (prompt or "").strip()
    if not prompt:
        return False, "Vui lòng nhập câu hỏi."
    if len(prompt) < 3:
        return False, "Câu hỏi quá ngắn."
    if len(prompt) > 2000:
        return False, "Câu hỏi quá dài. Vui lòng rút gọn."
    return True, ""


def get_pipeline_response(prompt: str, history: list[dict]) -> dict:
    if st.session_state.settings.get("use_mock", True):
        return mock_chat(prompt, history)

    try:
        from rag_pipeline.pipeline import chat

        return chat(prompt, history)
    except Exception as exc:  # noqa: BLE001 - UI must degrade cleanly for demo.
        st.session_state.debug_error = repr(exc)
        return {
            "answer": (
                "Xin lỗi, hệ thống đang gặp lỗi khi xử lý câu hỏi. "
                "Bạn vui lòng thử lại sau."
            ),
            "chunks": [],
            "out_of_scope": False,
            "query": prompt,
        }


def handle_submit(prompt: str) -> None:
    prompt = prompt.strip()
    is_valid, error_message = validate_prompt(prompt)
    if not is_valid:
        st.warning(error_message)
        return

    ensure_active_chat()
    st.session_state.messages.append(create_message("user", prompt))
    history = build_history_for_pipeline(st.session_state.messages)

    with st.spinner("LegalBot đang truy vấn nguồn và tạo câu trả lời..."):
        st.session_state.is_generating = True
        response = get_pipeline_response(prompt, history)
        st.session_state.is_generating = False

    chunks = response.get("chunks") or []
    if st.session_state.settings.get("top_k"):
        chunks = chunks[: st.session_state.settings["top_k"]]
    out_of_scope = bool(response.get("out_of_scope", False))
    answer = response.get("answer") or (
        "Xin lỗi, hệ thống chưa tạo được câu trả lời. "
        "Bạn vui lòng thử lại với câu hỏi cụ thể hơn."
    )

    st.session_state.last_response = {
        "answer": answer,
        "chunks": chunks,
        "out_of_scope": out_of_scope,
        "query": prompt,
    }
    st.session_state.last_chunks = [] if out_of_scope else chunks

    if st.session_state.current_chat_title == "Hội thoại mới":
        st.session_state.current_chat_title = generate_chat_title(st.session_state.messages)
    if st.session_state.current_chat_title == "Hoi thoai moi":
        st.session_state.current_chat_title = generate_chat_title(st.session_state.messages)

    st.session_state.pending_stream_response = {
        "answer": answer,
        "chunks": [] if out_of_scope else chunks,
        "out_of_scope": out_of_scope,
    }
    st.session_state.chats = load_chats()
    st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="LegalBot",
        page_icon="⚖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_css()
    init_session_state()
    st.session_state.chats = load_chats()

    prompt = render_main_layout()
    if prompt:
        handle_submit(prompt)


if __name__ == "__main__":
    main()
