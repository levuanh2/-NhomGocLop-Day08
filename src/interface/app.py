"""Streamlit entry point for the LegalBot interface."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
for path in (CURRENT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chat_store import load_chats, save_chat
from layout import render_main_layout
from mock_data import mock_chat
from state import (
    build_history_for_pipeline,
    create_message,
    ensure_active_chat,
    init_session_state,
)
from title_utils import generate_chat_title


def load_env_file() -> None:
    env_path = SRC_DIR.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        clean_key = key.strip()
        clean_value = value.strip().strip('"').strip("'")
        if clean_value and not os.environ.get(clean_key):
            os.environ[clean_key] = clean_value


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
    if st.session_state.settings.get("use_mock", False):
        return mock_chat(prompt, history)

    try:
        load_env_file()
        from rag_pipeline.generation import generate_with_citation

        return generate_with_citation(
            query=prompt,
            history=history,
            top_k=st.session_state.settings.get("top_k", 5),
        )
    except Exception as exc:  # noqa: BLE001 - UI must degrade cleanly for demo.
        st.session_state.debug_error = repr(exc)
        error_text = repr(exc)
        if "OPENAI_API_KEY" in error_text or "api_key" in error_text:
            answer = (
                "RAG pipeline chưa được cấu hình `OPENAI_API_KEY`, nên hệ thống chưa thể "
                "truy vấn embedding/generation để tạo câu trả lời. Vui lòng cấu hình API key "
                "rồi gửi lại câu hỏi."
            )
        elif "chromadb" in error_text or "rank_bm25" in error_text:
            answer = (
                "RAG pipeline thiếu dependency truy xuất dữ liệu (`chromadb` hoặc `rank_bm25`). "
                "Vui lòng cài dependencies của pipeline rồi gửi lại câu hỏi."
            )
        elif "drug_law_docs" in error_text or "vectorstore" in error_text:
            answer = (
                "RAG pipeline chưa tìm thấy vector store/corpus dữ liệu pháp luật để truy xuất. "
                "Vui lòng tạo hoặc trỏ đúng thư mục index dữ liệu rồi gửi lại câu hỏi."
            )
        else:
            answer = (
                "RAG pipeline đang gặp lỗi khi xử lý câu hỏi. Bật `Debug` trong phần "
                "Thông số hệ thống để xem lỗi kỹ thuật."
            )
        return {
            "answer": answer,
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
    history = build_history_for_pipeline(st.session_state.messages)
    st.session_state.messages.append(create_message("user", prompt))

    if st.session_state.current_chat_title == "Hội thoại mới":
        st.session_state.current_chat_title = generate_chat_title(st.session_state.messages)
    if st.session_state.current_chat_title == "Hoi thoai moi":
        st.session_state.current_chat_title = generate_chat_title(st.session_state.messages)

    save_chat(
        st.session_state.chat_id,
        st.session_state.current_chat_title,
        st.session_state.messages,
    )
    st.session_state.pending_rag_request = {
        "chat_id": st.session_state.chat_id,
        "chat_title": st.session_state.current_chat_title,
        "query": prompt,
        "history": history,
    }
    st.session_state.chats = load_chats()
    st.rerun()


def handle_pending_rag_request() -> None:
    pending = st.session_state.get("pending_rag_request")
    if not pending:
        return

    prompt = pending["query"]
    history = pending.get("history", [])
    target_chat_id = pending.get("chat_id")
    target_chat_title = pending.get("chat_title") or st.session_state.current_chat_title

    st.session_state.is_generating = True
    response = get_pipeline_response(prompt, history)
    st.session_state.is_generating = False
    st.session_state.pending_rag_request = None

    chunks = response.get("chunks") or []
    if st.session_state.settings.get("top_k"):
        chunks = chunks[: st.session_state.settings["top_k"]]
    out_of_scope = bool(response.get("out_of_scope", False))
    answer = response.get("answer") or (
        "Xin lỗi, hệ thống chưa tạo được câu trả lời. "
        "Bạn vui lòng thử lại với câu hỏi cụ thể hơn."
    )

    stream_response = {
        "chat_id": target_chat_id,
        "answer": answer,
        "chunks": [] if out_of_scope else chunks,
        "out_of_scope": out_of_scope,
    }

    if target_chat_id == st.session_state.chat_id:
        st.session_state.last_response = {
            "answer": answer,
            "chunks": chunks,
            "out_of_scope": out_of_scope,
            "query": prompt,
        }
        st.session_state.last_chunks = [] if out_of_scope else chunks
        st.session_state.pending_stream_response = stream_response
    else:
        chats = load_chats()
        target_chat = chats.get(target_chat_id)
        if target_chat:
            messages = list(target_chat.get("messages", []))
            messages.append(
                create_message(
                    "assistant",
                    answer,
                    chunks=[] if out_of_scope else chunks,
                    out_of_scope=out_of_scope,
                )
            )
            save_chat(target_chat_id, target_chat.get("title") or target_chat_title, messages)
            st.toast("Câu trả lời đã được lưu vào hội thoại gốc.")
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
    load_env_file()
    init_session_state()
    st.session_state.chats = load_chats()

    prompt = render_main_layout()
    handle_pending_rag_request()
    if prompt:
        handle_submit(prompt)


if __name__ == "__main__":
    main()
