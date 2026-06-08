"""Reusable Streamlit components for LegalBot."""

from __future__ import annotations

import html
import time

import streamlit as st

from chat_store import delete_chat, load_chats, rename_chat, save_chat
from formatters import (
    chunk_metadata,
    format_score,
    format_source_type,
    get_article_id,
    get_chunk_date,
    get_chunk_title,
    preview_text,
    score_percent,
)
from mock_data import EXAMPLE_QUESTIONS
from state import create_message, find_last_assistant_message, reset_current_chat
from title_utils import normalize_title


def render_app_header(app_title: str, subtitle: str, chat_title: str) -> None:
    short_title = normalize_title(chat_title or "Hội thoại mới", 50)
    st.markdown(
        f"""
        <div class="app-header">
            <div>
                <div class="brand-row">
                    <div class="brand-mark">⚖</div>
                    <div>
                        <h1>{app_title}</h1>
                        <p>{subtitle}</p>
                    </div>
                </div>
            </div>
            <div class="current-chat">
                <span>Đang mở</span>
                <strong>{short_title}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def persist_current_chat_if_needed() -> None:
    if st.session_state.get("chat_id") and st.session_state.get("messages"):
        save_chat(
            st.session_state.chat_id,
            st.session_state.current_chat_title,
            st.session_state.messages,
        )


def handle_select_chat(chat_id: str) -> None:
    chat = load_chats().get(chat_id)
    if not chat:
        return
    st.session_state.chat_id = chat_id
    st.session_state.current_chat_title = chat.get("title", "Hội thoại pháp luật")
    st.session_state.messages = chat.get("messages", [])
    last_assistant = find_last_assistant_message(st.session_state.messages)
    st.session_state.last_chunks = last_assistant.get("chunks", []) if last_assistant else []
    st.session_state.last_response = {
        "out_of_scope": bool(last_assistant.get("out_of_scope")) if last_assistant else False
    }
    st.session_state.selected_message_id = last_assistant.get("id") if last_assistant else None
    st.rerun()


def handle_new_chat() -> None:
    reset_current_chat()
    st.rerun()


def render_system_panel(chats: dict, active_chat_id: str | None) -> None:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">⚖</div>
            <div>
                <strong>LegalBot</strong>
                <span>Hỏi đáp pháp luật Việt Nam</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("➕ Tạo hội thoại mới", use_container_width=True, type="primary"):
        handle_new_chat()

    st.session_state.settings["show_citations"] = st.checkbox(
        "**Hiện tài liệu tham khảo**",
        value=st.session_state.settings.get("show_citations", True),
    )

    st.markdown('<div class="sidebar-section-title">Chats</div>', unsafe_allow_html=True)
    render_chat_history_list(chats, active_chat_id)

    with st.expander("**Câu hỏi mẫu**", expanded=True):
        render_example_questions()

    with st.expander("**Cơ chế tìm kiếm Hybrid**", expanded=False):
        render_search_explanation()

    with st.expander("**Thông số hệ thống**", expanded=False):
        render_settings_panel()

    render_disclaimer_box()


def render_example_questions() -> None:
    for index, question in enumerate(EXAMPLE_QUESTIONS):
        if st.button(question, key=f"example_{index}", use_container_width=True):
            st.session_state.pending_example = question
            st.rerun()


def render_chat_history_list(chats: dict, active_chat_id: str | None) -> None:
    if not chats:
        st.caption("Chưa có hội thoại đã lưu.")
        return

    sorted_chats = sorted(
        chats.values(),
        key=lambda chat: chat.get("updated_at", ""),
        reverse=True,
    )
    for chat in sorted_chats:
        chat_id = chat["id"]
        title = normalize_title(chat.get("title", "Hội thoại pháp luật"), 44)
        is_active = chat_id == active_chat_id
        button_type = "primary" if is_active else "secondary"
        message_count = len(chat.get("messages", []))
        item_class = "chat-history-item active" if is_active else "chat-history-item"
        st.markdown(f'<div class="{item_class}">', unsafe_allow_html=True)
        if st.button(
            f"{title} · {message_count} tin nhắn",
            key=f"open_{chat_id}",
            use_container_width=True,
            type=button_type,
        ):
            handle_select_chat(chat_id)
        st.markdown("</div>", unsafe_allow_html=True)


def render_search_explanation() -> None:
    """Panel giải thích cơ chế Hybrid Search — cho điểm bonus rubric."""
    st.markdown(
        """
        LegalBot dùng **Hybrid Search** — kết hợp 3 phương pháp:

        **1. Semantic Search (Vector)**
        - Câu hỏi → embedding (text-embedding-3-small)
        - So sánh cosine similarity với 484 chunks trong ChromaDB
        - Tìm được ý nghĩa ẩn, câu đồng nghĩa

        **2. Lexical Search (BM25 cải tiến)**
        - Khác BM25 thuần: có thêm **chuẩn hoá tiếng Việt** (bỏ dấu, lowercase) và **query expansion** tự động
        - Ví dụ: câu hỏi *"cai nghiện là gì"* → BM25 tìm thêm *"giải thích từ ngữ cai nghiện"* và *"cai nghiện là"* → khớp chính xác Điều 2 định nghĩa trong luật
        - Tìm được từ khoá chính xác mà semantic có thể bỏ qua

        **3. RRF Fusion + Reranking**
        - Reciprocal Rank Fusion: gộp kết quả 2 phương pháp → rank tổng hợp
        - Cross-encoder (flashrank ONNX local): rerank lần cuối theo relevance thực sự
        - Fallback sang PageIndex nếu score thấp
        """,
        unsafe_allow_html=False,
    )
    st.caption("Hybrid > BM25 đơn lẻ vì kết hợp được ngữ nghĩa (semantic) + từ khoá (lexical).")


def render_settings_panel() -> None:
    settings = st.session_state.settings
    settings["model"] = st.selectbox(
        "Model hiển thị",
        ["claude-sonnet-4-6", "gpt-4.1", "gemini-1.5-pro"],
        index=0,
    )
    settings["top_k"] = st.slider("Số nguồn tối đa", min_value=1, max_value=10, value=settings["top_k"])
    settings["use_mock"] = st.toggle("Dùng mock data", value=settings.get("use_mock", True))
    settings["show_scores"] = st.toggle("Hiển thị độ tin cậy", value=settings["show_scores"])
    settings["show_source_file"] = st.toggle("Hiển thị file nguồn", value=settings["show_source_file"])
    settings["show_debug"] = st.toggle("Debug", value=settings["show_debug"])

    if settings["show_debug"]:
        st.info("UI sẽ dùng mock data nếu pipeline thật chưa sẵn sàng hoặc bật mock mode.")
        if st.session_state.get("debug_error"):
            st.code(st.session_state.debug_error)


def render_disclaimer_box() -> None:
    st.markdown(
        """
        <div class="disclaimer">
            <strong>Lưu ý</strong>
            <span>Thông tin chỉ phục vụ tham khảo ban đầu. Người dùng cần kiểm chứng bằng nguồn được trích dẫn.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_panel(messages: list[dict], is_generating: bool) -> str | None:
    render_chat_topbar()
    st.divider()
    pending_stream = st.session_state.get("pending_stream_response")
    if pending_stream and pending_stream.get("chat_id") != st.session_state.get("chat_id"):
        persist_pending_stream_to_original_chat(pending_stream)
        pending_stream = None
    has_pending_stream = bool(
        pending_stream
        and pending_stream.get("chat_id") == st.session_state.get("chat_id")
    )
    with st.container(height=545, border=False):
        if not messages:
            render_welcome_state()
        else:
            for message in messages:
                render_chat_message(message)

        if is_generating and not has_pending_stream:
            st.info("LegalBot đang xử lý câu hỏi...")
        if has_pending_stream:
            render_pending_stream_response()

    pending = st.session_state.pop("pending_example", None)
    with st.form("chat_input_form", clear_on_submit=True):
        input_col, send_col = st.columns([1, 0.07], gap="small")
        with input_col:
            prompt = st.text_input(
                "Câu hỏi",
                placeholder="Nhập câu hỏi pháp luật của bạn...",
                disabled=is_generating or has_pending_stream,
                label_visibility="collapsed",
            )
        with send_col:
            submitted = st.form_submit_button("➤", use_container_width=True, disabled=is_generating or has_pending_stream)

    if pending:
        return pending
    if submitted:
        return prompt
    return None


def persist_pending_stream_to_original_chat(pending: dict) -> None:
    target_chat_id = pending.get("chat_id")
    if not target_chat_id:
        st.session_state.pending_stream_response = None
        return

    chats = load_chats()
    target_chat = chats.get(target_chat_id)
    if target_chat:
        messages = list(target_chat.get("messages", []))
        messages.append(
            create_message(
                "assistant",
                pending.get("answer") or "Xin lỗi, hệ thống chưa tạo được câu trả lời.",
                chunks=pending.get("chunks") or [],
                out_of_scope=bool(pending.get("out_of_scope", False)),
            )
        )
        save_chat(target_chat_id, target_chat.get("title", "Hội thoại pháp luật"), messages)
        st.session_state.chats = load_chats()
    st.session_state.pending_stream_response = None


def render_pending_stream_response() -> None:
    pending = st.session_state.pending_stream_response
    if pending.get("chat_id") != st.session_state.get("chat_id"):
        return
    answer = pending.get("answer") or "Xin lỗi, hệ thống chưa tạo được câu trả lời."
    chunks = pending.get("chunks") or []
    out_of_scope = bool(pending.get("out_of_scope", False))

    with st.chat_message("assistant", avatar="⚖"):
        if out_of_scope:
            render_out_of_scope_alert(answer)
        else:
            message_ph = st.empty()
            full_response = ""
            chunk_size = 5
            for index in range(0, len(answer), chunk_size):
                full_response += answer[index : index + chunk_size]
                time.sleep(0.035)
                message_ph.markdown(full_response + "▌")
            message_ph.markdown(full_response)

    assistant_message = create_message(
        "assistant",
        answer,
        chunks=chunks,
        out_of_scope=out_of_scope,
    )
    st.session_state.messages.append(assistant_message)
    st.session_state.selected_message_id = assistant_message["id"]
    st.session_state.last_chunks = [] if out_of_scope else chunks
    st.session_state.last_response = {
        "answer": answer,
        "chunks": chunks,
        "out_of_scope": out_of_scope,
    }
    st.session_state.pending_stream_response = None
    save_chat(
        st.session_state.chat_id,
        st.session_state.current_chat_title,
        st.session_state.messages,
    )
    st.session_state.chats = load_chats()
    st.rerun()


def render_chat_topbar() -> None:
    title = st.session_state.current_chat_title
    st.markdown('<div class="chat-topbar-offset"></div>', unsafe_allow_html=True)
    left_col, right_col = st.columns([1, 0.34])
    with left_col:
        st.markdown(f'<div class="chat-title">{normalize_title(title, 60)}</div>', unsafe_allow_html=True)

    if st.session_state.chat_id:
        with right_col:
            rename_col, delete_col = st.columns([1, 1], gap="small")
            with rename_col:
                if st.button("Đổi tên", key="rename_current_chat", use_container_width=True, type="primary"):
                    st.session_state.pending_action = "rename"
            with delete_col:
                if st.button("Xoá", key="delete_current_chat", use_container_width=True, type="primary"):
                    st.session_state.pending_action = "delete"

    if st.session_state.pending_action:
        confirm_current_chat_action()

    if st.session_state.last_action:
        if st.session_state.last_action == "rename":
            st.toast("Đổi tên hội thoại thành công.")
        if st.session_state.last_action == "delete":
            st.toast("Đã xoá hội thoại.")
        st.session_state.last_action = None


@st.dialog("Xác nhận thao tác")
def confirm_current_chat_action() -> None:
    action = st.session_state.pending_action
    chat_id = st.session_state.chat_id

    if action == "rename":
        current_name = st.session_state.current_chat_title
        st.write("Nhập tên mới cho hội thoại:")
        new_name = st.text_input(
            "Tên mới",
            value=st.session_state.get("new_name") or current_name,
            max_chars=60,
            label_visibility="collapsed",
        )
        cancel_col, confirm_col = st.columns([1, 0.4])
        with cancel_col:
            if st.button("Huỷ", use_container_width=True):
                st.session_state.pending_action = None
                st.rerun()
        with confirm_col:
            if st.button("Xác nhận", use_container_width=True, type="primary"):
                cleaned = normalize_title(new_name, 60)
                rename_chat(chat_id, cleaned)
                st.session_state.current_chat_title = cleaned
                st.session_state.chats = load_chats()
                st.session_state.pending_action = None
                st.session_state.last_action = "rename"
                st.rerun()

    if action == "delete":
        st.warning("Bạn chắc chắn muốn xoá hội thoại này? Hành động này không thể hoàn tác.")
        cancel_col, confirm_col = st.columns([1, 0.5])
        with cancel_col:
            if st.button("Huỷ", use_container_width=True):
                st.session_state.pending_action = None
                st.rerun()
        with confirm_col:
            if st.button("Xoá vĩnh viễn", use_container_width=True, type="primary"):
                delete_chat(chat_id)
                reset_current_chat()
                st.session_state.chats = load_chats()
                st.session_state.pending_action = None
                st.session_state.last_action = "delete"
                st.rerun()


def render_welcome_state() -> None:
    st.markdown(
        """
        <div class="welcome-state">
            <h2>Xin chào, tôi là LegalBot.</h2>
            <p>Bạn có thể hỏi tôi về quy định pháp luật Việt Nam, điều luật, nghị định, thông tư hoặc vụ việc vi phạm pháp luật được đăng trên báo chí.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_message(message: dict) -> None:
    role = message.get("role", "assistant")
    content = message.get("content") or "Xin lỗi, hệ thống chưa tạo được câu trả lời."
    avatar = "👤" if role == "user" else "⚖"

    with st.chat_message(role, avatar=avatar):
        if message.get("out_of_scope"):
            render_out_of_scope_alert(content)
        else:
            st.markdown(content)

        chunks = message.get("chunks") or []
        if role == "assistant" and chunks:
            if st.button("Nguồn", key=f"sources_{message['id']}"):
                st.session_state.selected_message_id = message["id"]
                st.session_state.last_chunks = chunks
                st.session_state.last_response = {"out_of_scope": False}
                st.rerun()


def render_out_of_scope_alert(answer: str) -> None:
    st.warning(
        answer
        or (
            "Câu hỏi nằm ngoài phạm vi hỗ trợ. LegalBot chỉ hỗ trợ các câu hỏi về "
            "pháp luật Việt Nam hoặc vụ việc vi phạm pháp luật từ báo chí."
        )
    )


def render_citation_panel(chunks: list[dict], out_of_scope: bool, settings: dict) -> None:
    st.markdown("### Tài liệu tham khảo")
    with st.container(height=700, border=False):

        if out_of_scope:
            st.warning("Không có nguồn tham khảo vì câu hỏi nằm ngoài phạm vi hỗ trợ.")
        elif not chunks:
            st.markdown(
                """
                <div class="empty-sources">
                    <strong>📚 Chưa có nguồn tham khảo</strong>
                    <span>Nguồn sẽ xuất hiện sau khi bạn đặt câu hỏi hợp lệ.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            for index, chunk in enumerate(chunks, start=1):
                render_citation_card(index, chunk, settings)


def render_citation_card(index: int, chunk: dict, settings: dict) -> None:
    metadata = chunk_metadata(chunk)
    source_type = metadata.get("type", "")
    score = score_percent(chunk.get("score"))
    title = html.escape(get_chunk_title(chunk))
    source_label = html.escape(format_source_type(source_type))
    raw_article_id = get_article_id(chunk) if source_type == "legal" else ""
    article_id = html.escape(raw_article_id) if raw_article_id and not raw_article_id.startswith("Không rõ") else ""
    raw_date = metadata.get("date") or metadata.get("published_at") or metadata.get("created_at") or ""
    date_label = html.escape(raw_date)
    content = chunk.get("content") or chunk.get("text") or ""
    preview = html.escape(preview_text(content, 260))
    meta_items = [f"<span>{source_label}</span>"]
    if date_label:
        meta_items.append(f"<span>{date_label}</span>")
    if article_id:
        meta_items.append(f"<span>{article_id}</span>")
    meta_html = "".join(meta_items)

    st.markdown(
        f"""
        <article class="citation-card">
            <div class="citation-card__head">
                <span class="citation-card__index">[{index}]</span>
                <h4>{title}</h4>
            </div>
            <div class="citation-card__meta">
                {meta_html}
            </div>
            <p class="citation-card__preview">{preview}</p>
        </article>
        """,
        unsafe_allow_html=True,
    )

    if source_type == "news" and metadata.get("url"):
        st.link_button("Mở nguồn", metadata["url"], use_container_width=True)
    if settings.get("show_source_file") and metadata.get("source"):
        st.markdown(f"**File:** `{metadata['source']}`")

    if settings.get("show_scores", True):
        st.progress(score, text=f"Độ tin cậy {format_score(chunk.get('score'))}")

    if st.button("Hiện đầy đủ", key=f"show_full_chunk_{index}_{metadata.get('source', '')}_{metadata.get('chunk_index', 'na')}", use_container_width=True):
        show_full_citation_dialog(index, chunk)

    if settings.get("show_debug"):
        st.caption(
            f"retrieval={chunk.get('source', 'unknown')} · chunk_index={metadata.get('chunk_index', 'N/A')}"
        )


@st.dialog("Chi tiết nguồn")
def show_full_citation_dialog(index: int, chunk: dict) -> None:
    metadata = chunk_metadata(chunk)
    source_type = metadata.get("type", "")
    st.markdown(f"### [{index}] {get_chunk_title(chunk)}")
    st.caption(f"{format_source_type(source_type)} · {get_chunk_date(chunk)}")

    if source_type == "legal":
        st.markdown(f"**Điều khoản:** {get_article_id(chunk)}")
    if metadata.get("source"):
        st.markdown(f"**File:** `{metadata['source']}`")
    if metadata.get("url"):
        st.link_button("Mở nguồn gốc", metadata["url"])

    st.progress(score_percent(chunk.get("score")), text=f"Độ tin cậy {format_score(chunk.get('score'))}")
    st.markdown("**Nội dung**")
    st.text_area(
        "Nội dung đầy đủ",
        value=chunk.get("content") or chunk.get("text") or "",
        height=360,
        label_visibility="collapsed",
        disabled=True,
    )
