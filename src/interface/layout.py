"""Main layout composition for the LegalBot interface."""

from __future__ import annotations

import streamlit as st

from components import render_citation_panel, render_chat_panel, render_system_panel


def render_main_layout() -> str | None:
    with st.sidebar:
        render_system_panel(st.session_state.chats, st.session_state.chat_id)

    chat_col, source_col = st.columns([0.68, 0.32], gap="large")

    with chat_col:
        prompt = render_chat_panel(
            st.session_state.messages,
            st.session_state.is_generating,
        )

    with source_col:
        if st.session_state.settings.get("show_citations", True):
            out_of_scope = bool(
                st.session_state.last_response
                and st.session_state.last_response.get("out_of_scope")
            )
            render_citation_panel(
                st.session_state.last_chunks,
                out_of_scope,
                st.session_state.settings,
            )
        else:
            st.info("Nguồn tham khảo đang được ẩn.")

    return prompt
