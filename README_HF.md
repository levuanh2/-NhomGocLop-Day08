---
title: LegalBot
emoji: ⚖
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
pinned: false
license: mit
---

# LegalBot — Chatbot Tư vấn Pháp luật Việt Nam

Chatbot hỏi-đáp pháp luật Việt Nam sử dụng kiến trúc RAG (Retrieval-Augmented Generation).

## Tính năng
- Trả lời câu hỏi về văn bản pháp luật Việt Nam
- Tìm kiếm thông tin vụ việc vi phạm pháp luật từ báo chí
- Trích dẫn nguồn cho mọi câu trả lời
- Hỗ trợ hội thoại nhiều lượt (multi-turn)

## Cài đặt trên HF Spaces

Thêm các Secrets sau trong Settings → Secrets:
- `OPENAI_API_KEY` — bắt buộc
- `JINA_API_KEY` — tuỳ chọn (reranking tốt hơn)
