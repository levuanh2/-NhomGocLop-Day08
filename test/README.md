# Test CLI — LegalBot RAG Pipeline

Công cụ test pipeline từ command line, không cần mở trình duyệt.

---

## Yêu cầu

- Python environment đã cài `requirements.txt`
- `OPENAI_API_KEY` trong file `.env` ở thư mục gốc (chỉ cần cho **full RAG mode**)
- Đã chạy `build_chroma_index` để có vector store (xem README chính)

---

## Hai chế độ chạy

### Chế độ 1 — Retrieve-only (`--retrieve`)

Chỉ chạy retrieval pipeline, **không gọi LLM**. Hiển thị các chunks tìm được kèm score và nguồn.

Dùng khi:
- Kiểm tra chất lượng index sau khi thêm/sửa tài liệu
- Debug retrieval mà không tốn token
- Không có `OPENAI_API_KEY` cho generation

```bash
# Từ thư mục gốc project

# Windows CMD
set PYTHONUTF8=1 && python test/cli_rag.py --retrieve

# Windows PowerShell
$env:PYTHONUTF8=1; python test/cli_rag.py --retrieve

# Mac/Linux
PYTHONUTF8=1 python test/cli_rag.py --retrieve

# Dùng conda env ai20k (Windows)
set PYTHONUTF8=1 && conda run -n ai20k python test/cli_rag.py --retrieve
```

**Output mẫu:**

```
══════════════════════════════════════════════════════════════════════
  LEGALBOT — Retrieve-Only Mode
  Vectorstore: .../data/vectorstore/chroma
══════════════════════════════════════════════════════════════════════
  Nhập câu hỏi để xem các chunks được retrieve.
  Gõ 'q' hoặc Enter trống để thoát.

Câu hỏi > tội tàng trữ ma túy bị phạt bao nhiêu năm?

──────────────────────────────────────────────────────────────────────
  Đang retrieve cho: "tội tàng trữ ma túy bị phạt bao nhiêu năm?"
──────────────────────────────────────────────────────────────────────

  [1] 📜 Luật  bo_luat_hinh_su_2015_chuong_xx.md  |  score=0.0156  |  via=hybrid
       Điều khoản: Điều 249
       "Điều 249. Tội tàng trữ trái phép chất ma túy
       1. Người nào tàng trữ trái phép chất ma túy mà không nhằm..."
```

---

### Chế độ 2 — Full RAG (retrieve + generate)

Retrieve + generate với GPT-4o-mini. Session lưu tự động vào `sessions/<timestamp>.json`.

Yêu cầu `OPENAI_API_KEY` trong `.env`.

```bash
# Windows CMD
set PYTHONUTF8=1 && python test/cli_rag.py

# Windows PowerShell
$env:PYTHONUTF8=1; python test/cli_rag.py

# Mac/Linux
PYTHONUTF8=1 python test/cli_rag.py
```

**Output mẫu:**

```
══════════════════════════════════════════════════════════════════════
  LEGALBOT — Full RAG Mode (Retrieve + Generate)
  Model: gpt-4o-mini
  Session: 20260608_143022  →  sessions/20260608_143022.json
══════════════════════════════════════════════════════════════════════
  Chatbot chỉ trả lời về pháp luật và các vụ vi phạm pháp luật.

Câu hỏi > tội tàng trữ ma túy bị phạt bao nhiêu năm?

──────────────────────────────────────────────────────────────────────
  Đang retrieve + generate…
──────────────────────────────────────────────────────────────────────

  SOURCES (chunks đã retrieve):
  [1] 📜 Luật  bo_luat_hinh_su_2015_chuong_xx.md  |  score=0.0156  |  via=hybrid
       Điều khoản: Điều 249
       "Điều 249. Tội tàng trữ trái phép chất ma túy..."

══════════════════════════════════════════════════════════════════════
  TRẢ LỜI
══════════════════════════════════════════════════════════════════════
Theo Điều 249 Bộ luật Hình sự 2015, tội tàng trữ trái phép chất ma túy
bị xử lý như sau:

**Khoản 1** (trường hợp thông thường): phạt tù từ **1 đến 5 năm**.
...

[Bộ luật Hình sự 2015 Chương XX, Điều 249]
══════════════════════════════════════════════════════════════════════
```

---

## Câu hỏi thử mẫu

| Loại | Câu hỏi |
|---|---|
| Hình phạt | `tội tàng trữ ma túy bị phạt tù bao nhiêu năm?` |
| Hình phạt | `sản xuất trái phép ma túy bị xử lý thế nào?` |
| Định nghĩa | `cai nghiện ma túy là gì?` |
| Định nghĩa | `chất ma túy là gì theo luật Việt Nam?` |
| Sự kiện | `nghệ sĩ nào bị bắt vì ma túy?` |
| Sự kiện | `ca sĩ Miu Lê bị bắt vì tội gì?` |
| Out-of-scope | `thời tiết hôm nay thế nào?` |
| Out-of-scope | `cho tôi công thức nấu phở` |

---

## Session files

Mỗi phiên full RAG tạo ra một file JSON trong `sessions/`:

```json
{
  "session_id": "20260608_143022",
  "created_at": "20260608_143022",
  "message_count": 4,
  "messages": [
    {"role": "user", "content": "tội tàng trữ..."},
    {"role": "assistant", "content": "Theo Điều 249..."},
    ...
  ]
}
```

File này được dùng làm `history` cho multi-turn memory — chatbot nhớ ngữ cảnh từ các câu hỏi trước trong cùng session.

---

## Ghi chú

- `PYTHONUTF8=1` bắt buộc trên Windows để hiển thị đúng tiếng Việt và emoji.
- `JINA_API_KEY` không bắt buộc — reranker tự fallback sang flashrank cross-encoder local.
- `PAGEINDEX_API_KEY` không bắt buộc — pipeline bỏ qua bước fallback vectorless nếu thiếu.
- Gõ `q` hoặc Enter trống để thoát khỏi vòng lặp interactive.
