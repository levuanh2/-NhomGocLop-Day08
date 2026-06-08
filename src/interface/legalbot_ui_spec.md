# UI Specification — LegalBot Interface

## 1. Mục tiêu giao diện

Giao diện LegalBot là lớp tương tác cuối cùng giữa người dùng và hệ thống RAG pháp luật. UI có nhiệm vụ:

1. Nhận câu hỏi tiếng Việt từ người dùng.
2. Gửi câu hỏi và lịch sử hội thoại vào RAG pipeline.
3. Hiển thị câu trả lời dạng Markdown.
4. Hiển thị nguồn tham khảo dưới dạng citation cards.
5. Hỗ trợ hội thoại nhiều lượt.
6. Lưu lịch sử hội thoại tạm thời bằng file JSON.
7. Cho phép tạo mới, đổi tên, xoá và mở lại hội thoại cũ.

LegalBot chỉ hỗ trợ câu hỏi thuộc phạm vi pháp luật Việt Nam và các vụ việc vi phạm pháp luật từ báo chí. Mọi câu trả lời hợp lệ cần có citation để người dùng kiểm chứng nguồn.

---

## 2. Layout tổng thể

Giao diện sử dụng bố cục **3 cột cố định**.

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ 🏛️ LegalBot                                      Tên hội thoại hiện tại       │
│ Chatbot tư vấn pháp luật Việt Nam có trích dẫn nguồn                         │
├───────────────────────┬────────────────────────────────┬─────────────────────┤
│ SYSTEM PANEL          │ CHAT PANEL                     │ CITATION PANEL      │
│                       │                                │                     │
│ + Hội thoại mới       │ Tin nhắn người dùng            │ Nguồn tham khảo     │
│ Lịch sử hội thoại     │ Tin nhắn LegalBot              │                     │
│ Câu hỏi mẫu           │                                │ [1] Citation card   │
│ Thông số hệ thống     │ Ô nhập câu hỏi                 │ [2] Citation card   │
│                       │                                │                     │
└───────────────────────┴────────────────────────────────┴─────────────────────┘
```

Tỉ lệ cột đề xuất:

```python
left_col, chat_col, source_col = st.columns([0.23, 0.52, 0.25])
```

| Cột | Tên | Vai trò |
|---|---|---|
| Cột trái | `SystemPanel` | Quản lý hội thoại, thông số, câu hỏi mẫu |
| Cột giữa | `ChatPanel` | Hiển thị hội thoại và input |
| Cột phải | `CitationPanel` | Hiển thị nguồn tham khảo của câu trả lời gần nhất hoặc câu trả lời đang chọn |

---

## 3. Trạng thái mặc định khi mở app

Khi app được mở lần đầu, mặc định là **cuộc hội thoại mới**.

```python
st.session_state.chat_id = None
st.session_state.current_chat_title = "Hội thoại mới"
st.session_state.messages = []
st.session_state.last_chunks = []
st.session_state.last_response = None
st.session_state.selected_message_id = None
```

### Header mặc định

```text
🏛️ LegalBot
Hội thoại mới
```

### Chat panel mặc định

```text
Xin chào, tôi là LegalBot.
Bạn có thể hỏi tôi về:
- Quy định pháp luật Việt Nam
- Điều luật, nghị định, thông tư
- Vụ việc vi phạm pháp luật được đăng trên báo chí
```

### Citation panel mặc định

```text
📚 Chưa có nguồn tham khảo
Nguồn sẽ xuất hiện sau khi bạn đặt câu hỏi hợp lệ.
```

---

## 4. Cấu trúc thư mục đề xuất

```text
src/
├── interface/
│   ├── app.py
│   ├── layout.py
│   ├── components.py
│   ├── state.py
│   ├── chat_store.py
│   ├── title_utils.py
│   ├── formatters.py
│   ├── mock_data.py
│   ├── styles.css
│   └── assets/
│       ├── user.png
│       ├── bot.png
│       └── logo.png
│
├── rag_pipeline/
│   ├── pipeline.py
│   ├── generation.py
│   ├── retrieval_pipeline.py
│   └── ...
│
data/
└── chat_history/
    └── chats.json
```

### Vai trò từng file

| File | Vai trò |
|---|---|
| `app.py` | Entry point chạy Streamlit |
| `layout.py` | Dựng layout 3 cột và header |
| `components.py` | Render các thành phần UI |
| `state.py` | Khởi tạo và quản lý `st.session_state` |
| `chat_store.py` | Lưu, đọc, xoá, đổi tên hội thoại trong JSON |
| `title_utils.py` | Tự động đặt tên hội thoại |
| `formatters.py` | Format score, source type, preview text |
| `mock_data.py` | Mock `BotResponse` để phát triển UI trước khi pipeline xong |
| `styles.css` | CSS tuỳ chỉnh |
| `assets/` | Avatar, logo, icon |

---

## 5. Data Contract giữa UI và RAG Pipeline

UI chỉ gọi một entry point chính:

```python
from rag_pipeline.pipeline import chat

response = chat(query, history)
```

Hàm `chat()` nhận:

```python
query: str
history: list[dict]
```

và trả về:

```python
BotResponse: dict
```

UI không tự gọi retrieval hoặc generation riêng. UI chỉ truyền câu hỏi và lịch sử hội thoại vào pipeline.

---

## 5.1. Schema `Message`

Dùng cho lịch sử hội thoại truyền vào pipeline.

```python
Message = {
    "role": str,      # "user" | "assistant"
    "content": str    # Nội dung tin nhắn
}
```

UI có thể mở rộng thêm field nội bộ để phục vụ hiển thị:

```python
UIMessage = {
    "id": str,
    "role": str,              # "user" | "assistant"
    "content": str,
    "created_at": str,
    "chunks": list[dict],     # chỉ có với assistant message
    "out_of_scope": bool      # chỉ có với assistant message
}
```

Khi truyền vào RAG pipeline, chỉ truyền dạng tối giản:

```python
history_for_pipeline = [
    {
        "role": msg["role"],
        "content": msg["content"]
    }
    for msg in st.session_state.messages[-10:]
]
```

Không truyền `id`, `chunks`, `created_at`, `out_of_scope` vào pipeline.

---

## 5.2. Schema `BotResponse`

UI nhận từ RAG pipeline:

```python
BotResponse = {
    "answer": str,
    "chunks": list[Chunk],
    "out_of_scope": bool,
    "query": str
}
```

| Field | Kiểu | Ý nghĩa |
|---|---|---|
| `answer` | `str` | Câu trả lời Markdown từ model |
| `chunks` | `list[Chunk]` | Danh sách nguồn đã dùng để sinh câu trả lời |
| `out_of_scope` | `bool` | Câu hỏi có nằm ngoài phạm vi hay không |
| `query` | `str` | Câu hỏi gốc của người dùng |

Quy định:

- Nếu `out_of_scope=True`, UI hiển thị cảnh báo màu vàng.
- Nếu `out_of_scope=True`, không render citation cards.
- Nếu `out_of_scope=False`, render câu trả lời và citation cards từ `chunks`.
- `answer` được render bằng Markdown.

---

## 5.3. Schema `Chunk`

Mỗi citation card được render từ một `Chunk`.

```python
Chunk = {
    "content": str,
    "score": float,
    "metadata": {
        "source": str,
        "type": str,          # "legal" | "news"
        "chunk_index": int,
        "url": str,
        "title": str,
        "date": str,
        "article_id": str
    },
    "source": str             # "hybrid" | "pageindex"
}
```

---

## 6. Cấu trúc JSON lưu hội thoại tạm

File lưu trữ:

```text
data/chat_history/chats.json
```

Cấu trúc đề xuất:

```json
{
  "version": "1.0",
  "updated_at": "2026-06-08T15:45:00",
  "chats": {
    "chat_20260608_154500_ab12": {
      "id": "chat_20260608_154500_ab12",
      "title": "Điều kiện hưởng án treo",
      "messages": [
        {
          "id": "msg_001",
          "role": "user",
          "content": "Điều kiện để được hưởng án treo là gì?",
          "created_at": "2026-06-08T15:45:10"
        },
        {
          "id": "msg_002",
          "role": "assistant",
          "content": "Theo quy định pháp luật Việt Nam...",
          "chunks": [],
          "out_of_scope": false,
          "created_at": "2026-06-08T15:45:20"
        }
      ],
      "created_at": "2026-06-08T15:45:00",
      "updated_at": "2026-06-08T15:45:20"
    }
  }
}
```

Quy định:

1. File JSON chỉ dùng cho prototype/demo.
2. Khi triển khai thật, có thể thay bằng SQLite, PostgreSQL hoặc database theo user.
3. Không lưu API key hoặc thông tin nhạy cảm vào JSON.
4. Có thể lưu `chunks` trong từng assistant message để mở lại hội thoại cũ vẫn xem được nguồn.
5. Nếu muốn đơn giản hơn, có thể chỉ lưu `last_chunks` ở cấp chat.
6. Không lưu hội thoại rỗng vào JSON.

---

## 7. Các thành phần UI và tham số

---

### 7.1. `AppHeader`

Hiển thị logo, tên app và tên hội thoại hiện tại.

#### Props

```python
AppHeaderProps = {
    "app_title": str,
    "subtitle": str,
    "chat_title": str
}
```

#### Ví dụ

```python
render_app_header(
    app_title="🏛️ LegalBot",
    subtitle="Chatbot tư vấn pháp luật Việt Nam có trích dẫn nguồn",
    chat_title=st.session_state.current_chat_title
)
```

#### Quy định render

| Trường hợp | Hiển thị |
|---|---|
| `chat_id is None` | `Hội thoại mới` |
| Chat đã có title | Hiển thị title |
| Title quá dài | Cắt còn khoảng 50 ký tự |

---

### 7.2. `SystemPanel`

Cột trái, chứa thông tin hệ thống.

#### Props

```python
SystemPanelProps = {
    "chats": dict,
    "active_chat_id": str | None,
    "settings": dict
}
```

#### Thành phần con

```text
1. NewChatButton
2. ExampleQuestions
3. ChatHistoryList
4. SettingsPanel
5. DisclaimerBox
```

#### Quy định

- Nút `+ Hội thoại mới` luôn hiển thị trên cùng.
- Danh sách hội thoại sắp xếp theo `updated_at` giảm dần.
- Hội thoại đang mở được highlight.
- Có thể đổi tên hoặc xoá hội thoại đang mở.
- Khi xoá hội thoại đang mở, UI quay về trạng thái `Hội thoại mới`.

---

### 7.3. `ChatHistoryList`

Hiển thị danh sách hội thoại cũ.

#### Props

```python
ChatHistoryListProps = {
    "chats": dict,
    "active_chat_id": str | None
}
```

#### Mỗi item

```python
ChatHistoryItem = {
    "id": str,
    "title": str,
    "updated_at": str,
    "message_count": int
}
```

#### Action

| Action | Kết quả |
|---|---|
| Click hội thoại | Load messages vào `st.session_state.messages` |
| Click rename | Mở dialog đổi tên |
| Click delete | Mở dialog xác nhận xoá |
| Click new chat | Reset về hội thoại mới |

---

### 7.4. `SettingsPanel`

Cho phép chỉnh một vài thông số hiển thị hoặc gọi pipeline.

#### Props

```python
Settings = {
    "model": str,
    "top_k": int,
    "show_debug": bool,
    "show_scores": bool,
    "show_source_file": bool
}
```

#### Mặc định

```python
default_settings = {
    "model": "claude-sonnet-4-6",
    "top_k": 5,
    "show_debug": False,
    "show_scores": True,
    "show_source_file": False
}
```

#### Ghi chú

Hiện tại `chat(query, history)` trong spec chưa nhận `top_k` hoặc `model` từ UI. Vì vậy các setting này có thể dùng cho hiển thị/debug trước. Nếu TV2 mở rộng hàm `chat()`, có thể truyền thêm sau.

---

### 7.5. `ChatPanel`

Cột giữa, hiển thị hội thoại và ô nhập câu hỏi.

#### Props

```python
ChatPanelProps = {
    "messages": list[UIMessage],
    "is_generating": bool
}
```

#### Thành phần

```text
1. Empty welcome state
2. Message list
3. Loading state
4. Chat input
```

#### Quy định

- Nếu `messages=[]`, hiển thị welcome state.
- Message `role=user` hiển thị bubble user.
- Message `role=assistant` hiển thị bubble bot.
- Nội dung assistant render bằng Markdown.
- Nếu assistant message có `out_of_scope=True`, dùng alert style.
- Khi gửi câu hỏi mới, input bị disable trong lúc đang xử lý.

---

### 7.6. `ChatMessage`

Render một tin nhắn.

#### Props

```python
ChatMessageProps = {
    "id": str,
    "role": str,
    "content": str,
    "created_at": str | None,
    "chunks": list[dict],
    "out_of_scope": bool
}
```

#### Quy định

| Role | Cách hiển thị |
|---|---|
| `user` | Avatar user, bubble câu hỏi |
| `assistant` | Avatar bot, Markdown answer |
| `assistant + out_of_scope=True` | Alert màu vàng |
| `assistant + chunks != []` | Có thể click để xem nguồn của message đó |

---

### 7.7. `CitationPanel`

Cột phải, hiển thị nguồn tham khảo.

#### Props

```python
CitationPanelProps = {
    "chunks": list[Chunk],
    "out_of_scope": bool,
    "selected_message_id": str | None
}
```

#### Quy định render

| Trường hợp | Hiển thị |
|---|---|
| Chưa có câu hỏi | Empty state |
| `out_of_scope=True` | Không có nguồn tham khảo |
| `chunks=[]` | Không tìm thấy nguồn phù hợp |
| `chunks != []` | Render danh sách citation cards |

Citation cards phải hiển thị đủ các trường chính:

- Số thứ tự `[1]`, `[2]`, `[3]`
- Tiêu đề
- Loại nguồn
- Ngày
- Link nếu là báo chí
- Điều khoản nếu là văn bản luật
- Đoạn trích 200 ký tự
- Độ tin cậy dạng phần trăm hoặc progress bar

---

### 7.8. `CitationCard`

Render một nguồn.

#### Props

```python
CitationCardProps = {
    "index": int,
    "chunk": Chunk,
    "show_debug": bool
}
```

#### Mapping field

```python
index                         -> số citation [1], [2], [3]
chunk.metadata.title          -> tiêu đề
chunk.metadata.type           -> loại nguồn
chunk.metadata.date           -> ngày
chunk.metadata.url            -> link nguồn nếu type == "news"
chunk.metadata.article_id     -> điều khoản nếu type == "legal"
chunk.content                 -> trích dẫn
chunk.score                   -> độ tin cậy
chunk.source                  -> nguồn retrieval, chỉ hiện khi debug
```

#### Nếu nguồn là văn bản luật

Điều kiện:

```python
chunk["metadata"]["type"] == "legal"
```

Hiển thị:

```text
[1] Bộ luật Hình sự 2015
Loại nguồn: Văn bản luật
Điều khoản: Điều 171
Ngày: 2015-11-27
Độ tin cậy: 92%
Trích dẫn: ...
```

#### Nếu nguồn là báo chí

Điều kiện:

```python
chunk["metadata"]["type"] == "news"
```

Hiển thị:

```text
[2] Tiêu đề bài báo
Loại nguồn: Báo chí
Ngày: 2024-03-15
Link: Mở nguồn
Độ tin cậy: 87%
Trích dẫn: ...
```

---

### 7.9. `OutOfScopeAlert`

Hiển thị khi câu hỏi nằm ngoài phạm vi.

#### Props

```python
OutOfScopeAlertProps = {
    "answer": str
}
```

#### Nội dung mặc định

```text
⚠️ Câu hỏi nằm ngoài phạm vi hỗ trợ

LegalBot chỉ hỗ trợ:
- Quy định pháp luật Việt Nam
- Văn bản luật, nghị định, thông tư
- Vụ việc vi phạm pháp luật được đăng trên báo chí
```

---

## 8. Luồng UI chính

---

### 8.1. Luồng khởi tạo app

```text
START
  ↓
Load CSS
  ↓
Init session state
  ↓
Load chats.json
  ↓
Set default state:
  - chat_id = None
  - current_chat_title = "Hội thoại mới"
  - messages = []
  - last_chunks = []
  ↓
Render 3-column layout
  ↓
END
```

Pseudo-code:

```python
def init_app():
    load_css()
    init_session_state()
    st.session_state.chats = load_chats()
    render_main_layout()
```

---

### 8.2. Luồng tạo hội thoại mới

```text
User click "+ Hội thoại mới"
  ↓
Nếu hội thoại hiện tại rỗng:
  không cần lưu
Nếu hội thoại hiện tại có messages:
  lưu vào chats.json
  ↓
Reset state:
  chat_id = None
  current_chat_title = "Hội thoại mới"
  messages = []
  last_chunks = []
  last_response = None
  selected_message_id = None
  ↓
Render empty chat
```

Pseudo-code:

```python
def handle_new_chat():
    persist_current_chat_if_needed()

    st.session_state.chat_id = None
    st.session_state.current_chat_title = "Hội thoại mới"
    st.session_state.messages = []
    st.session_state.last_chunks = []
    st.session_state.last_response = None
    st.session_state.selected_message_id = None

    st.rerun()
```

---

### 8.3. Luồng gửi câu hỏi đầu tiên

```text
User nhập prompt
  ↓
Validate prompt
  ↓
Vì chat_id = None:
  tạo chat_id mới
  title = "Cuộc trò chuyện mới"
  ↓
Append user message
  ↓
Build history_for_pipeline
  ↓
Call chat(query, history)
  ↓
Receive BotResponse
  ↓
Nếu out_of_scope=True:
  append assistant message dạng alert
  last_chunks = []
Nếu out_of_scope=False:
  append assistant message thường
  last_chunks = response.chunks
  ↓
Tự đặt tên hội thoại từ câu hỏi đầu tiên
  ↓
Save chat vào chats.json
  ↓
Render lại UI
```

Pseudo-code:

```python
def handle_submit(prompt: str):
    prompt = prompt.strip()
    if not prompt:
        return

    ensure_active_chat()

    user_msg = create_message(
        role="user",
        content=prompt
    )
    st.session_state.messages.append(user_msg)

    history = build_history_for_pipeline(st.session_state.messages)

    st.session_state.is_generating = True
    response = chat(prompt, history)
    st.session_state.is_generating = False

    assistant_msg = create_message(
        role="assistant",
        content=response["answer"],
        chunks=response.get("chunks", []),
        out_of_scope=response.get("out_of_scope", False)
    )

    st.session_state.messages.append(assistant_msg)

    if response.get("out_of_scope"):
        st.session_state.last_chunks = []
    else:
        st.session_state.last_chunks = response.get("chunks", [])

    auto_title_if_needed(prompt)
    persist_current_chat()
    st.rerun()
```

---

### 8.4. Luồng gửi câu hỏi nhiều lượt

```text
User nhập câu hỏi mới
  ↓
Append user message
  ↓
Lấy 10 messages gần nhất làm history
  ↓
Call chat(query, history)
  ↓
Append assistant answer
  ↓
Update citation panel theo response mới nhất
  ↓
Save chat
```

Quy định:

```python
history_for_pipeline = messages[-10:]
```

Chỉ truyền các field:

```python
{
    "role": msg["role"],
    "content": msg["content"]
}
```

Không truyền `chunks`, `created_at`, `id`, `out_of_scope` vào pipeline.

---

### 8.5. Luồng hiển thị citation

```text
Sau khi có BotResponse
  ↓
Nếu out_of_scope=True:
  CitationPanel hiển thị "Không có nguồn tham khảo"
Nếu chunks=[]:
  CitationPanel hiển thị "Không tìm thấy nguồn phù hợp"
Nếu chunks có dữ liệu:
  Render từng CitationCard theo thứ tự chunks
```

Pseudo-code:

```python
def render_citation_panel():
    chunks = st.session_state.last_chunks

    if st.session_state.last_response and st.session_state.last_response.get("out_of_scope"):
        render_no_sources_for_out_of_scope()
        return

    if not chunks:
        render_empty_sources()
        return

    for index, chunk in enumerate(chunks, start=1):
        render_citation_card(index, chunk)
```

---

### 8.6. Luồng mở hội thoại cũ

```text
User click vào một hội thoại trong SystemPanel
  ↓
Load chat từ chats.json
  ↓
Set:
  chat_id = selected id
  current_chat_title = chat.title
  messages = chat.messages
  last_chunks = chunks của assistant message cuối cùng nếu có
  ↓
Render chat panel
  ↓
Render citation panel
```

Pseudo-code:

```python
def handle_select_chat(chat_id: str):
    chat_data = get_chat(chat_id)

    st.session_state.chat_id = chat_id
    st.session_state.current_chat_title = chat_data["title"]
    st.session_state.messages = chat_data["messages"]

    last_assistant = find_last_assistant_message(chat_data["messages"])
    st.session_state.last_chunks = last_assistant.get("chunks", []) if last_assistant else []

    st.rerun()
```

---

### 8.7. Luồng đổi tên hội thoại

```text
User click Rename
  ↓
Mở dialog
  ↓
Nhập title mới
  ↓
Validate title không rỗng
  ↓
Update chats.json
  ↓
Update current_chat_title
  ↓
Toast "Đổi tên thành công"
```

Quy định:

- Không cho title rỗng.
- Title tối đa 60 ký tự.
- Nếu quá dài thì cắt hoặc báo lỗi.

---

### 8.8. Luồng xoá hội thoại

```text
User click Delete
  ↓
Mở confirm dialog
  ↓
Nếu Cancel:
  đóng dialog
Nếu Confirm:
  xoá khỏi chats.json
  nếu đang mở chat đó:
    reset về "Hội thoại mới"
  ↓
Toast "Đã xoá hội thoại"
```

---

## 9. Quy định đặt tên hội thoại

Mặc định:

```text
Hội thoại mới
```

Khi user gửi câu hỏi đầu tiên:

```text
Hội thoại mới
→ title được sinh tự động từ câu hỏi đầu tiên
```

Ví dụ:

| Câu hỏi đầu tiên | Tên hội thoại |
|---|---|
| `Điều kiện để được hưởng án treo là gì?` | `Điều kiện hưởng án treo` |
| `Tội cướp giật tài sản bị phạt tù bao nhiêu năm?` | `Tội cướp giật tài sản` |
| `Vụ Nguyễn Phương Hằng vi phạm pháp luật gì?` | `Vụ Nguyễn Phương Hằng` |

Hàm đề xuất:

```python
def generate_chat_title(messages: list[dict]) -> str:
    first_user_message = find_first_user_message(messages)
    if not first_user_message:
        return "Hội thoại mới"

    return fallback_title_from_query(first_user_message["content"])
```

Rule-based fallback:

```python
def fallback_title_from_query(query: str) -> str:
    title = query.strip()

    remove_phrases = [
        "cho tôi hỏi",
        "tôi muốn hỏi",
        "hãy giải thích",
        "giải thích",
        "là gì",
        "như thế nào",
        "bao nhiêu năm"
    ]

    lowered = title.lower()
    for phrase in remove_phrases:
        lowered = lowered.replace(phrase, "")

    title = lowered.strip(" ?.!").capitalize()

    if len(title) > 50:
        title = title[:47].strip() + "..."

    return title or "Hội thoại pháp luật"
```

---

## 10. Quy định hiển thị câu trả lời

### Với câu hỏi hợp lệ

Assistant message hiển thị:

```text
⚖️ LegalBot

Theo quy định pháp luật Việt Nam, ...
[1][2]
```

Quy định:

1. Render Markdown.
2. Giữ nguyên citation marker `[1]`, `[2]`.
3. Không tự sửa nội dung answer.
4. Không render citation nếu `chunks=[]`.
5. Nếu answer rỗng hoặc lỗi, hiển thị fallback error.

Fallback:

```text
Xin lỗi, hệ thống chưa tạo được câu trả lời. Bạn vui lòng thử lại với câu hỏi cụ thể hơn.
```

---

### Với câu hỏi ngoài phạm vi

Hiển thị alert:

```text
⚠️ Câu hỏi nằm ngoài phạm vi hỗ trợ

LegalBot chỉ hỗ trợ các câu hỏi về pháp luật Việt Nam hoặc vụ việc vi phạm pháp luật từ báo chí.
```

Quy định:

1. Không render citation cards.
2. Không gọi thêm nguồn ngoài.
3. Vẫn lưu message vào lịch sử hội thoại.
4. Citation panel hiển thị trạng thái không có nguồn.

---

## 11. Quy định validation input

Trước khi gọi `chat()`:

```python
def validate_prompt(prompt: str) -> tuple[bool, str]:
    if not prompt or not prompt.strip():
        return False, "Vui lòng nhập câu hỏi."

    if len(prompt.strip()) < 3:
        return False, "Câu hỏi quá ngắn."

    if len(prompt) > 2000:
        return False, "Câu hỏi quá dài. Vui lòng rút gọn."

    return True, ""
```

| Trường hợp | Cách xử lý |
|---|---|
| Input rỗng | Không gọi pipeline |
| Input quá ngắn | Hiển thị warning |
| Input quá dài | Yêu cầu rút gọn |
| Đang generating | Disable input |

---

## 12. Quy định xử lý lỗi

### Lỗi pipeline

Nếu `chat()` raise exception:

```python
try:
    response = chat(prompt, history)
except Exception as e:
    response = {
        "answer": "Xin lỗi, hệ thống đang gặp lỗi khi xử lý câu hỏi. Vui lòng thử lại sau.",
        "chunks": [],
        "out_of_scope": False,
        "query": prompt
    }
```

Nếu `show_debug=True`, hiển thị thêm lỗi kỹ thuật ở cột trái hoặc expander debug.

Không hiển thị stack trace cho user mặc định.

---

### Lỗi JSON storage

Nếu `chats.json` chưa tồn tại:

```text
Tự tạo file mới.
```

Nếu file JSON lỗi format:

```text
Backup file lỗi thành chats_corrupted_<timestamp>.json
Tạo file chats.json mới.
```

---

## 13. Các hàm phụ cần có

### `state.py`

```python
def init_session_state() -> None:
    ...

def reset_current_chat() -> None:
    ...

def ensure_active_chat() -> str:
    ...

def build_history_for_pipeline(messages: list[dict], limit: int = 10) -> list[dict]:
    ...
```

---

### `chat_store.py`

```python
def load_chats() -> dict:
    ...

def save_chats(chats: dict) -> None:
    ...

def create_new_chat() -> str:
    ...

def get_chat(chat_id: str) -> dict | None:
    ...

def save_chat(
    chat_id: str,
    title: str,
    messages: list[dict]
) -> None:
    ...

def rename_chat(chat_id: str, new_title: str) -> None:
    ...

def delete_chat(chat_id: str) -> None:
    ...

def cleanup_empty_chats() -> None:
    ...
```

---

### `title_utils.py`

```python
def generate_chat_title(messages: list[dict]) -> str:
    ...

def fallback_title_from_query(query: str) -> str:
    ...

def normalize_title(title: str, max_length: int = 50) -> str:
    ...
```

---

### `formatters.py`

```python
def format_source_type(source_type: str) -> str:
    ...

def format_score(score: float) -> str:
    ...

def preview_text(text: str, limit: int = 200) -> str:
    ...

def get_chunk_title(chunk: dict) -> str:
    ...

def get_chunk_date(chunk: dict) -> str:
    ...

def get_article_id(chunk: dict) -> str:
    ...
```

---

### `components.py`

```python
def render_app_header() -> None:
    ...

def render_system_panel() -> None:
    ...

def render_chat_history_list() -> None:
    ...

def render_settings_panel() -> None:
    ...

def render_chat_panel() -> None:
    ...

def render_chat_message(message: dict) -> None:
    ...

def render_citation_panel() -> None:
    ...

def render_citation_card(index: int, chunk: dict) -> None:
    ...

def render_out_of_scope_alert(answer: str) -> None:
    ...

def render_empty_chat_state() -> None:
    ...

def render_empty_sources_state() -> None:
    ...
```

---

## 14. Pseudo-code tổng thể cho AI agent

### `app.py`

```python
import streamlit as st

from interface.state import init_session_state
from interface.layout import render_main_layout
from interface.styles import load_css

st.set_page_config(
    page_title="LegalBot",
    page_icon="🏛️",
    layout="wide"
)

init_session_state()
load_css()
render_main_layout()
```

---

### `layout.py`

```python
import streamlit as st

from interface.components import (
    render_app_header,
    render_system_panel,
    render_chat_panel,
    render_citation_panel
)

def render_main_layout():
    render_app_header()

    left_col, chat_col, source_col = st.columns([0.23, 0.52, 0.25])

    with left_col:
        render_system_panel()

    with chat_col:
        render_chat_panel()

    with source_col:
        render_citation_panel()
```

---

### `components.py` — chat submit flow

```python
def render_chat_panel():
    if not st.session_state.messages:
        render_empty_chat_state()

    for message in st.session_state.messages:
        render_chat_message(message)

    prompt = st.chat_input("Nhập câu hỏi pháp luật của bạn...")

    if prompt:
        handle_submit(prompt)
```

---

### Controller/helper

```python
def handle_submit(prompt: str):
    valid, error = validate_prompt(prompt)
    if not valid:
        st.warning(error)
        return

    ensure_active_chat()

    user_msg = create_message("user", prompt)
    st.session_state.messages.append(user_msg)

    history = build_history_for_pipeline(st.session_state.messages)

    with st.spinner("Đang tìm kiếm nguồn luật và tạo câu trả lời..."):
        response = chat(prompt, history)

    assistant_msg = create_message(
        role="assistant",
        content=response["answer"],
        chunks=response.get("chunks", []),
        out_of_scope=response.get("out_of_scope", False)
    )

    st.session_state.messages.append(assistant_msg)
    st.session_state.last_response = response
    st.session_state.last_chunks = [] if response["out_of_scope"] else response.get("chunks", [])

    auto_title_if_needed(prompt)
    persist_current_chat()

    st.rerun()
```

---

## 15. Acceptance Criteria

UI được xem là hoàn thành khi:

```text
[ ] Mở app mặc định là "Hội thoại mới"
[ ] Giao diện có 3 cột cố định
[ ] Cột trái có lịch sử hội thoại, nút tạo mới, settings, câu hỏi mẫu
[ ] Cột giữa có chat window và chat input
[ ] Cột phải có citation panel
[ ] Gửi câu hỏi gọi được chat(query, history)
[ ] Câu trả lời Markdown hiển thị đúng
[ ] Citation cards hiển thị đủ field
[ ] out_of_scope hiển thị cảnh báo vàng và không render nguồn
[ ] Hội thoại nhiều lượt được giữ trong session
[ ] Hội thoại được lưu vào chats.json
[ ] Có thể mở lại hội thoại cũ
[ ] Có thể đổi tên hội thoại
[ ] Có thể xoá hội thoại
[ ] Tự động đặt tên hội thoại sau câu hỏi đầu tiên
[ ] Có xử lý lỗi pipeline và lỗi JSON
```

---

## 16. Ghi chú quan trọng cho AI agent triển khai

1. Không thay đổi schema `BotResponse`, `Chunk`, `Message`.
2. UI chỉ gọi `chat(query, history)`.
3. Không tự gọi retrieval hoặc generation riêng trong UI.
4. Mặc định app bắt đầu với `chat_id=None`, title là `"Hội thoại mới"`.
5. Chỉ tạo `chat_id` thật khi user gửi câu hỏi đầu tiên.
6. Không lưu hội thoại rỗng vào JSON.
7. Citation panel lấy dữ liệu từ `BotResponse.chunks`.
8. Nếu `out_of_scope=True`, không render citation cards.
9. Khi lưu history vào pipeline, chỉ truyền `role` và `content`.
10. JSON storage chỉ là tạm thời cho prototype/demo.

---

## 17. Gợi ý thứ tự triển khai

```text
1. Tạo cấu trúc thư mục interface
2. Tạo mock_data.py theo schema BotResponse
3. Dựng layout 3 cột cố định
4. Render empty state ban đầu
5. Render chat message
6. Render citation card
7. Viết chat_store.py để lưu JSON
8. Viết title_utils.py để đặt tên hội thoại
9. Thêm tạo mới / mở lại / đổi tên / xoá hội thoại
10. Tích hợp chat(query, history) thật từ rag_pipeline
11. Xử lý out_of_scope
12. Thêm CSS polish cho demo
```

---

## 18. Phạm vi phiên bản prototype

Phiên bản prototype cần ưu tiên:

```text
3 cột cố định
+ hội thoại mới mặc định
+ chat nhiều lượt
+ citation panel
+ lưu JSON tạm
+ tự đặt tên hội thoại
+ xử lý out_of_scope
```

Các tính năng nâng cao như authentication, user-specific database, real-time streaming, export PDF hoặc phân quyền người dùng có thể để sau.
