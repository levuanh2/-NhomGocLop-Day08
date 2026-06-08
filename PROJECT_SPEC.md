# Tài liệu Dự án: LegalBot — Chatbot Tư vấn Pháp luật RAG

**Phiên bản:** 1.0  
**Ngày:** 2026-06-08  
**Nhóm:** 3 thành viên  

---

## 1. Tổng quan Dự án

### Mục tiêu
Xây dựng chatbot hỏi-đáp pháp luật sử dụng kiến trúc RAG (Retrieval-Augmented Generation), chỉ trả lời các câu hỏi thuộc phạm vi:
- Các quy định pháp luật Việt Nam (văn bản luật, nghị định, thông tư)
- Thông tin các vụ vi phạm pháp luật từ báo chí

Mọi câu trả lời **bắt buộc** đính kèm citation (trích dẫn nguồn) để đảm bảo độ tin cậy.

### Kiến trúc tổng thể

```
┌─────────────────────────────────────────────────────────────────┐
│                         NGƯỜI DÙNG                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │ Câu hỏi (tiếng Việt)
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  THÀNH VIÊN 3 — Interface                                        │
│  • Giao diện chat (Streamlit / Gradio)                           │
│  • Hiển thị câu trả lời + citation cards                         │
│  • Quản lý lịch sử hội thoại                                     │
└─────────────────────────┬───────────────────────────────────────┘
                          │ query: str, history: list[Message]
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  THÀNH VIÊN 2 — RAG Pipeline                                     │
│  • retrieve(query) → List[Chunk]                                 │
│  • generate(query, chunks, history) → BotResponse                │
│  • Hybrid search: Semantic + BM25 + RRF + Rerank + Fallback      │
└──────────────┬──────────────────────────────────────────────────┘
               │ truy vấn vector store + ChromaDB
               ▼
┌─────────────────────────────────────────────────────────────────┐
│  THÀNH VIÊN 1 — Data Processing                                  │
│  • Thu thập & chuẩn hóa dữ liệu → data/standardized/            │
│  • Chunking & embedding → data/vectorstore/chroma/               │
│  • Metadata schema chuẩn cho mỗi chunk                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Phân công Công việc

| Thành viên | Module | Thư mục |
|---|---|---|
| TV1 | Data Processing | `src/data_processing/` |
| TV2 | RAG Pipeline | `src/rag_pipeline/` |
| TV3 | Interface | `src/interface/` |

---

## 3. Hợp đồng Dữ liệu (Data Contract)

> **Đây là phần quan trọng nhất** — tất cả 3 thành viên phải tuân thủ các schema dưới đây. Thay đổi schema phải được cả nhóm đồng ý trước khi sửa code.

### 3.1 Schema: Chunk (đơn vị dữ liệu nền tảng)

Mỗi đoạn văn bản được lưu trong ChromaDB và truyền giữa các module phải có cấu trúc:

```python
Chunk = {
    "content": str,          # Nội dung đoạn văn bản (UTF-8, tiếng Việt)
    "score":   float,        # Điểm liên quan [0.0, 1.0] — cao hơn = liên quan hơn
    "metadata": {
        "source":      str,  # Tên file gốc, VD: "luat_hinh_su_2015.md"
        "type":        str,  # "legal" | "news"
        "chunk_index": int,  # Thứ tự chunk trong file gốc (0-indexed)
        "url":         str,  # URL nguồn (bắt buộc với "news", "" nếu "legal")
        "title":       str,  # Tiêu đề bài báo / tên văn bản luật
        "date":        str,  # Ngày ban hành / đăng báo, định dạng "YYYY-MM-DD" hoặc ""
        "article_id":  str,  # Số điều luật VD: "Điều 134" hoặc "" nếu không có
    },
    "source": str            # "hybrid" | "pageindex" — do TV2 điền khi retrieve
}
```

**Ràng buộc:**
- `content` không được rỗng, tối thiểu 50 ký tự
- `score` luôn là float trong khoảng [0.0, 1.0]
- `metadata.type` chỉ nhận đúng 2 giá trị: `"legal"` hoặc `"news"`
- `metadata.source` là tên file (có đuôi `.md`), không phải đường dẫn đầy đủ

---

### 3.2 Schema: BotResponse (đầu ra của RAG Pipeline → Interface)

```python
BotResponse = {
    "answer":  str,          # Câu trả lời đã sinh bởi LLM (tiếng Việt, Markdown)
    "chunks":  list[Chunk],  # Danh sách chunks đã dùng để sinh câu trả lời (top 5)
    "out_of_scope": bool,    # True nếu câu hỏi nằm ngoài phạm vi pháp luật
    "query":   str,          # Câu hỏi gốc của người dùng (để hiển thị lại)
}
```

**Ràng buộc:**
- Nếu `out_of_scope = True`, `answer` phải là chuỗi từ chối tiêu chuẩn, `chunks` là `[]`
- `chunks` tối đa 5 phần tử, đã được sắp xếp theo `score` giảm dần
- `answer` được viết bằng Markdown, có thể dùng `[1]`, `[2]`... để tham chiếu tới `chunks`

---

### 3.3 Schema: Message (lịch sử hội thoại)

```python
Message = {
    "role":    str,   # "user" | "assistant"
    "content": str,   # Nội dung tin nhắn
}
```

---

## 4. Đặc tả Chi tiết từng Module

---

### TV1 — Data Processing (`src/data_processing/`)

#### Nhiệm vụ
1. Thu thập dữ liệu pháp luật và bài báo vi phạm pháp luật
2. Chuẩn hóa sang định dạng Markdown
3. Chunking văn bản
4. Tạo embeddings và lưu vào ChromaDB

#### Đầu ra bắt buộc (TV2 và TV3 phụ thuộc vào đây)

**A. File hệ thống:**
```
data/
├── standardized/
│   ├── legal/          # Các văn bản luật dạng .md
│   │   └── <ten_van_ban>.md
│   └── news/           # Bài báo dạng .md (frontmatter YAML)
│       └── <slug_bai_bao>.md
└── vectorstore/
    └── chroma/         # ChromaDB persistent storage
        ├── chroma.sqlite3
        └── ...
```

**B. Định dạng file `.md` cho văn bản luật:**
```markdown
# Tên văn bản luật

**Loại:** Luật / Nghị định / Thông tư  
**Số hiệu:** 100/2015/QH13  
**Ngày ban hành:** 2015-11-27  

## Điều 1. Phạm vi điều chỉnh
...nội dung...

## Điều 2. Đối tượng áp dụng
...nội dung...
```

**C. Định dạng file `.md` cho bài báo (bắt buộc có YAML frontmatter):**
```markdown
---
title: "Tiêu đề bài báo"
url: "https://..."
date: "2024-03-15"
source: "ten_file.md"
type: "news"
---

Nội dung bài báo...
```

**D. ChromaDB collection:** tên collection phải là `"legal_chunks"`, dùng `cosine` distance metric.

**E. Metadata mỗi document trong ChromaDB** phải theo đúng schema `Chunk.metadata` ở mục 3.1.

#### Thông số Chunking
- Chunk size: 512 tokens (khoảng 1000-1500 ký tự tiếng Việt)
- Overlap: 64 tokens (~128 ký tự)
- Mỗi chunk là 1 document trong ChromaDB

#### Kiểm tra đầu ra (TV1 tự kiểm)
```python
# Chạy script này để verify trước khi bàn giao
import chromadb
client = chromadb.PersistentClient(path="data/vectorstore/chroma")
col = client.get_collection("legal_chunks")
results = col.query(query_texts=["phạt tù"], n_results=3)
# Phải có kết quả, metadata phải đủ các field theo schema
print(results["metadatas"])
```

---

### TV2 — RAG Pipeline (`src/rag_pipeline/`)

#### Codebase hiện có (đã implement)
- `semantic_search.py` — Dense search qua ChromaDB + OpenAI embeddings
- `lexical_search.py` — BM25 keyword search
- `reranking.py` — Cross-encoder, MMR, RRF reranking
- `pageindex_vectorless.py` — Fallback không cần vector
- `retrieval_pipeline.py` — Orchestrator tổng hợp

#### Nhiệm vụ còn lại
Thêm module `generation.py` để sinh câu trả lời từ chunks đã retrieve:

**File cần tạo: `src/rag_pipeline/generation.py`**

```python
# Giao diện bắt buộc — TV3 sẽ gọi hàm này
def generate(
    query: str,
    chunks: list[dict],      # List[Chunk] từ retrieve()
    history: list[dict],     # List[Message]
    model: str = "claude-sonnet-4-6"
) -> dict:                   # Trả về BotResponse
    ...
```

**System prompt bắt buộc:**
```
Bạn là trợ lý pháp luật. Chỉ trả lời các câu hỏi liên quan đến:
1. Quy định pháp luật Việt Nam
2. Vụ việc vi phạm pháp luật được đăng trên báo chí

Nếu câu hỏi nằm ngoài phạm vi này, từ chối lịch sự.
Luôn trích dẫn nguồn bằng ký hiệu [1], [2]... tương ứng với thứ tự chunks được cung cấp.
Trả lời bằng tiếng Việt, rõ ràng, ngắn gọn.
```

**Entry point chính TV3 gọi:**

```python
# src/rag_pipeline/pipeline.py  ← file mới, TV3 chỉ import file này
from rag_pipeline.retrieval_pipeline import retrieve
from rag_pipeline.generation import generate

def chat(
    query: str,
    history: list[dict]   # List[Message]
) -> dict:                # Trả về BotResponse
    chunks = retrieve(query, top_k=5)
    return generate(query, chunks, history)
```

#### Đầu ra bắt buộc (TV3 phụ thuộc vào đây)

Module `src/rag_pipeline/pipeline.py` export hàm `chat()` với signature trên.  
Trả về đúng schema `BotResponse` ở mục 3.2.

---

### TV3 — Interface (`src/interface/`)

#### Nhiệm vụ
Xây dựng giao diện người dùng hiển thị chat + citations.

#### Đầu vào (TV2 cung cấp)
```python
from rag_pipeline.pipeline import chat
# chat(query, history) → BotResponse
```

#### Yêu cầu giao diện

**A. Layout chính:**
```
┌─────────────────────────────────────────────────────┐
│  🏛️  LegalBot — Chatbot Tư vấn Pháp luật            │
├────────────────────────────┬────────────────────────┤
│                            │                        │
│   CHAT WINDOW              │   CITATIONS PANEL      │
│                            │                        │
│  [User]: câu hỏi...        │  [1] Nguồn: ...        │
│                            │      Đoạn trích: ...   │
│  [Bot]: câu trả lời [1][2] │                        │
│                            │  [2] Nguồn: ...        │
│                            │      Đoạn trích: ...   │
│  ________________________  │                        │
│  [Nhập câu hỏi...] [Gửi]   │                        │
└────────────────────────────┴────────────────────────┘
```

**B. Citation Card** — mỗi chunk hiển thị như một card:

| Field hiển thị | Lấy từ |
|---|---|
| Số thứ tự `[1]` | Thứ tự trong `BotResponse.chunks` |
| Tiêu đề | `chunk.metadata.title` |
| Loại nguồn | `chunk.metadata.type` → "Văn bản luật" / "Báo chí" |
| Ngày | `chunk.metadata.date` |
| Link | `chunk.metadata.url` (chỉ hiện nếu `type == "news"`) |
| Điều khoản | `chunk.metadata.article_id` (chỉ hiện nếu `type == "legal"`) |
| Trích dẫn | 200 ký tự đầu của `chunk.content` + "..." |
| Độ tin cậy | `chunk.score` hiển thị dạng thanh tiến trình hoặc % |

**C. Xử lý `out_of_scope`:**
- Nếu `BotResponse.out_of_scope == True`: hiển thị message màu vàng, không có citation panel

**D. File cần tạo:**
```
src/interface/
├── app.py          # Entry point, chạy: streamlit run src/interface/app.py
├── components.py   # CitationCard, ChatMessage, OutOfScopeAlert
└── styles.css      # CSS tuỳ chỉnh (nếu dùng Streamlit custom CSS)
```

---

## 5. Môi trường & Cấu hình

### Biến môi trường (file `.env` — không commit lên git)

| Biến | Dùng bởi | Bắt buộc |
|---|---|---|
| `OPENAI_API_KEY` | TV2 (embeddings + generation) | ✅ |
| `ANTHROPIC_API_KEY` | TV2 (generation với Claude) | ✅ nếu dùng Claude |
| `JINA_API_KEY` | TV2 (reranking nâng cao) | ❌ optional |
| `PAGEINDEX_API_KEY` | TV2 (vectorless fallback) | ❌ optional |

### Cài đặt môi trường
```bash
# Tạo virtual env
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Mac/Linux

# Cài dependencies
pip install -r requirements.txt
```

### `requirements.txt` (TV2 chịu trách nhiệm maintain)
```
openai>=1.0.0
anthropic>=0.25.0
chromadb>=0.4.0
rank-bm25>=0.2.2
numpy>=1.24.0
python-dotenv>=1.0.0
streamlit>=1.35.0
requests>=2.31.0
```

---

## 6. Luồng Tích hợp (Integration Flow)

### Bước 1 — TV1 bàn giao TV2
TV1 hoàn thành khi:
- [ ] `data/vectorstore/chroma/` tồn tại và có ít nhất 100 documents
- [ ] `data/standardized/legal/` có ít nhất 5 file `.md`
- [ ] `data/standardized/news/` có ít nhất 10 file `.md`
- [ ] Chạy script kiểm tra (mục 4, phần TV1) không có lỗi
- [ ] File `data/DATA_REPORT.md` ghi số lượng docs, sources, ngày thu thập

### Bước 2 — TV2 bàn giao TV3
TV2 hoàn thành khi:
- [ ] `src/rag_pipeline/pipeline.py` tồn tại, export hàm `chat()`
- [ ] `chat("phạt tù treo là gì?", [])` trả về `BotResponse` đúng schema
- [ ] `chat("thời tiết hôm nay thế nào?", [])` trả về `out_of_scope=True`
- [ ] `BotResponse.chunks` không rỗng khi có câu hỏi hợp lệ
- [ ] File `src/rag_pipeline/PIPELINE_TEST.md` ghi kết quả test 5 câu hỏi mẫu

### Bước 3 — TV3 hoàn thiện
TV3 hoàn thành khi:
- [ ] `streamlit run src/interface/app.py` chạy không lỗi
- [ ] Citation cards hiển thị đủ field theo mục 4 phần TV3
- [ ] Câu hỏi out-of-scope hiển thị cảnh báo rõ ràng
- [ ] Lịch sử hội thoại được giữ trong session

---

## 7. Quy trình Làm việc Nhóm

### Nhánh Git
```
main        ← Production, chỉ merge khi đã test
dev         ← Integration branch
tv1/data    ← TV1 làm việc trên đây
tv2/rag     ← TV2 làm việc trên đây  
tv3/ui      ← TV3 làm việc trên đây
```

### Quy tắc Commit
```
feat(data): add legal document scraper
feat(rag): implement generation module
feat(ui): add citation card component
fix(rag): handle empty chunks in generate()
```

### Milestone
| Ngày | Mốc |
|---|---|
| T+3 ngày | TV1 có ≥ 50 docs trong ChromaDB, TV2 hoàn thiện retrieve() |
| T+5 ngày | TV2 hoàn thiện chat(), TV3 có UI prototype hiển thị được |
| T+7 ngày | Integration test, fix bugs |
| T+8 ngày | Demo sẵn sàng |

---

## 8. Câu hỏi Test Mẫu

Dùng để kiểm tra pipeline end-to-end:

| # | Câu hỏi | Kỳ vọng |
|---|---|---|
| 1 | "Tội danh cướp giật tài sản bị phạt tù bao nhiêu năm?" | `out_of_scope=False`, citation từ Bộ luật Hình sự |
| 2 | "Vụ án Nguyễn Phương Hằng vi phạm pháp luật gì?" | `out_of_scope=False`, citation từ bài báo |
| 3 | "Điều kiện để được hưởng án treo là gì?" | `out_of_scope=False`, citation từ văn bản luật |
| 4 | "Hôm nay thời tiết thế nào?" | `out_of_scope=True` |
| 5 | "Nấu ăn món gì ngon?" | `out_of_scope=True` |

---

## 9. Câu hỏi Thường gặp (FAQ nhóm)

**Q: TV3 có thể bắt đầu làm UI trước khi TV2 xong không?**  
A: Có. TV3 dùng mock data theo đúng schema `BotResponse` để develop UI song song. Khi TV2 xong, chỉ cần thay mock bằng `from rag_pipeline.pipeline import chat`.

**Q: TV2 có thể test retrieve() trước khi TV1 có đủ data không?**  
A: Có. TV2 dùng bất kỳ `.md` file nào làm test data, miễn đúng format. Quan trọng là ChromaDB collection phải tên `"legal_chunks"`.

**Q: Nếu schema cần thay đổi thì làm thế nào?**  
A: Tạo issue trên GitHub, tag cả nhóm, chờ đồng ý trước khi sửa. Không tự ý đổi schema.
