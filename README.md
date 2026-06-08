# LegalBot — Chatbot Tư vấn Pháp luật RAG

Dự án môn học AI in Action — Nhóm Gốc Lớp, Day 08.

---

## Mô tả Đề tài

**LegalBot** là chatbot hỏi-đáp pháp luật Việt Nam sử dụng kiến trúc **RAG (Retrieval-Augmented Generation)**. Hệ thống chỉ trả lời các câu hỏi thuộc phạm vi:

- Quy định pháp luật Việt Nam (luật, nghị định, thông tư)
- Thông tin các vụ vi phạm pháp luật từ báo chí

Mọi câu trả lời đều kèm **citation** (trích dẫn nguồn) để đảm bảo độ tin cậy và truy xuất nguồn gốc.

### Kiến trúc hệ thống

```
Người dùng
    │ câu hỏi (tiếng Việt)
    ▼
┌─────────────────────────────────────────────────┐
│  Interface (Streamlit)                           │
│  • Chat window + citation cards                  │
│  • Quản lý lịch sử hội thoại                    │
└──────────────────────┬──────────────────────────┘
                       │ query + history
                       ▼
┌─────────────────────────────────────────────────┐
│  RAG Pipeline                                    │
│  • Hybrid search: Semantic + BM25 + RRF          │
│  • Reranking (Jina cross-encoder)                │
│  • Fallback PageIndex (vectorless)               │
│  • Generation với citation (GPT-4o-mini)         │
└──────────────────────┬──────────────────────────┘
                       │ truy vấn ChromaDB
                       ▼
┌─────────────────────────────────────────────────┐
│  Data Processing                                 │
│  • Thu thập & chuẩn hóa dữ liệu                 │
│  • Chunking + embedding → ChromaDB               │
└─────────────────────────────────────────────────┘
```

---

## Thành viên & Đóng góp

| # | Họ và tên | MSSV | Module | Thư mục |
|---|---|---|---|---|
| 1 | Lê Vũ Anh | 2A202600809 | Data Processing | `src/data_processing/` |
| 2 | Lê Trung Kiên | 2A202600834 | RAG Pipeline | `src/rag_pipeline/` |
| 3 | Đỗ Thị Thanh Bình | 2A202600717 | Interface | `src/interface/` |

### Thành viên 1 — Lê Vũ Anh (Data Processing)

- Thu thập dữ liệu văn bản luật và bài báo vi phạm pháp luật
- Chuẩn hóa dữ liệu sang định dạng Markdown (có YAML frontmatter cho bài báo)
- Chunking văn bản với `RecursiveCharacterTextSplitter` (chunk 1200 ký tự, overlap 128)
- Tạo embeddings bằng `text-embedding-3-small` (OpenAI, 1536 chiều)
- Lưu vào ChromaDB collection `"legal_chunks"` với cosine similarity

### Thành viên 2 — Lê Trung Kiên (RAG Pipeline)

- **Task 5** — Semantic search: truy vấn ChromaDB bằng cosine similarity
- **Task 6** — Lexical search: BM25Okapi với whitespace tokenization tiếng Việt
- **Task 7** — Reranking: RRF (Reciprocal Rank Fusion) + Jina cross-encoder
- **Task 8** — PageIndex fallback: vectorless search khi hybrid score thấp
- **Task 9** — Retrieval pipeline: query expansion + continuation chunks + diversification
- **Task 10** — Generation: GPT-4o-mini với citation, lost-in-the-middle prevention, multi-turn memory

### Thành viên 3 — Đỗ Thị Thanh Bình (Interface)

- Giao diện chat Streamlit với chat window + citation panel
- Citation cards hiển thị nguồn, loại tài liệu, ngày, URL, trích dẫn, độ tin cậy
- Xử lý trạng thái `out_of_scope` (hiển thị cảnh báo, không có citation)
- Quản lý lịch sử hội thoại trong session

---

## Cấu trúc Thư mục

```
.
├── data/
│   ├── standardized/
│   │   ├── legal/          # Văn bản luật .md
│   │   └── news/           # Bài báo .md (YAML frontmatter)
│   └── vectorstore/
│       └── chroma/         # ChromaDB persistent storage
├── src/
│   ├── data_processing/    # TV1: thu thập + chunking + embedding
│   │   ├── config.py
│   │   ├── chunk_and_index.py
│   │   └── crawlers/
│   ├── rag_pipeline/       # TV2: retrieve + generate
│   │   ├── semantic_search.py
│   │   ├── lexical_search.py
│   │   ├── reranking.py
│   │   ├── pageindex_vectorless.py
│   │   ├── retrieval_pipeline.py
│   │   ├── generation.py
│   │   └── pipeline.py     # Entry point cho TV3: chat()
│   └── interface/          # TV3: Streamlit UI
│       ├── app.py
│       ├── components.py
│       └── styles.css
├── test/
│   ├── cli_rag.py          # CLI test (retrieve-only hoặc full RAG)
│   └── test_root_pipeline.py
├── sessions/               # Lưu lịch sử hội thoại (JSON)
├── PROJECT_SPEC.md         # Đặc tả kỹ thuật chi tiết
├── requirements.txt
└── .env                    # API keys (không commit)
```

---

## Công nghệ Sử dụng

| Thành phần | Công nghệ |
|---|---|
| Embedding model | OpenAI `text-embedding-3-small` (1536 dim) |
| Vector store | ChromaDB (cosine similarity, persistent) |
| Lexical search | BM25Okapi (`rank-bm25`) |
| Reranker | Jina Reranker v2 (cross-encoder API) |
| Score fusion | Reciprocal Rank Fusion (RRF) |
| Generation | GPT-4o-mini (`temperature=0.3`, `top_p=0.9`) |
| Interface | Streamlit |
| Language | Python 3.10+ |

---

## Cài đặt & Chạy

### 1. Cài đặt môi trường

```bash
# Dùng conda (khuyến nghị — env ai20k)
conda activate ai20k

# Hoặc tạo venv mới
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### 2. Cấu hình API keys

Tạo file `.env` ở thư mục gốc:

```env
OPENAI_API_KEY=sk-...
JINA_API_KEY=...        # Optional: reranking nâng cao
PAGEINDEX_API_KEY=...   # Optional: vectorless fallback
```

### 3. Index dữ liệu (TV1)

```bash
python src/data_processing/chunk_and_index.py
```

### 4. Chạy giao diện (TV3)

```bash
streamlit run src/interface/app.py
```

### 5. Test CLI (không cần UI)

```bash
# Chỉ retrieve (không cần OPENAI_API_KEY cho generation)
python test/cli_rag.py --retrieve

# Full RAG (retrieve + generate)
python test/cli_rag.py
```

---

## Schema Dữ liệu

### Chunk

```python
{
    "content":  str,          # Nội dung đoạn văn bản
    "score":    float,        # Điểm liên quan [0.0, 1.0]
    "metadata": {
        "source":      str,   # Tên file .md
        "type":        str,   # "legal" | "news"
        "chunk_index": int,   # Thứ tự chunk trong file (0-indexed)
        "url":         str,   # URL nguồn (bài báo)
        "title":       str,   # Tiêu đề
        "date":        str,   # "YYYY-MM-DD" hoặc ""
        "article_id":  str,   # Số điều luật hoặc ""
    },
    "source": str             # "hybrid" | "pageindex"
}
```

### BotResponse

```python
{
    "answer":       str,          # Câu trả lời Markdown có citation
    "chunks":       list[Chunk],  # Tối đa 5 chunks, score giảm dần
    "out_of_scope": bool,         # True nếu câu hỏi ngoài phạm vi pháp luật
    "query":        str,          # Câu hỏi gốc của người dùng
}
```

---

## Câu hỏi Test Mẫu

| Câu hỏi | Kỳ vọng |
|---|---|
| "Tội tàng trữ trái phép chất ma tuý bị phạt như thế nào?" | `out_of_scope=False`, citation từ Bộ luật Hình sự |
| "Những nghệ sĩ nào đã bị bắt vì liên quan ma tuý?" | `out_of_scope=False`, citation từ bài báo |
| "Điều kiện để được hưởng án treo là gì?" | `out_of_scope=False`, citation từ văn bản luật |
| "Hôm nay thời tiết thế nào?" | `out_of_scope=True` |
| "Nấu ăn món gì ngon?" | `out_of_scope=True` |

---

## Liên kết

- Đặc tả kỹ thuật chi tiết: [PROJECT_SPEC.md](PROJECT_SPEC.md)
- Môn học: AI in Action — Day 08
