# LegalBot — Chatbot Tư vấn Pháp luật Việt Nam

Dự án môn học **AI in Action — Nhóm Gốc Lớp, Day 08**.

LegalBot là chatbot hỏi-đáp pháp luật Việt Nam sử dụng kiến trúc **RAG (Retrieval-Augmented Generation)**. Hệ thống trả lời câu hỏi về quy định pháp luật và các vụ vi phạm từ báo chí, luôn kèm **trích dẫn nguồn** cho mọi câu trả lời.

---

## Mục lục

1. [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
2. [Thành viên & Đóng góp](#thành-viên--đóng-góp)
3. [Cấu trúc thư mục](#cấu-trúc-thư-mục)
4. [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
5. [Cài đặt môi trường](#cài-đặt-môi-trường)
6. [Cấu hình API Keys](#cấu-hình-api-keys)
7. [Hướng dẫn chạy chi tiết](#hướng-dẫn-chạy-chi-tiết)
   - [Bước 1 — Chuẩn bị dữ liệu](#bước-1--chuẩn-bị-dữ-liệu)
   - [Bước 2 — Chunking & Embedding](#bước-2--chunking--embedding)
   - [Bước 3 — Chạy giao diện](#bước-3--chạy-giao-diện)
   - [Bước 4 — Test CLI](#bước-4--test-cli)
8. [Công nghệ sử dụng](#công-nghệ-sử-dụng)
9. [Pipeline chi tiết](#pipeline-chi-tiết)
10. [Schema dữ liệu](#schema-dữ-liệu)
11. [Câu hỏi test mẫu](#câu-hỏi-test-mẫu)
12. [Deploy lên Hugging Face Spaces](#deploy-lên-hugging-face-spaces)

---

## Kiến trúc hệ thống

```
Người dùng (câu hỏi tiếng Việt)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Interface (Streamlit)                               │
│  • 3-column layout: sidebar | chat | citation panel │
│  • Lịch sử hội thoại persistent (JSON)              │
└────────────────────────┬────────────────────────────┘
                         │ query + history
                         ▼
┌─────────────────────────────────────────────────────┐
│  RAG Pipeline                                        │
│  1. Query expansion (định nghĩa → từ khoá luật)     │
│  2. HyDE — Hypothetical Document Embedding           │
│  3. Semantic search (OpenAI embed → ChromaDB)       │
│  4. Lexical search (BM25Okapi)                      │
│  5. RRF merge + source diversification              │
│  6. Reranking (flashrank cross-encoder ONNX)        │
│  7. Fallback PageIndex (vectorless)                 │
│  8. Continuation chunks (3 chunk kế tiếp)           │
│  9. Generation — GPT-4o-mini với citation           │
└────────────────────────┬────────────────────────────┘
                         │ ChromaDB query
                         ▼
┌─────────────────────────────────────────────────────┐
│  Data Layer                                          │
│  • ChromaDB: 477 chunks, cosine similarity          │
│  • 10 bài báo + 7 văn bản pháp luật                 │
│  • Embedding: text-embedding-3-small (1536 dim)     │
└─────────────────────────────────────────────────────┘
```

---

## Thành viên & Đóng góp

| # | Họ và tên | MSSV | Module | Thư mục |
|---|---|---|---|---|
| 1 | Lê Vũ Anh | 2A202600809 | Data Processing | `src/data_processing/` |
| 2 | Lê Trung Kiên | 2A202600834 | RAG Pipeline | `src/rag_pipeline/` |
| 3 | Đỗ Thị Thanh Bình | 2A202600717 | Interface | `src/interface/` |

### Lê Vũ Anh — Data Processing (Task 1–4)

- Thu thập dữ liệu: crawl văn bản luật và bài báo vi phạm pháp luật
- Chuẩn hóa sang Markdown với YAML frontmatter (`title`, `url`, `date`, `source`)
- Chunking bằng `RecursiveCharacterTextSplitter` — chunk 1200 ký tự, overlap 128
- Embedding với `text-embedding-3-small` (OpenAI, 1536 chiều), lưu vào ChromaDB

### Lê Trung Kiên — RAG Pipeline (Task 5–10)

- **Task 5** — Semantic search: ChromaDB cosine similarity query
- **Task 6** — Lexical search: BM25Okapi với Vietnamese normalization
- **Task 7** — Reranking: RRF + flashrank cross-encoder ONNX (không cần GPU)
- **Task 8** — PageIndex vectorless fallback khi score < 0.008
- **Task 9** — Retrieval pipeline: HyDE + query expansion + diversification + continuation chunks
- **Task 10** — Generation: GPT-4o-mini, citation format, lost-in-middle prevention, multi-turn memory

### Đỗ Thị Thanh Bình — Interface (Group Project)

- Giao diện chat Streamlit 3 cột: sidebar | chat | citation panel
- Citation cards với badge loại tài liệu, metadata, preview, progress bar độ tin cậy
- Chat history persistent (JSON), session state management
- Out-of-scope detection với cảnh báo rõ ràng

---

## Cấu trúc thư mục

```
.
├── app.py                          # Entry point (HF Spaces + local)
├── requirements.txt
├── .env                            # API keys (KHÔNG commit)
├── .env.example                    # Template .env
├── README.md                       # Tài liệu này
├── README_HF.md                    # Metadata Hugging Face Spaces
│
├── data/
│   ├── standardized/
│   │   ├── legal/                  # Văn bản pháp luật .md
│   │   │   ├── luat_phong_chong_ma_tuy_2021.md
│   │   │   ├── bo_luat_hinh_su_2015_chuong_xx.md
│   │   │   └── ...                 # 7 văn bản (8 file, 1 trùng bị skip)
│   │   └── news/                   # Bài báo .md (YAML frontmatter)
│   │       ├── ca_si_miu_le_bi_bat_*.md
│   │       └── ...                 # 10 bài báo
│   └── vectorstore/
│       └── chroma/                 # ChromaDB persistent (477 chunks)
│
├── src/
│   ├── data_processing/
│   │   ├── config.py               # Paths, chunk params, model config
│   │   ├── chunk_and_index.py      # MAIN: load → chunk → embed → ChromaDB
│   │   ├── convert_markdown.py     # Chuyển đổi định dạng tài liệu
│   │   ├── collect_data.py         # Thu thập dữ liệu
│   │   └── crawl_more_news.py      # Crawl thêm bài báo
│   │
│   ├── rag_pipeline/
│   │   ├── pipeline.py             # Entry point: chat(query, history)
│   │   ├── retrieval_pipeline.py   # Orchestrator: HyDE + hybrid + rerank + fallback
│   │   ├── semantic_search.py      # ChromaDB cosine similarity
│   │   ├── lexical_search.py       # BM25Okapi
│   │   ├── reranking.py            # RRF + flashrank cross-encoder + MMR
│   │   ├── pageindex_vectorless.py # PageIndex API fallback
│   │   └── generation.py           # GPT-4o-mini với citation
│   │
│   └── interface/
│       ├── app.py                  # Streamlit main app
│       ├── layout.py               # 3-column layout
│       ├── components.py           # Header, chat panel, citation cards
│       ├── state.py                # Session state helpers
│       ├── chat_store.py           # JSON chat history persistence
│       ├── formatters.py           # Format text, scores, dates
│       └── title_utils.py          # Chat title generation
│
├── test/
│   ├── cli_rag.py                  # CLI test: retrieve-only hoặc full RAG
│   └── README.md                   # Hướng dẫn test CLI
│
└── sessions/                       # Lịch sử CLI sessions (JSON, auto-created)
```

---

## Yêu cầu hệ thống

| Thành phần | Yêu cầu |
|---|---|
| Python | 3.10+ |
| RAM | ≥ 4 GB (flashrank ONNX model ~34 MB) |
| Disk | ≥ 500 MB (ChromaDB + models) |
| Internet | Cần thiết khi embedding và generation |
| GPU | Không cần (flashrank chạy CPU-only) |

---

## Cài đặt môi trường

### Dùng conda (khuyến nghị)

```bash
# Activate conda env đã có
conda activate ai20k

# Hoặc tạo env mới
conda create -n legalbot python=3.11
conda activate legalbot
pip install -r requirements.txt
```

### Dùng venv

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

> **Lưu ý Windows:** Nếu gặp lỗi encoding khi chạy, thêm biến môi trường `PYTHONUTF8=1` trước lệnh Python.

---

## Cấu hình API Keys

Tạo file `.env` ở thư mục gốc (copy từ `.env.example`):

```bash
cp .env.example .env   # Mac/Linux
copy .env.example .env  # Windows
```

Nội dung file `.env`:

```env
# BẮT BUỘC — dùng cho embedding và generation
OPENAI_API_KEY=sk-proj-...

# TÙY CHỌN — reranking nâng cao (flashrank local nếu không có key này)
JINA_API_KEY=jina_...

# TÙY CHỌN — vectorless fallback khi hybrid score thấp
PAGEINDEX_API_KEY=...

# TÙY CHỌN — nếu dùng OpenAI-compatible endpoint khác
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

| Key | Bắt buộc? | Dùng cho |
|---|---|---|
| `OPENAI_API_KEY` | **Có** | Embedding (chunk_and_index) + Generation (GPT-4o-mini) |
| `JINA_API_KEY` | Không | Reranking cross-encoder API (fallback sang flashrank local nếu thiếu) |
| `PAGEINDEX_API_KEY` | Không | Vectorless fallback (bỏ qua nếu thiếu) |

---

## Hướng dẫn chạy chi tiết

### Bước 1 — Chuẩn bị dữ liệu

Tài liệu đã được chuẩn hóa sẵn trong thư mục `data/standardized/`:

```
data/standardized/
├── legal/     ← 7 văn bản pháp luật (.md)
└── news/      ← 10 bài báo (.md với YAML frontmatter)
```

**Format file `.md` cho bài báo** (phải có YAML frontmatter):

```markdown
---
title: "Tiêu đề bài báo"
url: "https://..."
date: "2024-01-15"
source: "tuoitre.vn"
---

Nội dung bài báo...
```

**Format file `.md` cho văn bản luật** (có hoặc không có frontmatter):

```markdown
---
title: "Luật Phòng, chống ma túy 2021"
---

## Điều 1. Phạm vi điều chỉnh
...
```

Để thêm tài liệu mới: đặt file `.md` đúng định dạng vào thư mục tương ứng rồi chạy lại **Bước 2**.

---

### Bước 2 — Chunking & Embedding

Đây là bước quan trọng nhất: đọc toàn bộ tài liệu trong `data/standardized/`, chia thành chunks, tạo embedding và lưu vào ChromaDB.

```bash
# Chạy từ thư mục gốc project
python -c "
import sys
sys.path.insert(0, 'src')
from data_processing.chunk_and_index import build_chroma_index
result = build_chroma_index(reset=True)
print(f\"Documents: {result['documents_loaded']}\")
print(f\"Chunks indexed: {result['chunks_indexed']}\")
"
```

Hoặc dùng Python shell:

```python
import sys
sys.path.insert(0, 'src')
from data_processing.chunk_and_index import build_chroma_index

# reset=True: xóa collection cũ và tạo lại từ đầu
result = build_chroma_index(reset=True)
```

**Output mong đợi:**

```
Loading standardized documents...
Loaded 17 documents. Chunking...
Embedding 477 chunks with model text-embedding-3-small...
Embedded 477 chunks.
Indexed 477 chunks into collection 'legal_chunks'.
```

**Tham số cấu hình** (trong `src/data_processing/config.py`):

| Tham số | Giá trị | Mô tả |
|---|---|---|
| `CHUNK_SIZE` | 1200 | Số ký tự tối đa mỗi chunk |
| `CHUNK_OVERLAP` | 128 | Overlap giữa các chunk liên tiếp |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `EMBEDDING_DIM` | 1536 | Số chiều embedding |
| `COLLECTION_NAME` | `legal_chunks` | Tên ChromaDB collection |

> **Lưu ý chi phí:** Embedding 477 chunks với `text-embedding-3-small` tốn khoảng **$0.001** (1 xu). Model này rẻ nhất của OpenAI.

> **Khi nào cần chạy lại:** Mỗi khi thêm, sửa, hoặc xóa file trong `data/standardized/`. Tham số `reset=True` sẽ xóa toàn bộ index cũ và tạo lại.

---

### Bước 3 — Chạy giao diện

#### Cách 1: Chạy từ root (khuyến nghị)

```bash
# Từ thư mục gốc project
streamlit run app.py
```

#### Cách 2: Chạy trực tiếp từ src

```bash
cd src
streamlit run interface/app.py
```

Trình duyệt tự động mở tại **http://localhost:8501**.

**Giao diện gồm 3 cột:**

- **Trái (23%)** — Sidebar: nút New Chat, lịch sử hội thoại, câu hỏi mẫu, cài đặt
- **Giữa (52%)** — Chat: nhập câu hỏi, xem câu trả lời có citation
- **Phải (25%)** — Citation panel: nguồn tài liệu, badge loại (📜 Luật / 📰 Báo), score

**Luồng xử lý khi người dùng gửi câu hỏi:**

```
Input → Query contextualization (rewrite ambiguous pronouns)
      → retrieve() → top-5 chunks (hybrid search + reranking)
      → _add_continuations() → thêm tối đa 3 chunk kế tiếp
      → generate_with_citation() → GPT-4o-mini → câu trả lời có [Source, Điều X]
      → out_of_scope check → hiển thị kết quả / cảnh báo
```

---

### Bước 4 — Test CLI

Test pipeline mà không cần mở trình duyệt:

#### Retrieve-only mode (không cần `OPENAI_API_KEY` cho generation)

Chỉ chạy bước retrieval — hiển thị các chunks tìm được kèm score và nguồn. Hữu ích để kiểm tra chất lượng index và retrieval mà không tốn token generation.

```bash
# Windows (thêm PYTHONUTF8=1 để hiển thị đúng tiếng Việt)
set PYTHONUTF8=1 && python test/cli_rag.py --retrieve

# Mac/Linux
PYTHONUTF8=1 python test/cli_rag.py --retrieve
```

**Output mẫu:**

```
══════════════════════════════════════════════════════════════════════
  LEGALBOT — Retrieve-Only Mode
══════════════════════════════════════════════════════════════════════
  Nhập câu hỏi để xem các chunks được retrieve.

Câu hỏi > tội tàng trữ ma túy bị phạt bao nhiêu năm?

──────────────────────────────────────────────────────────────────────

  [1] 📜 Luật  bo_luat_hinh_su_2015_chuong_xx.md  |  score=0.0156  |  via=hybrid
       Điều khoản: Điều 249
       "Điều 249. Tội tàng trữ trái phép chất ma túy..."
```

#### Full RAG mode (cần `OPENAI_API_KEY`)

Retrieve + Generate với GPT-4o-mini. Session được lưu tự động vào `sessions/`.

```bash
set PYTHONUTF8=1 && python test/cli_rag.py
```

**Output mẫu:**

```
  SOURCES (chunks đã retrieve):
  [1] 📜 Luật  bo_luat_hinh_su_2015_chuong_xx.md  |  score=0.0156

══════════════════════════════════════════════════════════════════════
  TRẢ LỜI
══════════════════════════════════════════════════════════════════════
Theo Điều 249 Bộ luật Hình sự 2015, tội tàng trữ trái phép chất ma túy bị phạt:
- Phạt tù từ 1 đến 5 năm (trường hợp thông thường)
- Phạt tù từ 5 đến 10 năm (có tình tiết tăng nặng)
...
[Bộ luật Hình sự 2015, Điều 249]
```

---

## Công nghệ sử dụng

| Thành phần | Công nghệ | Ghi chú |
|---|---|---|
| Embedding | OpenAI `text-embedding-3-small` | 1536 dim, cosine similarity |
| Vector store | ChromaDB | Persistent, local |
| Lexical search | BM25Okapi (`rank-bm25`) | Vietnamese normalization |
| Score fusion | Reciprocal Rank Fusion (RRF) | `score = 1/(60 + rank)` |
| Reranker | flashrank cross-encoder ONNX | CPU-only, ~34 MB, không cần GPU |
| Vectorless fallback | PageIndex API | Khi hybrid score < 0.008 |
| Generation | GPT-4o-mini | `temperature=0.3`, `top_p=0.9` |
| HyDE | GPT-4o-mini | Hypothetical Document Embeddings |
| Interface | Streamlit 1.35+ | 3-column layout |
| Text splitting | `langchain-text-splitters` | RecursiveCharacterTextSplitter |

---

## Pipeline chi tiết

### Retrieval Pipeline (`src/rag_pipeline/retrieval_pipeline.py`)

```
Query
  │
  ├─ [HyDE] Sinh hypothetical document → embed (thay cho embed query trực tiếp)
  │          └─ Fallback: dùng query gốc nếu OpenAI lỗi
  │
  ├─ Query expansion
  │    └─ Nếu query chứa "là gì / định nghĩa / được hiểu":
  │         thêm "giải thích từ ngữ X" và "X là" → BM25 khớp điều luật định nghĩa
  │
  ├─ Semantic Search (ChromaDB, top_k × 4 candidates)
  │    └─ embed(semantic_query) → cosine similarity → top results
  │
  ├─ Lexical Search (BM25Okapi, top_k × 4 candidates mỗi query)
  │    └─ Vietnamese normalization → BM25 scoring → top results
  │
  ├─ RRF Merge: score = Σ 1/(60 + rank) over semantic + lexical lists
  │
  ├─ Source Diversification: tối đa 3 chunks / file nguồn
  │
  ├─ Reranking (flashrank cross-encoder ONNX)
  │    └─ Fallback: sort by score nếu flashrank lỗi
  │
  └─ Threshold check: best_score < 0.008?
       └─ Có → PageIndex vectorless fallback
       └─ Không → return top_k results
```

### Generation Pipeline (`src/rag_pipeline/generation.py`)

```
(query, history)
  │
  ├─ Query contextualization: rewrite câu hỏi mơ hồ dựa trên 3 lượt history cuối
  │
  ├─ retrieve(query) → top-5 chunks
  │
  ├─ _add_continuations(): thêm tối đa 3 chunk kế tiếp từ cùng file nguồn
  │    └─ Lý do: một điều luật thường span nhiều chunks liên tiếp
  │
  ├─ Context trimming: cắt nếu > 24,000 ký tự
  │
  ├─ Lost-in-middle prevention: reorder [0,2,4,...] + đảo ngược [1,3,...]
  │    └─ Best chunks ở đầu và cuối, worst ở giữa → LLM chú ý tốt hơn
  │
  ├─ Format context: đánh số [1], [2],... kèm source, type, article_id
  │
  ├─ GPT-4o-mini call:
  │    • System prompt: rules about citation, exhaustive listing, out-of-scope detection
  │    • temperature=0.3, top_p=0.9
  │
  └─ Out-of-scope detection: câu trả lời bắt đầu bằng "NGOÀI PHẠM VI:"?
       └─ Có → out_of_scope=True, không hiển thị citation
```

---

## Schema dữ liệu

### Chunk (output của retrieval)

```python
{
    "content":  str,          # Nội dung đoạn văn bản (≥ 50 ký tự)
    "score":    float,        # Điểm liên quan (RRF scale: 0.0–0.016)
    "metadata": {
        "source":      str,   # Tên file .md (ví dụ: "luat_phong_chong_ma_tuy_2021.md")
        "type":        str,   # "legal" | "news"
        "chunk_index": int,   # Thứ tự chunk trong file (0-indexed)
        "url":         str,   # URL nguồn (bài báo), "" nếu là luật
        "title":       str,   # Tiêu đề tài liệu
        "date":        str,   # "YYYY-MM-DD" hoặc ""
        "article_id":  str,   # "Điều X" hoặc "" (chỉ có với văn bản luật)
    },
    "source":   str           # "hybrid" | "pageindex"
}
```

### BotResponse (output của generation)

```python
{
    "answer":       str,          # Câu trả lời Markdown, inline citation [Source, Điều X]
    "chunks":       list[Chunk],  # Tối đa 5 chunks, score giảm dần
    "out_of_scope": bool,         # True nếu câu hỏi ngoài phạm vi pháp luật
    "query":        str,          # Câu hỏi gốc
}
```

---

## Câu hỏi test mẫu

| Loại | Câu hỏi | Kỳ vọng |
|---|---|---|
| Hình phạt | `Tội tàng trữ trái phép chất ma tuý bị phạt như thế nào?` | `out_of_scope=False`, citation Bộ luật Hình sự |
| Định nghĩa | `Cai nghiện ma túy là gì?` | `out_of_scope=False`, citation Luật PCMT 2021 |
| Liệt kê | `Những nghệ sĩ nào đã bị bắt vì liên quan ma tuý?` | `out_of_scope=False`, citation từ bài báo |
| Sự kiện | `Ca sĩ Miu Lê bị bắt vì tội gì?` | `out_of_scope=False`, citation bài báo cụ thể |
| Out-of-scope | `Hôm nay thời tiết thế nào?` | `out_of_scope=True` |
| Out-of-scope | `Nấu ăn món gì ngon?` | `out_of_scope=True` |

---

## Deploy lên Hugging Face Spaces

Dự án đã được cấu hình sẵn cho Hugging Face Spaces.

### Cấu hình (`README_HF.md`)

```yaml
---
title: LegalBot
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
---
```

### Các bước deploy

1. Push code lên HF Spaces repository
2. Vào **Settings → Variables and secrets**, thêm:
   - `OPENAI_API_KEY` ← **bắt buộc**
   - `JINA_API_KEY` ← tùy chọn
   - `PAGEINDEX_API_KEY` ← tùy chọn
3. HF Spaces tự động chạy `streamlit run app.py` trên port 7860

> **Lưu ý:** ChromaDB vector store đã được commit vào repo tại `data/vectorstore/chroma/`. HF Spaces sẽ dùng index có sẵn này — không cần chạy lại embedding trên Spaces.

---

## Troubleshooting

### Lỗi encoding trên Windows

```
UnicodeEncodeError: 'charmap' codec can't encode character
```

**Giải pháp:**

```bash
set PYTHONUTF8=1 && python test/cli_rag.py
# hoặc
$env:PYTHONUTF8=1; python test/cli_rag.py  # PowerShell
```

### Lỗi `chromadb.errors.InvalidCollectionException`

Collection chưa tồn tại. Chạy lại **Bước 2** để tạo index.

### Reranking bị skip

Không phải lỗi — flashrank cross-encoder tự khởi tạo lần đầu chạy (download model ~34 MB). Nếu có lỗi, pipeline fallback về sort-by-score tự động.

### Generation trả về "NGOÀI PHẠM VI:"

Câu hỏi không liên quan đến pháp luật hoặc các vụ vi phạm trong dữ liệu. Đây là hành vi đúng của hệ thống.

### Chunk count thấp hơn kỳ vọng

Một file có thể bị skip nếu:
- Trùng URL hoặc tiêu đề với file khác (deduplication)
- Body rỗng sau khi parse frontmatter
- Tất cả chunks < 50 ký tự

Xem log `[SKIP]` và `[WARN]` khi chạy `build_chroma_index`.
