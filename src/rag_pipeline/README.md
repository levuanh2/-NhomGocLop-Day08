# RAG Pipeline — Quy trình Hoạt động

## Sơ đồ Tổng thể

```
                        ┌─────────────────────┐
                        │    Câu hỏi người dùng│
                        └──────────┬──────────┘
                                   │
                                   ▼
                     ┌─────────────────────────┐
                     │     Query Expansion      │
                     │  (nếu có từ "là gì",     │
                     │  "định nghĩa", v.v.)      │
                     │  → thêm 1-2 query variant│
                     └──────────┬──────────────┘
                                │
               ┌────────────────┴────────────────┐
               │                                 │
               ▼                                 ▼
  ┌────────────────────────┐       ┌─────────────────────────┐
  │    Semantic Search     │       │     Lexical Search       │
  │  (semantic_search.py)  │       │   (lexical_search.py)    │
  │                        │       │                          │
  │  Query → Embedding     │       │  BM25 trên toàn corpus   │
  │  (OpenAI              │       │  k1=1.5, b=0.75          │
  │  text-embedding-3-    │       │                          │
  │  small, 1536 dim)     │       │  Tìm theo từ khóa        │
  │                        │       │  (không cần embedding)   │
  │  ChromaDB cosine sim  │       │                          │
  │  → top_k * 4 chunks   │       │  → top_k * 4 chunks      │
  └───────────┬────────────┘       └────────────┬────────────┘
              │                                 │
              └────────────────┬────────────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │    RRF Merge           │
                  │   (reranking.py)       │
                  │                        │
                  │  Reciprocal Rank Fusion│
                  │  score = Σ 1/(60+rank) │
                  │                        │
                  │  Kết hợp 2 danh sách   │
                  │  → dedup theo content  │
                  └────────────┬───────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │    Source Diversify    │
                  │                        │
                  │  Tối đa 3 chunks       │
                  │  từ cùng 1 file nguồn  │
                  │  → tránh kết quả       │
                  │    toàn từ 1 bài báo   │
                  └────────────┬───────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │      Reranking         │
                  │   (reranking.py)       │
                  │                        │
                  │  Cross-encoder         │
                  │  (Jina multilingual)   │
                  │                        │
                  │  Nếu không có API key  │
                  │  → fallback: sort score│
                  └────────────┬───────────┘
                               │
               ┌───────────────┴──────────────┐
               │  score >= 0.008?              │
               │                               │
          YES   ▼                         NO    ▼
  ┌─────────────────────┐      ┌──────────────────────────┐
  │   Kết quả Hybrid    │      │   PageIndex Fallback      │
  │  source = "hybrid"  │      │ (pageindex_vectorless.py) │
  │                     │      │                           │
  │  top_k chunks       │      │  Vectorless RAG           │
  │  có score, metadata │      │  → poll API cho kết quả   │
  └─────────────────────┘      │  source = "pageindex"     │
                                └──────────────────────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │    Generation          │
                  │   (generation.py)      │
                  │                        │
                  │  Reorder chunks        │
                  │  (tránh lost-in-middle)│
                  │                        │
                  │  Format context        │
                  │  [Doc 1 | Source: ...] │
                  │  [Doc 2 | Source: ...] │
                  │                        │
                  │  GPT-4o-mini           │
                  │  temp=0.3, top_p=0.9   │
                  └────────────┬───────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │  Câu trả lời + Citation │
                  │                        │
                  │  answer: str           │
                  │  sources: List[Chunk]  │
                  └────────────────────────┘
```

---

## Các Module

| File | Nhiệm vụ |
|---|---|
| `semantic_search.py` | Dense retrieval — embed query bằng OpenAI, tìm trong ChromaDB |
| `lexical_search.py` | Sparse retrieval — BM25 trên toàn corpus |
| `reranking.py` | RRF merge + Cross-encoder rerank (Jina) + MMR |
| `pageindex_vectorless.py` | Fallback khi confidence thấp, không cần vector |
| `retrieval_pipeline.py` | Orchestrator — kết nối toàn bộ pipeline trên |
| `generation.py` | Gọi LLM sinh câu trả lời có citation từ chunks |

---

## Định dạng Chunk (đầu ra của retrieve)

```python
{
    "content":  str,    # Nội dung đoạn văn bản
    "score":    float,  # Điểm liên quan (RRF hoặc cosine sim)
    "metadata": {
        "source":      str,  # Tên file, VD: "Luật hình sự.md"
        "type":        str,  # "legal" | "news"
        "chunk_index": int,  # Thứ tự chunk trong file
        "url":         str,  # URL bài báo (nếu type="news")
    },
    "source":   str,    # "hybrid" | "pageindex"
}
```

---

## Biến Môi trường Cần thiết

| Biến | Dùng bởi | Bắt buộc |
|---|---|---|
| `OPENAI_API_KEY` | `semantic_search.py`, `generation.py` | Có |
| `JINA_API_KEY` | `reranking.py` (cross-encoder) | Không — tự fallback |
| `PAGEINDEX_API_KEY` | `pageindex_vectorless.py` | Không — chỉ dùng khi fallback |

---

## Chạy Nhanh (CLI Test)

```bash
# Retrieve only — xem chunks trả về, không gọi LLM
PYTHONUTF8=1 /c/Users/Admin/miniconda3/envs/ai20k/python.exe test/cli_rag.py --retrieve

# Full pipeline — retrieve + sinh câu trả lời
PYTHONUTF8=1 /c/Users/Admin/miniconda3/envs/ai20k/python.exe test/cli_rag.py
```
