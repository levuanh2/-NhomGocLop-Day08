# Day08 RAG Pipeline Report

## Implemented Tasks

- Task 1: Legal document collection (3 PDF/DOCX files)
- Task 2: News crawling (5 articles from VnExpress)
- Task 3: Markdown standardization (5 markdown files with frontmatter)
- Task 4: Chunking/Indexing (53 chunks, FAISS vector store)
- Task 5: Semantic search (dense retrieval with BAAI/bge-m3)
- Task 6: Lexical search (BM25 with rank-bm25 + fallback)
- Task 7: Reranking (RRF, MMR, cross-encoder fallback)
- Task 8: PageIndex vectorless fallback (graceful if no API key)
- Task 9: Retrieval pipeline (hybrid semantic + lexical + RRF)
- Task 10: Generation with citation (LLM + extractive fallback)

## Technical Choices

### Chunking

- Strategy: RecursiveCharacterTextSplitter
- chunk_size = 500
- chunk_overlap = 50
- Reason: Robust for mixed Vietnamese legal/news Markdown because it preserves paragraph boundaries while keeping chunks small enough for retrieval.

### Embedding

- Model: BAAI/bge-m3
- Embedding dimension: 1024
- Reason: bge-m3 supports multilingual retrieval and works well for Vietnamese.

### Vector Store

- FAISS IndexFlatIP
- Embeddings normalized, so inner product approximates cosine similarity.

### Lexical Search

- BM25 with rank-bm25
- Fallback keyword scoring if library missing.

### Reranking

- RRF (Reciprocal Rank Fusion) for merging semantic and lexical retrieval.
- MMR (Maximal Marginal Relevance) with cosine similarity when embeddings available.
- Cross-encoder fallback to original scores if no model/API configured.

### PageIndex

- Implemented as optional fallback.
- If PAGEINDEX_API_KEY is missing, functions return []/0 without crashing.

### Generation

- Citation-aware generation with OpenAI (gpt-4o-mini).
- Extractive fallback with source citations if no OPENAI_API_KEY.

## Test Results

```
pytest tests/test_individual.py -v --no-header -p no:deepeval

======================== 35 passed in 96.73s (0:01:36) ========================
```

All 35 tests passed (0 skipped, 0 failed).

### Manual Task 4 run result:

```
Documents loaded: 5
Chunks created: 55
Embedding model: BAAI/bge-m3
Embedding dim: 1024
Vector store type: faiss
```

### Manual Task 5-10 run results:

- Task 5 (semantic search): Returns relevant chunks sorted by cosine similarity
- Task 6 (lexical search): Returns BM25-scored chunks (query "Chi Dan ma tuy" returns 10 relevant results)
- Task 7 (reranking): RRF merges semantic + lexical results correctly
- Task 8 (PageIndex): Graceful warning if PAGEINDEX_API_KEY missing, no crash
- Task 9 (retrieval pipeline): Hybrid results from semantic + lexical via RRF
- Task 10 (generation): Extractive fallback answer with citations if no OpenAI key

## Pipeline Architecture

```
Query
  ├─ Semantic Search (dense, FAISS + bge-m3)
  ├─ Lexical Search (sparse, BM25)
  │
  ├─ RRF Merge
  ├─ Rerank (optional)
  │
  └─ PageIndex Fallback (if results empty)
       └─ Generation with Citation
```

## How to Run

```bash
cd LEVUANH_2A202600809

# Run indexing
python src/task4_chunking_indexing.py

# Search
python src/task5_semantic_search.py "Chi Dan bi cao buoc hanh vi gi?"
python src/task6_lexical_search.py "Chi Dan ma tuy"

# Retrieval pipeline
python src/task9_retrieval_pipeline.py

# Generation
python src/task10_generation.py "Nhung nghe si nao lien quan toi ma tuy?"

# Run all tests
pytest tests/test_individual.py -v --no-header -p no:deepeval
```

## Project Structure

```
LEVUANH_2A202600809/
├── data/
│   ├── landing/
│   │   ├── legal/          # Raw legal documents (PDF/DOCX)
│   │   └── news/           # Crawled news (JSON/HTML)
│   ├── standardized/
│   │   ├── news/          # Converted markdown files
│   │   └── legal/          # Converted markdown files
│   └── index/
│       ├── chunks.json      # Chunked documents
│       ├── vector_store.faiss  # FAISS vector index
│       └── index_metadata.json # Index metadata
├── src/
│   ├── task1_collect_legal_docs.py
│   ├── task2_crawl_news.py
│   ├── task3_convert_markdown.py
│   ├── task4_chunking_indexing.py
│   ├── task5_semantic_search.py
│   ├── task6_lexical_search.py
│   ├── task7_reranking.py
│   ├── task8_pageindex_vectorless.py
│   ├── task9_retrieval_pipeline.py
│   └── task10_generation.py
└── tests/
    └── test_individual.py
```
