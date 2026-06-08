"""
Task 4 — Chunking & Indexing vào Vector Store.

Chunking strategy: SemanticChunker (langchain_experimental)
  Dùng OpenAI embeddings để phát hiện ranh giới ngữ nghĩa tự nhiên — các câu có
  cosine distance lớn so với câu kế tiếp được dùng làm điểm cắt (percentile 85).
  Nếu semantic chunk vượt CHUNK_SIZE, áp dụng RecursiveCharacterTextSplitter để
  chia tiếp → đảm bảo mọi chunk đều ≤ CHUNK_SIZE.

Embedding model: text-embedding-3-small (OpenAI)
  - 1536 chiều, chất lượng cao với tiếng Việt
  - Cùng model với SemanticChunker → vector space nhất quán
  - Nhanh, giá rẻ (~$0.02/1M tokens)

Vector store: ChromaDB (local, data/vectorstore/chroma/)
  - Không cần Docker hay cloud credentials
  - Hỗ trợ cosine similarity và persistent storage
  - Đủ cho tập dữ liệu nhỏ (~15 files)

CHUNK_SIZE = 800    — ~100 token tiếng Việt, cân bằng precision vs context
CHUNK_OVERLAP = 100 — 12.5% overlap giữ liên kết ngữ cảnh tại ranh giới chunk
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
VECTORSTORE_DIR = Path(__file__).parent.parent / "data" / "vectorstore" / "chroma"


# =============================================================================
# CONFIGURATION
# =============================================================================

# SemanticChunker tách tại ranh giới ngữ nghĩa — CHUNK_SIZE / CHUNK_OVERLAP
# dùng cho fallback splitter khi semantic chunk quá dài.
CHUNK_SIZE = 800        # ~100 token tiếng Việt
CHUNK_OVERLAP = 100     # 12.5% — đủ để giữ ngữ cảnh, không gây trùng lặp nhiều

CHUNKING_METHOD = "semantic"   # SemanticChunker + fallback RecursiveCharacterTextSplitter

# Cùng model cho chunking và final embedding → vector space nhất quán
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

VECTOR_STORE = "chromadb"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ .md files từ data/standardized/ (đã được clean inline trong task3).

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str, 'url': str}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"

        url = ""
        if doc_type == "news":
            for line in content.split("\n")[:6]:
                if line.startswith("**Source:**"):
                    url = line.replace("**Source:**", "").strip()
                    break

        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "type": doc_type,
                "url": url,
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    2-pass semantic chunking:
      1. SemanticChunker xác định ranh giới ngữ nghĩa (percentile 85)
      2. Bất kỳ chunk nào > CHUNK_SIZE được chia tiếp bằng RecursiveCharacterTextSplitter

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    semantic_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85,
    )
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        semantic_chunks = semantic_splitter.split_text(doc["content"])
        chunk_index = 0
        for sc in semantic_chunks:
            if len(sc) <= CHUNK_SIZE:
                chunks.append({
                    "content": sc,
                    "metadata": {**doc["metadata"], "chunk_index": chunk_index},
                })
                chunk_index += 1
            else:
                # Fallback: chia tiếp semantic chunk quá dài
                for sub in fallback_splitter.split_text(sc):
                    chunks.append({
                        "content": sub,
                        "metadata": {**doc["metadata"], "chunk_index": chunk_index},
                    })
                    chunk_index += 1
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng text-embedding-3-small, batch 100 requests mỗi lần.

    Returns:
        Chunks với key 'embedding': list[float] (1536 chiều)
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    texts = [c["content"] for c in chunks]

    batch_size = 100
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend(item.embedding for item in response.data)
        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks")

    for chunk, emb in zip(chunks, all_embeddings):
        chunk["embedding"] = emb
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào ChromaDB local (data/vectorstore/chroma/).
    Collection name: 'drug_law_docs'. Re-index sẽ xoá collection cũ.
    """
    import chromadb

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))

    try:
        client.delete_collection("drug_law_docs")
    except Exception:
        pass

    collection = client.create_collection(
        name="drug_law_docs",
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    documents_text = [c["content"] for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    # ChromaDB metadata values must be str/int/float/bool
    metadatas = [
        {k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
         for k, v in c["metadata"].items()}
        for c in chunks
    ]

    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        collection.add(
            ids=ids[i : i + batch_size],
            documents=documents_text[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )
        print(f"  Indexed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

    print(f"  Collection 'drug_law_docs' → {collection.count()} vectors @ {VECTORSTORE_DIR}")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 55)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking : {CHUNKING_METHOD}  (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL}  ({EMBEDDING_DIM} dim)")
    print(f"  Store    : {VECTOR_STORE}")
    print("=" * 55)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents from {STANDARDIZED_DIR}")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to ChromaDB")


if __name__ == "__main__":
    run_pipeline()
