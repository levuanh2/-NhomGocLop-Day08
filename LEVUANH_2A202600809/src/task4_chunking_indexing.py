"""
Task 4 — Chunk documents, create embeddings, and build vector index.

Input:
  - data/standardized/news/*.md
  - data/standardized/legal/*.md

Output:
  - data/index/chunks.json
  - data/index/vector_store.faiss (or numpy fallback)
  - data/index/index_metadata.json
"""

import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Fix Windows console encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Paths - tự động xác định LEVUANH subdirectory
# __file__ = E:\...\LEVUANH\src\task4.py
# __file__.parent = LEVUANH/src
# __file__.parent.parent = LEVUANH (root of project)
_BASE = Path(__file__).resolve().parent.parent
# Nếu _BASE chưa có src mà có LEVUANH con → nhảy vào
if not (_BASE / "src").exists() and (_BASE / "LEVUANH").exists():
    _BASE = _BASE / "LEVUANH"

STANDARDIZED_DIR = _BASE / "data" / "standardized"
NEWS_DIR = STANDARDIZED_DIR / "news"
LEGAL_DIR = STANDARDIZED_DIR / "legal"
INDEX_DIR = _BASE / "data" / "index"

# =============================================================================
# CONFIGURATION — Các hằng số cấu hình chunking và embedding
# =============================================================================

"""
Chunking strategy:
- RecursiveCharacterTextSplitter
- chunk_size = 500
- chunk_overlap = 50
Reason:
Recursive splitter is robust for mixed Vietnamese legal/news Markdown because
it preserves paragraph boundaries while keeping chunks small enough for retrieval.

Embedding:
- BAAI/bge-m3
- dimension = 1024
Reason:
bge-m3 supports multilingual retrieval and works well for Vietnamese.
"""

CHUNKING_STRATEGY = "RecursiveCharacterTextSplitter"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# Embedding models (theo thứ tự ưu tiên)
EMBEDDING_MODELS = [
    "BAAI/bge-m3",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
]

# Global state
embedding_model = None
embedding_model_name = None
embedding_dim = 0


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_directories() -> None:
    """Tạo thư mục index nếu chưa tồn tại."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# YAML Frontmatter Parser
# ---------------------------------------------------------------------------

def parse_frontmatter(md_text: str) -> tuple[dict, str]:
    """
    Tách YAML frontmatter khỏi nội dung markdown.

    Returns:
        (metadata_dict, body_text)
    """
    if not md_text.startswith("---"):
        return {}, md_text

    # Tìm closing ---
    lines = md_text.split("\n")
    if len(lines) < 3:
        return {}, md_text

    end_idx = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break

    if end_idx < 0:
        return {}, md_text

    # Parse YAML đơn giản (key: "value" hoặc key: value)
    yaml_lines = lines[1:end_idx]
    metadata = {}
    for line in yaml_lines:
        line = line.strip()
        if not line or ":" not in line:
            continue
        # Split at first colon
        idx = line.index(":")
        key = line[:idx].strip().strip('"').strip("'")
        value = line[idx + 1:].strip().strip('"').strip("'")
        if key:
            metadata[key] = value

    body = "\n".join(lines[end_idx + 1:]).strip()
    return metadata, body


# ---------------------------------------------------------------------------
# Document Loading
# ---------------------------------------------------------------------------

def load_markdown_documents() -> list[dict]:
    """
    Đọc tất cả .md từ standardized/news và standardized/legal.
    Parse frontmatter, tách body, giữ metadata.
    """
    documents = []
    dirs_to_scan = []

    if NEWS_DIR.exists():
        dirs_to_scan.append(("news", NEWS_DIR))
    if LEGAL_DIR.exists():
        dirs_to_scan.append(("legal", LEGAL_DIR))

    # Deduplicate globally across all directories
    seen_dedup_keys = set()

    for doc_type, dir_path in dirs_to_scan:
        md_files = sorted(dir_path.glob("*.md"))
        for md_path in md_files:
            try:
                content = md_path.read_text(encoding="utf-8")
                metadata, body = parse_frontmatter(content)

                # Nếu không có title trong frontmatter, dùng filename stem
                title = metadata.get("title") or md_path.stem
                url = metadata.get("url", "")

                # Deduplicate theo url hoặc normalized title
                dedup_key = _normalize_dedup_key(url, title)
                if dedup_key:
                    if dedup_key in seen_dedup_keys:
                        print(f"  [SKIP] Duplicate document: {md_path.name} (url/title already indexed)")
                        continue
                    seen_dedup_keys.add(dedup_key)

                doc = {
                    "doc_id": str(uuid.uuid4()),
                    "source_file": md_path.name,
                    "source_path": str(md_path),
                    "title": title,
                    "url": url,
                    "source": metadata.get("source", md_path.stem),
                    "doc_type": doc_type,
                    "content": body,
                    "content_length": len(body),
                }
                documents.append(doc)
                print(f"  [LOADED] {md_path.name} ({len(body)} chars)")
            except Exception as e:
                print(f"  [WARN] Cannot load {md_path.name}: {type(e).__name__}: {e}")

    return documents


def _normalize_dedup_key(url: str, title: str) -> str:
    """Tạo dedup key từ URL hoặc title."""
    if url and url.strip():
        return url.strip().lower()
    if title and title.strip():
        # Normalize title: lowercase, strip extra spaces
        return title.strip().lower()
    return ""


def load_documents() -> list[dict]:
    """
    Wrapper để tương thích với tests.

    Returns:
        List of documents với format:
        {
            "doc_id": str,
            "source_file": str,
            "source_path": str,
            "title": str,
            "url": str,
            "source": str,
            "doc_type": str,
            "content": str,
            "content_length": int
        }
    """
    return load_markdown_documents()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk từng document dùng RecursiveCharacterTextSplitter.
    Mỗi chunk giữ metadata đầy đủ.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=SEPARATORS,
            length_function=len,
        )
    except ImportError:
        print("[ERROR] langchain_text_splitters not installed. Using simple split.")
        return _simple_chunk(documents)

    chunks = []
    max_chunk_size = CHUNK_SIZE + 100  # Guard: không vượt quá CHUNK_SIZE + 100

    for doc in documents:
        doc_id = doc["doc_id"]
        source_file = doc["source_file"]
        title = doc["title"]
        url = doc["url"]
        source = doc["source"]
        doc_type = doc["doc_type"]
        content = doc["content"]

        # Split content
        try:
            text_chunks = text_splitter.split_text(content)
        except Exception as e:
            print(f"  [WARN] Split failed for {source_file}: {e}")
            continue

        for chunk_idx, chunk_text in enumerate(text_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            # Noise filter: skip invalid chunks
            chunk_len = len(chunk_text)
            if chunk_len < 80:
                continue
            if "javascript:" in chunk_text.lower():
                continue
            if "Chia sẻ bài viết" in chunk_text:
                continue
            if "Bình luận" in chunk_text:
                continue
            if "Copy link" in chunk_text:
                continue
            # Skip chunks that are only title headings
            if chunk_text.startswith("#") and chunk_len < 150:
                continue
            # Skip chunks that are only images/links
            if re.match(r"^!\[.*\]\(.*\)$", chunk_text.strip()):
                continue
            if re.match(r"^\[.*\]\(.*\)$", chunk_text.strip()) and chunk_len < 100:
                continue

            # Guard: cắt nếu vượt quá max_chunk_size
            if len(chunk_text) > max_chunk_size:
                chunk_text = chunk_text[:max_chunk_size]

            chunk = {
                "chunk_id": f"{doc_id}_{chunk_idx}",
                "doc_id": doc_id,
                "source_file": source_file,
                "title": title,
                "url": url,
                "source": source,
                "doc_type": doc_type,
                "chunk_index": chunk_idx,
                "text": chunk_text,
                "text_length": len(chunk_text),
                "content": chunk_text,  # Alias cho tương thích tests
            }
            chunks.append(chunk)

    return chunks


def _simple_chunk(documents: list[dict]) -> list[dict]:
    """Simple fallback chunking khi không có langchain."""
    chunks = []
    max_chunk_size = CHUNK_SIZE + 100

    for doc in documents:
        content = doc["content"]
        doc_id = doc["doc_id"]
        # Split by double newlines
        parts = content.split("\n\n")
        for idx, part in enumerate(parts):
            part = part.strip()
            if not part or len(part) < 50:
                continue
            # Guard chunk size
            if len(part) > max_chunk_size:
                part = part[:max_chunk_size]

            chunk = {
                "chunk_id": f"{doc_id}_{idx}",
                "doc_id": doc_id,
                "source_file": doc["source_file"],
                "title": doc["title"],
                "url": doc["url"],
                "source": doc["source"],
                "doc_type": doc["doc_type"],
                "chunk_index": idx,
                "text": part,
                "text_length": len(part),
                "content": part,
            }
            chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def load_embedding_model() -> Any:
    """
    Thử load embedding model theo thứ tự ưu tiên.
    Fallback nếu model không hoạt động.
    """
    global embedding_model, embedding_model_name, embedding_dim

    # Thử sentence-transformers trước
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[ERROR] sentence-transformers not installed.")
        return None

    for model_name in EMBEDDING_MODELS:
        try:
            print(f"  Trying model: {model_name}...")
            model = SentenceTransformer(model_name)
            # Test encode
            test_embedding = model.encode(["test"])
            embedding_dim = test_embedding.shape[1]
            embedding_model = model
            embedding_model_name = model_name
            print(f"  [OK] Loaded: {model_name} (dim={embedding_dim})")
            return model
        except Exception as e:
            print(f"  [FAIL] {model_name}: {type(e).__name__}: {e}")
            continue

    print("[ERROR] No embedding model could be loaded.")
    return None


def embed_chunks(chunks: list[dict]) -> tuple[list[dict], Any]:
    """
    Tạo embedding cho từng chunk text.

    Returns:
        (chunks_with_embeddings, embeddings_matrix)
    """
    if not embedding_model:
        print("[WARN] No embedding model, skipping embeddings.")
        return chunks, None

    texts = [chunk["text"] for chunk in chunks]

    print(f"  Encoding {len(texts)} chunks...")
    try:
        embeddings = embedding_model.encode(texts, show_progress_bar=True)
        print(f"  [OK] Embeddings shape: {embeddings.shape}")
        return chunks, embeddings
    except Exception as e:
        print(f"  [WARN] Embedding failed: {e}")
        return chunks, None


# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------

HAS_FAISS = False
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    pass

import numpy as np


def build_vector_index(embeddings: Any) -> tuple[bool, str]:
    """
    Xây dựng vector index từ embeddings.

    Returns:
        (success, store_type)
    """
    if embeddings is None:
        return False, "none"

    embeddings = np.array(embeddings).astype("float32")
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    # Thử FAISS trước
    if HAS_FAISS:
        try:
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalize)
            index.add(embeddings)
            faiss.write_index(index, str(INDEX_DIR / "vector_store.faiss"))
            print(f"  [OK] FAISS index saved: vector_store.faiss ({index.ntotal} vectors)")
            return True, "faiss"
        except Exception as e:
            print(f"  [WARN] FAISS failed: {e}")

    # Fallback: Lưu numpy array
    try:
        np.save(INDEX_DIR / "embeddings.npy", embeddings)
        print(f"  [OK] Numpy embeddings saved: embeddings.npy ({embeddings.shape})")
        return True, "numpy"
    except Exception as e:
        print(f"  [ERROR] Numpy save failed: {e}")
        return False, "none"


# ---------------------------------------------------------------------------
# Save Outputs
# ---------------------------------------------------------------------------

def save_outputs(
    chunks: list[dict],
    embeddings: Any,
    documents: list[dict],
    store_type: str,
) -> None:
    """
    Lưu chunks.json, vector store, và index_metadata.json.
    """
    # 1. Save chunks.json
    chunks_path = INDEX_DIR / "chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"  [OK] Chunks saved: {chunks_path} ({len(chunks)} chunks)")

    # 2. Save index_metadata.json
    actual_model_name = embedding_model_name or EMBEDDING_MODEL_NAME
    actual_dim = embedding_dim or EMBEDDING_DIM

    metadata = {
        "chunking_strategy": CHUNKING_STRATEGY,
        "embedding_model": actual_model_name,
        "embedding_dim": actual_dim,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "vector_store": store_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "separators": SEPARATORS,
        "input_dirs": [
            str(NEWS_DIR) if NEWS_DIR.exists() else None,
            str(LEGAL_DIR) if LEGAL_DIR.exists() else None,
        ],
        "documents": [
            {
                "doc_id": d["doc_id"],
                "title": d["title"],
                "source_file": d["source_file"],
                "doc_type": d["doc_type"],
                "content_length": d["content_length"],
            }
            for d in documents
        ],
    }
    metadata_path = INDEX_DIR / "index_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  [OK] Metadata saved: {metadata_path}")

    # 3. Vector store đã được lưu trong build_vector_index
    if store_type == "faiss":
        print(f"  [OK] Vector index: {INDEX_DIR / 'vector_store.faiss'}")
    elif store_type == "numpy":
        print(f"  [OK] Vector index: {INDEX_DIR / 'embeddings.npy'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Main function: chunk, embed, and index documents."""
    setup_directories()

    print("=" * 70)
    print("TASK 4: CHUNKING & INDEXING")
    print(f"Standardized:  {STANDARDIZED_DIR}")
    print(f"Index output:  {INDEX_DIR}")
    print(f"Chunk size:    {CHUNK_SIZE}")
    print(f"Chunk overlap: {CHUNK_OVERLAP}")
    print(f"Strategy:      {CHUNKING_STRATEGY}")
    print(f"Embedding:     {EMBEDDING_MODEL_NAME} (dim={EMBEDDING_DIM})")
    print(f"FAISS:         {'available' if HAS_FAISS else 'not available'}")
    print("=" * 70)

    # Step 1: Load documents
    print("\n--- Loading documents ---")
    documents = load_documents()
    if not documents:
        print("[ERROR] No documents loaded. Exiting.")
        return
    print(f"  Total documents loaded: {len(documents)}")

    # Step 2: Chunk documents
    print("\n--- Chunking documents ---")
    chunks = chunk_documents(documents)
    if not chunks:
        print("[ERROR] No chunks created. Exiting.")
        return
    print(f"  Total chunks created: {len(chunks)}")

    # Step 3: Load embedding model
    print("\n--- Loading embedding model ---")
    load_embedding_model()
    if not embedding_model:
        print("[WARN] No embedding model loaded. Skipping vector index.")

    # Step 4: Embed chunks
    print("\n--- Embedding chunks ---")
    chunks, embeddings = embed_chunks(chunks)

    # Step 5: Build vector index
    print("\n--- Building vector index ---")
    success, store_type = build_vector_index(embeddings)
    if not success:
        print("[WARN] Vector index build failed.")

    # Step 6: Save outputs
    print("\n--- Saving outputs ---")
    save_outputs(chunks, embeddings, documents, store_type)

    # Summary
    print("\n" + "=" * 70)
    print("INDEXING SUMMARY")
    print(f"  Documents loaded:  {len(documents)}")
    print(f"  Chunks created:    {len(chunks)}")
    print(f"  Embedding model:   {embedding_model_name or 'none'}")
    print(f"  Embedding dim:     {embedding_dim}")
    print(f"  Vector store type: {store_type}")
    print(f"  Output directory:  {INDEX_DIR}")
    print(f"  chunks.json:       {INDEX_DIR / 'chunks.json'}")
    if store_type == "faiss":
        print(f"  vector_store.faiss: {INDEX_DIR / 'vector_store.faiss'}")
    elif store_type == "numpy":
        print(f"  embeddings.npy:    {INDEX_DIR / 'embeddings.npy'}")
    print(f"  index_metadata.json: {INDEX_DIR / 'index_metadata.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
