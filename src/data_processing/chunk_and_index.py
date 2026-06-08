import re
import sys
from pathlib import Path
from typing import Any

from data_processing.config import (
    STANDARDIZED_LEGAL_DIR,
    STANDARDIZED_NEWS_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
)

try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    HAS_LANGCHAIN_SPLITTER = True
except ImportError:
    HAS_LANGCHAIN_SPLITTER = False

try:
    import chromadb

    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def _simple_chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - CHUNK_OVERLAP
    return chunks


def parse_frontmatter(md_text: str) -> tuple[dict, str]:
    if not md_text.startswith("---"):
        return {}, md_text
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
    metadata: dict[str, Any] = {}
    for line in lines[1:end_idx]:
        line = line.strip()
        if not line or ":" not in line:
            continue
        idx = line.index(":")
        key = line[:idx].strip().strip('"').strip("'")
        value = line[idx + 1:].strip().strip('"').strip("'")
        if key:
            metadata[key] = value
    body = "\n".join(lines[end_idx + 1:]).strip()
    return metadata, body


def extract_article_id(text: str) -> str:
    match = re.search(r"(Điều\s+\d+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def load_standardized_documents() -> list[dict]:
    documents: list[dict] = []
    dirs = []
    if STANDARDIZED_NEWS_DIR.exists():
        dirs.append(("news", STANDARDIZED_NEWS_DIR))
    if STANDARDIZED_LEGAL_DIR.exists():
        dirs.append(("legal", STANDARDIZED_LEGAL_DIR))

    seen_dedup_keys = set()
    for doc_type, dir_path in dirs:
        for md_path in sorted(dir_path.glob("*.md")):
            try:
                content = md_path.read_text(encoding="utf-8")
                metadata, body = parse_frontmatter(content)
                if not body:
                    continue

                title = metadata.get("title") or md_path.stem
                url = metadata.get("url", "")
                date = metadata.get("date", "")
                source = metadata.get("source", md_path.name)

                dedup_key = (url or title).strip().lower()
                if dedup_key in seen_dedup_keys:
                    print(f"  [SKIP] Duplicate document: {md_path.name}")
                    continue
                seen_dedup_keys.add(dedup_key)

                documents.append({
                    "content": body,
                    "source": md_path.name,
                    "type": doc_type,
                    "url": url,
                    "title": title,
                    "date": date,
                    "article_id": extract_article_id(body) if doc_type == "legal" else "",
                })
            except Exception as e:
                print(f"  [WARN] Cannot load {md_path.name}: {type(e).__name__}: {e}")
    return documents


def chunk_text(text: str) -> list[str]:
    if HAS_LANGCHAIN_SPLITTER:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        return splitter.split_text(text)
    return _simple_chunk_text(text)


def get_openai_client():
    if not HAS_OPENAI:
        raise RuntimeError("openai package is not installed. Please install it first.")
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Please create a .env file and set OPENAI_API_KEY."
        )
    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    if not texts:
        return []
    client = get_openai_client()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    return all_embeddings


def build_chroma_index(reset: bool = True) -> dict:
    if not HAS_CHROMADB:
        raise RuntimeError("chromadb package is not installed.")

    print("Loading standardized documents...")
    documents = load_standardized_documents()
    if not documents:
        raise RuntimeError("No standardized documents found.")

    print(f"Loaded {len(documents)} documents. Chunking...")
    all_chunks: list[dict] = []
    for doc in documents:
        text_chunks = chunk_text(doc["content"])
        for idx, raw_chunk in enumerate(text_chunks):
            chunk = raw_chunk.strip()
            if not chunk or len(chunk) < 50:
                continue
            all_chunks.append({
                "content": chunk,
                "metadata": {
                    "source": doc["source"],
                    "type": doc["type"],
                    "chunk_index": idx,
                    "url": doc["url"],
                    "title": doc["title"],
                    "date": doc["date"],
                    "article_id": doc["article_id"],
                },
            })

    if not all_chunks:
        raise RuntimeError("No valid chunks generated.")

    print(f"Embedding {len(all_chunks)} chunks with model {EMBEDDING_MODEL}...")
    texts = [c["content"] for c in all_chunks]
    embeddings = embed_texts(texts)
    print(f"Embedded {len(embeddings)} chunks.")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = []
    metadatas = []
    for idx, chunk in enumerate(all_chunks):
        ids.append(f"{chunk['metadata']['type']}_{chunk['metadata']['source']}_{idx}")
        metadatas.append(chunk["metadata"])

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"Indexed {len(all_chunks)} chunks into collection '{COLLECTION_NAME}'.")

    return {
        "documents_loaded": len(documents),
        "chunks_indexed": len(all_chunks),
        "collection_name": COLLECTION_NAME,
        "chroma_path": str(CHROMA_DIR),
        "embedding_model": EMBEDDING_MODEL,
        "metadata_fields": [
            "source", "type", "chunk_index", "url", "title", "date", "article_id"
        ],
    }
