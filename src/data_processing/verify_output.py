import sys
from pathlib import Path
from typing import Any

from data_processing.config import (
    STANDARDIZED_LEGAL_DIR,
    STANDARDIZED_NEWS_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
)
from data_processing.chunk_and_index import embed_texts

try:
    import chromadb

    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


def _check_exists(path: Path, label: str) -> tuple[bool, str]:
    exists = path.exists() and any(path.iterdir())
    status = "PASS" if exists else "FAIL"
    message = f"{status}: {label} exists and has files" if exists else f"{status}: {label} missing or empty"
    return exists, message


def verify_tv1_output() -> dict:
    checks: dict[str, Any] = {}
    passed = True
    sample_metadata: list[dict] = []

    # 1. standardized/legal exists
    ok, msg = _check_exists(STANDARDIZED_LEGAL_DIR, "data/standardized/legal")
    checks["standardized_legal_exists"] = ok
    passed = passed and ok
    print(msg)

    # 2. standardized/news exists
    ok, msg = _check_exists(STANDARDIZED_NEWS_DIR, "data/standardized/news")
    checks["standardized_news_exists"] = ok
    passed = passed and ok
    print(msg)

    # 3. at least 1 legal md
    legal_md_count = len(list(STANDARDIZED_LEGAL_DIR.glob("*.md"))) if STANDARDIZED_LEGAL_DIR.exists() else 0
    ok = legal_md_count >= 1
    checks["legal_md_count"] = legal_md_count
    passed = passed and ok
    print(f"{'PASS' if ok else 'FAIL'}: legal markdown files = {legal_md_count}")

    # 4. at least 1 news md
    news_md_count = len(list(STANDARDIZED_NEWS_DIR.glob("*.md"))) if STANDARDIZED_NEWS_DIR.exists() else 0
    ok = news_md_count >= 1
    checks["news_md_count"] = news_md_count
    passed = passed and ok
    print(f"{'PASS' if ok else 'FAIL'}: news markdown files = {news_md_count}")

    if not HAS_CHROMADB:
        checks["chroma_exists"] = False
        checks["collection_exists"] = False
        checks["collection_count"] = 0
        checks["query_ok"] = False
        checks["metadata_schema_ok"] = False
        checks["type_values_ok"] = False
        passed = False
        print("FAIL: chromadb not installed")
        return {
            "passed": passed,
            "checks": checks,
            "collection_count": 0,
            "sample_metadata": [],
        }

    # 5. vectorstore/chroma exists
    ok = CHROMA_DIR.exists()
    checks["chroma_exists"] = ok
    passed = passed and ok
    print(f"{'PASS' if ok else 'FAIL'}: chroma directory exists")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # 6. collection exists
    try:
        collection = client.get_collection(COLLECTION_NAME)
        ok = True
    except Exception:
        ok = False
        collection = None
    checks["collection_exists"] = ok
    passed = passed and ok
    print(f"{'PASS' if ok else 'FAIL'}: collection '{COLLECTION_NAME}' exists")

    # 7. count > 0
    count = collection.count() if collection else 0
    ok = count > 0
    checks["collection_count"] = count
    passed = passed and ok
    print(f"{'PASS' if ok else 'FAIL'}: collection count = {count}")

    # 8. query
    query_ok = False
    if collection:
        try:
            query_embedding = embed_texts(["phạt tù"])[0]
            results = collection.query(query_embeddings=[query_embedding], n_results=3)
            metadatas = results.get("metadatas", [[]])[0]
            query_ok = len(metadatas) > 0
            if metadatas:
                sample_metadata = metadatas[:3]
        except Exception as e:
            print(f"[WARN] query failed: {e}")
    checks["query_ok"] = query_ok
    passed = passed and query_ok
    print(f"{'PASS' if query_ok else 'FAIL'}: query returned results")

    # 9. metadata fields
    required_fields = {"source", "type", "chunk_index", "url", "title", "date", "article_id"}
    schema_ok = True
    type_ok = True
    if sample_metadata:
        for meta in sample_metadata:
            if not required_fields.issubset(meta.keys()):
                schema_ok = False
            if meta.get("type") not in ("legal", "news"):
                type_ok = False
    else:
        schema_ok = False
        type_ok = False
    checks["metadata_schema_ok"] = schema_ok
    checks["type_values_ok"] = type_ok
    passed = passed and schema_ok and type_ok
    print(f"{'PASS' if schema_ok else 'FAIL'}: metadata has all required fields")
    print(f"{'PASS' if type_ok else 'FAIL'}: metadata.type is only 'legal' or 'news'")

    print("\nSample metadata:")
    for meta in sample_metadata:
        print(meta)

    return {
        "passed": passed,
        "checks": checks,
        "collection_count": count,
        "sample_metadata": sample_metadata,
    }
