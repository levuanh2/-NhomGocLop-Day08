"""Check what's actually in the key chunks for cases 04, 05, 08."""
import sys, re
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv; load_dotenv(ROOT / ".env")

# Load corpus from ChromaDB
import chromadb
client = chromadb.PersistentClient(path=str(ROOT / "data" / "vectorstore" / "chroma"))
col = client.get_collection("drug_law_docs")
data = col.get(include=["documents", "metadatas"])

docs = {i: (d, m) for i, (d, m) in enumerate(zip(data["documents"], data["metadatas"]))}

def search_chunks(query: str, source_filter: str = None, top=3):
    """Find chunks containing query string."""
    results = []
    for idx, (doc, meta) in docs.items():
        src = meta.get("source", "")
        if source_filter and source_filter not in src:
            continue
        if query.lower() in doc.lower():
            results.append((idx, doc, meta))
    return results[:top]

print("=== article_01: looking for names ===")
hits = search_chunks("Vũ Khương An", "article_01")
for idx, doc, meta in hits:
    ci = meta.get("chunk_index", "?")
    print(f"  chunk_index={ci}:")
    print(f"  {doc[:400]}")
    print()

print("=== article_03: looking for ketamine quantities ===")
hits = search_chunks("5,35g", "article_03")
for idx, doc, meta in hits:
    ci = meta.get("chunk_index", "?")
    print(f"  chunk_index={ci}:")
    print(f"  {doc[:400]}")
    print()

print("=== law: looking for Dieu 4 sources of finance ===")
hits = search_chunks("tài trợ", "Luật phồng")
for idx, doc, meta in hits:
    ci = meta.get("chunk_index", "?")
    print(f"  chunk_index={ci}:")
    print(f"  {doc[:400]}")
    print()

print("=== law: looking for Dieu 5 behaviors 2-11 ===")
hits = search_chunks("Chiếm đoạt chất ma túy", "Luật phồng")
for idx, doc, meta in hits:
    ci = meta.get("chunk_index", "?")
    print(f"  chunk_index={ci}:")
    print(f"  {doc[:500]}")
    print()
