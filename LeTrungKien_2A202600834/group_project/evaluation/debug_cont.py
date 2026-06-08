"""Debug continuation logic for cases 01 and 08."""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv; load_dotenv(ROOT / ".env")

from src.task9_retrieval_pipeline import retrieve, _add_continuations

queries = [
    ("case01_behaviors", "Theo Luật số 120/2025/QH15, những hành vi nào bị nghiêm cấm liên quan đến chất ma túy?"),
    ("case08_finance",   "Luật Phòng, chống ma túy quy định những nguồn tài chính nào được dùng cho công tác phòng, chống ma túy?"),
    ("case04_arrest",    "Ca sĩ Miu Lê bị bắt vào ngày nào, ở địa điểm nào, và cùng với bao nhiêu người?"),
]

for name, query in queries:
    print(f"\n{'='*70}")
    print(f"[{name}]")
    retrieved = retrieve(query, top_k=5)
    print(f"Retrieved ({len(retrieved)} chunks):")
    for r in retrieved:
        meta = r.get("metadata", {})
        print(f"  chunk_index={meta.get('chunk_index','?')} src={meta.get('source','?')[:40]}")

    extended = _add_continuations(retrieved)
    print(f"\nAfter _add_continuations ({len(extended)} chunks):")
    for r in extended:
        meta = r.get("metadata", {})
        ci = meta.get("chunk_index", "?")
        src = meta.get("source", "?")[:30]
        print(f"  chunk_index={ci} src={src}")
        print(f"  >>> {r['content'][:180]}")
