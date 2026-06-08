"""Debug worst cases to understand retrieval failures."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.task9_retrieval_pipeline import retrieve

cases = [
    ("case09_definition", "Cai nghiện ma túy được định nghĩa như thế nào trong Luật Phòng, chống ma túy?"),
    ("case01_11behaviors", "Theo Luật số 120/2025/QH15, những hành vi nào bị nghiêm cấm liên quan đến chất ma túy?"),
    ("case04_arrest_details", "Ca sĩ Miu Lê bị bắt vào ngày nào, ở địa điểm nào, và cùng với bao nhiêu người?"),
    ("case10_multi_artists", "Những nghệ sĩ Việt Nam nào đã bị bắt liên quan đến ma túy trong năm 2026?"),
]

for name, query in cases:
    print(f"\n{'='*70}")
    print(f"[{name}]")
    print(f"Query: {query}")
    print("="*70)
    try:
        results = retrieve(query, top_k=5)
        for i, r in enumerate(results, 1):
            src = r.get("metadata", {}).get("source", "?")
            print(f"  {i}. [{r['score']:.4f}] [{src}]")
            print(f"     {r['content'][:250]}")
    except Exception as e:
        print(f"  ERROR: {e}")
