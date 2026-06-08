"""Root-level entry point for Hugging Face Spaces and local runs.

Hugging Face Spaces expects app.py at repo root for Streamlit apps.
This file just delegates to the actual app inside src/interface/.

Run locally:
    streamlit run app.py

On HF Spaces: push repo, set SDK=streamlit in README frontmatter,
HF will auto-run `streamlit run app.py`.
"""

import sys
from pathlib import Path

# Ensure src/ is on the path so rag_pipeline imports resolve
ROOT = Path(__file__).parent
for p in (ROOT / "src", ROOT / "src" / "interface"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Delegate to the actual Streamlit app
from interface.app import main  # noqa: E402

main()
