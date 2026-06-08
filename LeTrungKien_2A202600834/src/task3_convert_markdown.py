"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Với file .doc (Word cũ): fallback sang win32com (yêu cầu Microsoft Word đã cài).

Cài đặt:
    pip install markitdown
"""

import json
from pathlib import Path

from markitdown import MarkItDown
from markitdown._exceptions import UnsupportedFormatException

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _convert_doc_with_word(filepath: Path) -> str:
    """Dùng Microsoft Word qua win32com để đọc .doc/.docx."""
    import pythoncom
    import win32com.client

    pythoncom.CoInitialize()
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    try:
        doc = word.Documents.Open(str(filepath.resolve()))
        text = doc.Content.Text
        doc.Close(False)
        return text
    finally:
        word.Quit()
        pythoncom.CoUninitialize()


def _get_clean_fns():
    """Lazy import cleaning functions (cùng package hoặc script trực tiếp)."""
    try:
        from .clean_data import clean_legal, clean_news
    except ImportError:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent))
        from clean_data import clean_legal, clean_news  # noqa: PLC0415
    return clean_legal, clean_news


def convert_legal_docs():
    """Convert PDF/DOCX/DOC files trong data/landing/legal/ sang markdown.
    Áp dụng clean_legal() trước khi ghi → standardized/ chứa sẵn data sạch.
    """
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_legal, _ = _get_clean_fns()
    md = MarkItDown()
    converted = 0

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in (".pdf", ".docx", ".doc"):
            continue

        print(f"Converting: {filepath.name}")
        try:
            result = md.convert(str(filepath))
            text = result.text_content
        except UnsupportedFormatException:
            # .doc cũ không được MarkItDown hỗ trợ — dùng Word trực tiếp
            print(f"  → MarkItDown không hỗ trợ, dùng win32com fallback...")
            text = _convert_doc_with_word(filepath)

        text = clean_legal(text)  # normalize unicode/whitespace/control chars
        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(text, encoding="utf-8")
        print(f"  ✓ Saved: {output_path.name}  ({len(text):,} chars)")
        converted += 1

    return converted


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown.
    Áp dụng clean_news() trước khi ghi → standardized/ chứa sẵn data sạch.
    """
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    _, clean_news = _get_clean_fns()
    converted = 0
    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue

        print(f"Converting: {filepath.name}")
        data = json.loads(filepath.read_text(encoding="utf-8"))
        output_path = output_dir / f"{filepath.stem}.md"

        header = f"# {data.get('title', 'Unknown')}\n\n"
        header += f"**Source:** {data.get('url', 'N/A')}\n"
        header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"

        content = clean_news(header + data.get("content_markdown", ""))
        output_path.write_text(content, encoding="utf-8")
        print(f"  ✓ Saved: {output_path.name}  ({len(content):,} chars)")
        converted += 1

    return converted


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    n_legal = convert_legal_docs()

    print("\n--- News Articles ---")
    n_news = convert_news_articles()

    print(f"\n✓ Done! {n_legal} legal docs + {n_news} news articles → {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
