"""
Task 3 — Convert landing data sang Markdown chuẩn hóa.

Input:
  - data/landing/legal/*   (PDF, DOCX, HTML, TXT, MD)
  - data/landing/news/*.json (Task 2 output)

Output:
  - data/standardized/legal/*.md
  - data/standardized/news/*.md

Mỗi output .md có YAML frontmatter và nội dung đã clean.
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Fix Windows console encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Try import markitdown
try:
    from markitdown import MarkItDown

    HAS_MARKITDOWN = True
except ImportError:
    HAS_MARKITDOWN = False

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
LANDING_DIR = BASE_DIR / "data" / "landing"
LEGAL_DIR = LANDING_DIR / "legal"
NEWS_DIR = LANDING_DIR / "news"
STANDARDIZED_DIR = BASE_DIR / "data" / "standardized"
LEGAL_OUT = STANDARDIZED_DIR / "legal"
NEWS_OUT = STANDARDIZED_DIR / "news"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_directories() -> None:
    """Tạo thư mục output nếu chưa tồn tại."""
    LEGAL_OUT.mkdir(parents=True, exist_ok=True)
    NEWS_OUT.mkdir(parents=True, exist_ok=True)


def slugify(text: str, max_length: int = 80) -> str:
    """
    Chuyển text thành slug an toàn cho filename (ASCII, dùng _ thay khoảng trắng).
    """
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")
    if len(text) > max_length:
        text = text[:max_length].rstrip("_")
    return text if text else "untitled"


def build_frontmatter(metadata: dict) -> str:
    """
    Tạo YAML frontmatter từ dict metadata.

    Ví dụ:
        ---
        title: "foo"
        source: "bar"
        ---
    """
    lines = ["---"]
    for key, value in metadata.items():
        if value is None:
            lines.append(f"{key}: null")
        elif isinstance(value, bool):
            lines.append(f"{key}: {str(value).lower()}")
        else:
            safe = str(value).replace('"', '\\"')
            lines.append(f'{key}: "{safe}"')
    lines.append("---")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# News conversion
# ---------------------------------------------------------------------------

# Các dòng menu/header/footer phổ biến cần loại bỏ
NOISE_PATTERNS = [
    # Navigation
    r"^\[?\s*Trang\s+chủ\s*\]?\(?.*?\)?\s*$",
    r"^\[?\s*Mới\s+nhất\s*\]?\(?.*?\)?\s*$",
    r"^Tin\s+theo\s+khu\s+vực\s*$",
    r"^International\s*$",
    r"^\[?\s*Đăng\s+nhập\s*\]?\(?.*?\)?\s*$",
    r"^\[?\s*Đăng\s+ký\s*\]?\(?.*?\)?\s*$",
    r"^Thông\s+báo\s*$",
    r"^Ý\s+kiến\s*$",
    r"^Tin\s+đã\s+xem\s*$",
    r"^Bật\s+nhận\s+thông\s+báo\s*$",
    r"^Đang\s+tải\s+dữ\s+liệu\.\.\.\s*$",
    r"^\[?\s*Tải\s+ứng\s+dụng\s*\]?\(?.*?\)?\s*$",
    # Social/Action buttons
    r"^\[?\s*Chia\s+sẻ\s+bài\s+viết\s*\]?\(?.*?\)?\s*$",
    r"^\[?\s*Chia\s+sẻ\s+.*?\s*\]?\(?.*?\)?\s*$",
    r"^\[?\s*Bình\s+luận\s*\]?\(?.*?\)?\s*$",
    r"^\[?\s*In\s*\]?\(?.*?\)?\s*$",
    r"^\[?\s*Copy\s+link.*?\s*\]?\(?.*?\)?\s*$",
    r"^\[?\s*Gửi\s+email.*?\s*\]?\(?.*?\)?\s*$",
    r"^\[?\s*Mail\s*\]?\(?.*?\)?\s*$",
    # Navigation back
    r"^\[?\s*Trở\s+lại\s+.*?\s*\]?\(?.*?\)?\s*$",
    r"^Về\s+trang\s+chủ\s*$",
    r"^Lên\s+đầu\s+trang\s*$",
    # Footer
    r"^Liên\s+hệ\s+quảng\s+cáo\s*$",
    r"^Hotline\s*:?\s*$",
    r"^Copyright.*?$",
    r"^Điều\s+khoản\s*sử\s*dụng.*?$",
    # Date header
    r"^\[?\s*Thứ\s+\w+,\s*\d+/\d+/\d+\s*\]?\(?.*?\)?\s*$",
    # Location header (standalone)
    r"^(TP\s+HCM|Hà\s+Nội|Đà\s+Nẵng|Hải\s+Phòng)\s*$",
    # Topic links
    r"^\*.*?\[.*?\]\(https?://vnexpress\.net/(topic|chu-de).*?\)\s*$",
    r"^\s*\*\s*\[.*?\]\(https?://vnexpress\.net/(topic|chu-de).*?\)\s*$",
]

COMPILED_NOISE = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]


def clean_news_markdown(content: str) -> str:
    """
    Loại bỏ menu/header/footer và các dòng noise khỏi content markdown.
    Giữ lại nội dung bài báo thật.
    """
    lines = content.split("\n")
    cleaned = []
    skip_next_empty = False

    for line in lines:
        stripped = line.strip()

        # Skip empty lines liên tiếp
        if not stripped:
            if skip_next_empty:
                continue
            skip_next_empty = True
            cleaned.append("")
            continue
        skip_next_empty = False

        # Xóa link javascript
        if re.match(r"^\[\]\(javascript:", stripped, re.IGNORECASE):
            continue
        if re.match(r"^\[\s*\]\(javascript:", stripped, re.IGNORECASE):
            continue

        # Xóa dòng logo/menu đơn thuần (chỉ chứa ảnh hoặc link ngắn)
        if re.match(r"^!\[\s*\]\(https?://.*(logo|menu|graphics).*\)$", stripped, re.IGNORECASE):
            continue
        if re.match(r"^\[\s*\]\(https?://.*(logo|menu).*\)$", stripped, re.IGNORECASE):
            continue

        # Xóa các dòng noise
        is_noise = False
        for pattern in COMPILED_NOISE:
            if pattern.match(stripped):
                is_noise = True
                break
        if is_noise:
            continue

        cleaned.append(line)

    # Giảm nhiều dòng trống liên tiếp
    result = "\n".join(cleaned)
    result = re.sub(r"\n{3,}", "\n\n", result)

    # Loại bỏ heading lặp lại (giữ H1 đầu tiên, bỏ các H1/H2 trùng)
    result = re.sub(r"(# .+)\n+# \1", r"\1", result, flags=re.IGNORECASE)
    result = re.sub(r"(# .+)\n+## \1", r"\1", result, flags=re.IGNORECASE)

    return result.strip()


def convert_news_json_file(json_path: Path) -> bool:
    """
    Convert 1 file JSON từ Task 2 sang Markdown.

    Returns True nếu convert thành công, False nếu lỗi/không hợp lệ.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            article = json.load(f)

        # Chỉ xử lý bài hợp lệ và đúng topic
        if article.get("status") != "success":
            return False
        if not article.get("is_valid_article", False):
            return False
        if not article.get("is_relevant_topic", False):
            return False

        title = article.get("title", "Untitled")
        source = article.get("source", "unknown")
        url = article.get("url", "")
        date_crawled = article.get("date_crawled", "")
        content = article.get("content_markdown", "")

        if not content:
            return False

        # Clean content
        cleaned_content = clean_news_markdown(content)

        # Loại bỏ các heading trùng lặp với title trong content
        title_escaped = re.escape(title)
        cleaned_content = re.sub(
            rf"^#+\s*{title_escaped}\s*$",
            "",
            cleaned_content,
            flags=re.MULTILINE | re.IGNORECASE
        )

        # Build frontmatter
        frontmatter = build_frontmatter({
            "title": title,
            "source": source,
            "url": url,
            "date_crawled": date_crawled,
            "doc_type": "news",
        })

        # Output: đúng 1 heading chính
        markdown = f"{frontmatter}\n\n# {title}\n\n{cleaned_content.strip()}\n"

        # Filename
        filename = f"{slugify(title)}.md"
        out_path = NEWS_OUT / filename

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        return True

    except Exception as e:
        print(f"  [WARN] Cannot convert {json_path.name}: {type(e).__name__}: {e}")
        return False


def convert_legal_file(file_path: Path) -> bool:
    """
    Convert 1 file legal (PDF/DOCX/HTML/TXT/MD) sang Markdown.

    Returns True nếu convert thành công, False nếu lỗi.
    """
    try:
        ext = file_path.suffix.lower()
        source_file = file_path.name
        converted_at = datetime.now(timezone.utc).isoformat()

        # Đọc trực tiếp nếu là .md hoặc .txt
        if ext in (".md", ".txt"):
            content = file_path.read_text(encoding="utf-8")
        else:
            # Dùng MarkItDown nếu có
            if not HAS_MARKITDOWN:
                print(f"  [WARN] MarkItDown not installed, skipping {file_path.name}")
                return False

            converter = MarkItDown()
            result = converter.convert(str(file_path))
            content = result.text_content if hasattr(result, "text_content") else str(result)

        if not content or not content.strip():
            print(f"  [WARN] Empty content in {file_path.name}")
            return False

        # Trích xuất title từ dòng đầu hoặc filename
        lines = content.strip().split("\n")
        title = lines[0].strip().lstrip("#").strip() if lines else file_path.stem
        if len(title) < 3:
            title = file_path.stem

        # Clean content đơn giản
        cleaned = content.strip()
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        frontmatter = build_frontmatter({
            "title": title,
            "source_file": source_file,
            "doc_type": "legal",
            "converted_at": converted_at,
        })

        markdown = f"{frontmatter}\n\n# {title}\n\n{cleaned}\n"

        filename = f"{slugify(title)}.md"
        out_path = LEGAL_OUT / filename

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        return True

    except Exception as e:
        print(f"  [WARN] Cannot convert {file_path.name}: {type(e).__name__}: {e}")
        return False


# ---------------------------------------------------------------------------
# Batch conversion
# ---------------------------------------------------------------------------

def convert_news_articles() -> tuple[int, int, int]:
    """
    Convert toàn bộ news JSON hợp lệ sang Markdown.

    Returns:
        (converted, skipped, failed)
    """
    converted = 0
    skipped = 0
    failed = 0

    if not NEWS_DIR.exists():
        print(f"  [INFO] News directory not found: {NEWS_DIR}")
        return 0, 0, 0

    json_files = sorted(NEWS_DIR.glob("*.json"))
    if not json_files:
        print(f"  [INFO] No JSON files found in {NEWS_DIR}")
        return 0, 0, 0

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                article = json.load(f)

            status = article.get("status")
            is_valid = article.get("is_valid_article", False)
            is_relevant = article.get("is_relevant_topic", False)

            if status == "success" and is_valid and is_relevant:
                ok = convert_news_json_file(json_path)
                if ok:
                    converted += 1
                else:
                    failed += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  [WARN] Cannot read {json_path.name}: {type(e).__name__}: {e}")
            failed += 1

    return converted, skipped, failed


def convert_legal_docs() -> tuple[int, int]:
    """
    Convert toàn bộ legal docs sang Markdown.

    Returns:
        (converted, failed)
    """
    converted = 0
    failed = 0

    if not LEGAL_DIR.exists():
        print(f"  [INFO] Legal directory not found: {LEGAL_DIR}")
        return 0, 0

    legal_files = []
    for ext in ("*.pdf", "*.docx", "*.html", "*.htm", "*.txt", "*.md"):
        legal_files.extend(LEGAL_DIR.glob(ext))

    legal_files = sorted(set(legal_files))

    if not legal_files:
        print(f"  [INFO] No legal files found in {LEGAL_DIR}")
        return 0, 0

    for file_path in legal_files:
        if file_path.is_file():
            ok = convert_legal_file(file_path)
            if ok:
                converted += 1
            else:
                failed += 1

    return converted, failed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Main function: convert landing data sang standardized markdown."""
    setup_directories()

    print("=" * 70)
    print("TASK 3: CONVERT LANDING DATA TO STANDARDIZED MARKDOWN")
    print(f"Landing dir:    {LANDING_DIR}")
    print(f"Standardized:   {STANDARDIZED_DIR}")
    print(f"MarkItDown:     {'available' if HAS_MARKITDOWN else 'not installed'}")
    print("=" * 70)

    # Convert news
    print("\n--- Converting news articles ---")
    news_converted, news_skipped, news_failed = convert_news_articles()
    print(f"  News converted: {news_converted}")
    print(f"  News skipped:   {news_skipped}")
    print(f"  News failed:    {news_failed}")

    # Convert legal docs
    print("\n--- Converting legal documents ---")
    legal_converted, legal_failed = convert_legal_docs()
    print(f"  Legal converted: {legal_converted}")
    print(f"  Legal failed:    {legal_failed}")

    # Summary
    total_converted = news_converted + legal_converted
    total_skipped = news_skipped
    total_failed = news_failed + legal_failed

    print("\n" + "=" * 70)
    print("CONVERSION SUMMARY")
    print(f"  Legal converted:   {legal_converted}")
    print(f"  News converted:    {news_converted}")
    print(f"  Skipped:           {total_skipped}")
    print(f"  Failed:            {total_failed}")
    print(f"  Total converted:   {total_converted}")
    print(f"  News output:      {NEWS_OUT}")
    print(f"  Legal output:     {LEGAL_OUT}")
    print("=" * 70)

    # Warning nếu thiếu file
    if news_converted < 5:
        print("\nWARNING: Task 3 expected at least 5 standardized news markdown files.")


if __name__ == "__main__":
    main()
