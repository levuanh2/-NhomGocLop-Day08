"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    python -m playwright install chromium
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

ARTICLE_URLS = [
    "https://vov.vn/giai-tri/chua-day-1-thang-3-nghe-si-viet-bi-khoi-to-vi-lien-quan-ma-tuy-gay-chan-dong-post1293496.vov",
    "https://vietnamnet.vn/sao-viet-bi-bat-ngoi-tu-mat-danh-tieng-vi-chat-cam-2513746.html",
    "https://nld.com.vn/showbiz-viet-nhung-nghe-si-gay-soc-vi-be-boi-ma-tuy-196250725113547841.htm",
    "https://baolaocai.vn/bao-dong-tinh-trang-nghe-si-dung-ma-tuy-va-nhung-he-luy-voi-xa-hoi-post900028.html",
    "https://tienphong.vn/lien-tiep-nghe-si-dung-chat-cam-post1842599.tpo"
]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _strip_markdown_links(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]+\)", r"\1", text)
    return text.strip()


def extract_article_content(markdown: str, title: str) -> str:
    """Lọc markdown để chỉ giữ tiêu đề và nội dung chính của bài báo."""
    lines = markdown.splitlines()
    normalized_title = _normalize_text(title)
    start_index = 0

    for i, line in enumerate(lines):
        plain_line = _normalize_text(_strip_markdown_links(line).lstrip("# "))
        if normalized_title and normalized_title in plain_line:
            start_index = i
            break

    stop_markers = (
        "tag:",
        "tags:",
        "từ khóa:",
        "tin liên quan",
        "bình luận",
        "viết bình luận",
        "gửi bình luận",
        "xem thêm",
        "đọc tiếp",
        "cùng chuyên mục",
        "có thể bạn quan tâm",
        "theo dõi chúng tôi",
        "mời quý độc giả theo dõi",
    )
    skip_markers = (
        "google news",
        "chia sẻ",
        "sao chép liên kết",
        "copy link",
        "báo nói",
        "font size",
    )

    cleaned_lines = []
    previous_line = ""

    for raw_line in lines[start_index:]:
        stripped = raw_line.strip()
        plain = _strip_markdown_links(stripped)
        normalized = _normalize_text(plain.lstrip("# "))

        if cleaned_lines and (
            normalized in stop_markers
            or any(normalized.startswith(marker) for marker in stop_markers)
        ):
            break

        if not stripped or stripped.startswith("![") or stripped.startswith("[!["):
            continue
        if re.match(r"^#{2,6}\s*\[", stripped):
            continue
        if any(marker in normalized for marker in skip_markers):
            continue
        if re.fullmatch(r"[\[\]\(\){}#*_\-\\/\s|:;,.!]*", plain):
            continue

        if normalized_title and normalized_title in normalized:
            line = f"# {title}"
        else:
            line = plain

        if line and line != previous_line:
            cleaned_lines.append(line)
            previous_line = line

    return "\n\n".join(cleaned_lines).strip()


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        metadata = getattr(result, "metadata", {}) or {}
        title = metadata.get("title", "Unknown")
        raw_markdown = getattr(result, "markdown", "") or ""
        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": extract_article_content(raw_markdown, title),
        }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2))
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
