"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    # Vụ ca sĩ Miu Lê
    "https://vnexpress.net/ca-si-miu-le-bi-bat-qua-tang-dung-ma-tuy-o-bai-bien-5072657.html",
    "https://vnexpress.net/ca-si-miu-le-bi-bat-voi-cao-buoc-to-chuc-su-dung-ma-tuy-5074769.html",
    "https://tuoitre.vn/khoi-to-tam-giam-ca-si-miu-le-20260516213959367.htm",
    "https://tuoitre.vn/khoi-to-3-bi-can-trong-vu-ca-si-miu-le-su-dung-ma-tuy-o-cat-ba-20260514230349573.htm",
    # Vụ ca sĩ Long Nhật và Sơn Ngọc Minh
    "https://tuoitre.vn/bat-ca-si-long-nhat-va-ca-si-son-ngoc-minh-vi-lien-quan-ma-tuy-20260520082138943.htm",
    "https://thanhnien.vn/ca-si-long-nhat-bi-bat-showbiz-viet-lien-tiep-chan-dong-vi-ma-tuy-18526052013032001.htm",
    "https://thanhnien.vn/ca-si-son-ngoc-minh-vua-bi-bat-vi-lien-quan-den-ma-tuy-la-ai-18526052012481811.htm",
    "https://tuoitre.vn/ca-si-long-nhat-khai-su-dung-ma-tuy-da-cung-quan-ly-20260520132251413.htm",
    # Vụ ca sĩ Châu Việt Cường và tổng quan
    "https://tienphong.vn/nhin-lai-vu-ca-si-chau-viet-cuong-nhet-toi-lam-chet-nguoi-rung-dong-showbiz-viet-post1095412.tpo",
    "https://thanhnien.vn/ma-tuy-va-showbiz-su-thanh-loc-can-bat-dau-tu-nghe-si-185260513123425952.htm",
]


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
        title = "Unknown"
        if result.metadata:
            title = result.metadata.get("title", "Unknown")
        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": result.markdown or "",
        }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)
            filename = f"article_{i:02d}.json"
            filepath = DATA_DIR / filename
            filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  ✓ Saved: {filepath.name}  (title: {article['title'][:60]})")
        except Exception as e:
            print(f"  ✗ Lỗi khi crawl {url}: {e}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
