"""
Task 2 — Crawl bài báo về nghệ sĩ Việt Nam liên quan tới ma túy.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo về nghệ sĩ Việt Nam liên quan đến ma túy.
    2. Sử dụng Crawl4AI (AsyncWebCrawler).
    3. Lưu output vào data/landing/news/ (success) hoặc data/landing/news_failed/ (failed).
    4. Mỗi bài lưu 1 file JSON với metadata đầy đủ.
"""

import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler

# Fix Windows console encoding for Unicode
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Output directories
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"
OUTPUT_DIR_FAILED = Path(__file__).parent.parent / "data" / "landing" / "news_failed"

# Vietnamese news articles about Vietnamese artists involved in drug-related incidents
# All URLs are verified real articles from reputable Vietnamese news sources
ARTICLE_URLS: list[str] = [
    # VnExpress - Andrea Aybar và Chi Dân bị bắt
    "https://vnexpress.net/nguoi-mau-andrea-aybar-va-ca-si-chi-dan-bi-bat-4814295.html",
    # VnExpress - Anh em ca sĩ Chi Dân rủ nhiều người chơi ma túy
    "https://vnexpress.net/anh-em-ca-si-chi-dan-ru-nhieu-nguoi-choi-ma-tuy-nhu-the-nao-4929804.html",
    # VnExpress - Ca sĩ Long Nhật, Sơn Ngọc Minh bị bắt vì liên quan ma túy
    "https://vnexpress.net/ca-si-long-nhat-son-ngoc-minh-bi-bat-vi-lien-quan-ma-tuy-5060857.html",
    # VnExpress - Ca sĩ Miu Lê bị bắt với cáo buộc tổ chức sử dụng ma túy
    "https://vnexpress.net/ca-si-miu-le-bi-bat-voi-cao-buoc-to-chuc-su-dung-ma-tuy-5074769.html",
    # VnExpress/Ngôi sao - Chi Dân và Andrea Aybar bị khởi tố
    "https://ngoisao.vnexpress.net/chi-dan-va-andrea-aybar-bi-khoi-to-vi-to-chuc-su-dung-ma-tuy-4815983.html",
]

# Keywords for drug-related content
DRUG_KEYWORDS = [
    "ma túy", "ma tuý", "cần sa", "ketamine", "amphetamine",
    "heroin", "cocaine", "thuốc lắc", "chất cấm", "dương tính",
    "sử dụng ma túy", "tàng trữ ma túy", "tổ chức sử dụng ma túy",
    "mua bán ma túy", "vận chuyển ma túy", "ma túy đá", "nước vui",
    "methamphetamine", "mdma", "chất kích thích",
]

# Keywords for artists/showbiz content
ARTIST_KEYWORDS = [
    "nghệ sĩ", "ca sĩ", "diễn viên", "rapper", "người mẫu",
    "hoa hậu", "showbiz", "MC", "DJ", "nhạc sĩ", "diễn viên hài",
    "người có tầm ảnh hưởng", "influencer", "KOL", "artist", "singer",
]

# Generic titles that indicate non-article pages
GENERIC_TITLES = {
    "công an nhân dân - hệ sinh thái truyền thông đa nền tảng",
    "tin tức",
    "pháp luật",
    "trang chủ",
    "404 not found",
    "page not found",
    "không tìm thấy đường dẫn này",
    "406 not acceptable",
    "403 forbidden",
    "500 internal server error",
    "vnexpress - báo tiếng việt nhiều người xem nhất",
}

# Patterns that indicate non-article content (only fail if content is short)
INVALID_CONTENT_PATTERNS = [
    r"không tìm thấy đường dẫn này",
    r"404\s*not\s*found",
    r"page\s*not\s*found",
]


def setup_directories() -> None:
    """Tạo thư mục output nếu chưa tồn tại."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_FAILED.mkdir(parents=True, exist_ok=True)


def is_valid_article_url(url: str) -> bool:
    """
    Kiểm tra URL có phải là bài viết thật hay không.
    
    Args:
        url: URL cần kiểm tra.
        
    Returns:
        True nếu URL có pattern bài viết, False nếu là category/homepage/search/tag.
    """
    parsed = urlparse(url)
    path = parsed.path.lower()
    domain = parsed.netloc.lower().replace('www.', '')
    
    # Bỏ query string và fragment
    path = path.split('?')[0].split('#')[0]
    
    # Check domain cụ thể
    if 'cand' in domain or 'cand.vn' in domain:
        # CAND: phải chứa "-post" và kết thúc ".html"
        if '-post' in path and path.endswith('.html'):
            return True
        return False
    
    elif 'vnexpress' in domain or 'ngoisao.vnexpress' in domain:
        # VnExpress: kết thúc .html và có id số
        if path.endswith('.html') and re.search(r'-\d{6,}\.html$', path):
            return True
        # Ngôi sao VnExpress
        if 'ngoisao.vnexpress' in domain and path.endswith('.html'):
            return True
        return False
    
    elif 'tuoitre' in domain:
        # Tuổi Trẻ: kết thúc .htm hoặc .html
        if path.endswith('.htm') or path.endswith('.html'):
            if '/tim-kiem/' in path:
                return False
            if path.count('/') <= 2:
                return False
            return True
        return False
    
    elif 'thanhnien' in domain:
        # Thanh Niên: kết thúc .html
        if path.endswith('.html'):
            if '/tim-kiem/' in path or '/search/' in path:
                return False
            if path.count('/') <= 2:
                return False
            return True
        return False
    
    elif 'vietnamplus' in domain:
        # VietnamPlus: chứa "post" hoặc kết thúc ".vnp"
        if 'post' in path:
            return True
        if path.endswith('.vnp'):
            return True
        return False
    
    # Default: kiểm tra có ending rõ ràng không
    if path.endswith(('.html', '.htm', '.vnp')):
        return True
    
    return False


def validate_article_content(article: dict) -> tuple[bool, str]:
    """
    Kiểm tra content có phải là bài viết hợp lệ hay không.
    
    Args:
        article: Dict chứa dữ liệu article.
        
    Returns:
        Tuple (is_valid, reason) - True nếu valid, False nếu invalid kèm lý do.
    """
    title = article.get("title", "").lower().strip()
    content = article.get("content_markdown", "")
    content_length = len(content)
    
    # Check 0: URL redirect detection - title chứa brand names khác URL
    url = article.get("url", "").lower()
    if 'vnexpress' in url and 'vnexpress' not in title and content_length < 10000:
        if 'không tìm thấy' in content or '404' in content:
            return False, "404 page (article not found)"
    
    if 'cand' in url or 'cand.vn' in url:
        if 'không tìm thấy' in content and content_length < 10000:
            return False, "404 page (article not found)"
    
    # Check 1: Content rỗng hoặc quá ngắn
    if content_length < 800:
        return False, f"Content too short ({content_length} chars, min 800 required)"
    
    # Check 2: Title rỗng
    if not title or len(title) < 5:
        return False, "Title too short or empty"
    
    # Check 3: Title quá generic
    if title in GENERIC_TITLES:
        return False, f"Generic title: '{title}'"
    
    # Check 4: Title chứa generic keywords
    generic_patterns = [
        r"^tin\s*tức$",
        r"^pháp\s*luật$",
        r"^trang\s*chủ$",
        r"^thời\s*sự$",
        r"^vnexpress\s*-\s*báo\s*tiếng\s*việt\s*nhiều\s*người\s*xem\s*nhất$",  # Homepage title
    ]
    for pattern in generic_patterns:
        if re.match(pattern, title, re.IGNORECASE):
            return False, f"Generic title pattern: '{title}'"
    
    # Check 5: Content chứa invalid patterns (only fail if short)
    for pattern in INVALID_CONTENT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            if content_length < 10000:
                return False, f"Invalid content pattern found: '{pattern}'"
    
    # Check 6: Too many links (indicates menu/listing page)
    link_density = len(re.findall(r'\[.*?\]\(.*?\)', content)) / max(content_length / 100, 1)
    if link_density > 3.0:
        return False, f"Too many links (density: {link_density:.2f}) - likely a menu/listing page"
    
    # Check 7: Too few paragraphs
    paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 100]
    if len(paragraphs) < 3:
        return False, f"Too few paragraphs ({len(paragraphs)}, min 3 required)"
    
    return True, "Valid article"


def is_relevant_artist_drug_article(article: dict) -> tuple[bool, str]:
    """
    Kiểm tra bài báo có đúng chủ đề 'nghệ sĩ Việt Nam liên quan ma túy' không.
    
    Args:
        article: Dict chứa dữ liệu article.
        
    Returns:
        Tuple (is_relevant, reason).
    """
    title = article.get("title", "").lower()
    content = article.get("content_markdown", "").lower()
    
    # Check drug keywords
    has_drug = any(kw in title or kw in content for kw in DRUG_KEYWORDS)
    
    # Check artist/showbiz keywords
    has_artist = any(kw in title or kw in content for kw in ARTIST_KEYWORDS)
    
    if has_drug and has_artist:
        return True, "Relevant: article about Vietnamese artists and drugs"
    
    if not has_drug and not has_artist:
        return False, "Article is not about Vietnamese artists related to drugs (missing both topics)"
    
    if not has_drug:
        return False, "Article is not about Vietnamese artists related to drugs (missing drug topic)"
    
    if not has_artist:
        return False, "Article is not about Vietnamese artists related to drugs (missing artist/showbiz topic)"
    
    return False, "Article is not about Vietnamese artists related to drugs"


def make_safe_filename(text: str, max_length: int = 80) -> str:
    """
    Chuyển text thành filename an toàn cho Windows filesystem.
    """
    safe_chars = []
    for char in text:
        if char.isascii() and (char.isalnum() or char in '_-'):
            safe_chars.append(char)
        elif char.isspace():
            safe_chars.append('_')
        else:
            safe_chars.append('x')
    result = ''.join(safe_chars)
    result = re.sub(r'_+', '_', result)
    result = result.strip('_')
    if len(result) > max_length:
        result = result[:max_length].rstrip('_')
    return result if result else "untitled"


def extract_title(result, fallback_domain: str) -> str:
    """
    Trích xuất title từ kết quả crawl.
    """
    if hasattr(result, 'metadata') and result.metadata:
        title = result.metadata.get("title")
        if title and isinstance(title, str) and title.strip():
            return title.strip()

    if hasattr(result, 'markdown') and result.markdown:
        md = result.markdown
        if hasattr(md, 'raw_markdown'):
            md = md.raw_markdown
        if isinstance(md, str):
            lines = md.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('# ') and len(line) > 2:
                    return line[2:].strip()

    return fallback_domain


def extract_markdown_content(result) -> str:
    """
    Trích xuất markdown content từ kết quả crawl.
    """
    if hasattr(result, 'markdown') and result.markdown:
        md = result.markdown
        if hasattr(md, 'raw_markdown'):
            return str(md.raw_markdown)
        return str(md)

    if hasattr(result, 'cleaned_html') and result.cleaned_html:
        return f"<!-- HTML fallback -->\n{result.cleaned_html}"

    return ""


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Args:
        url: URL của bài báo cần crawl.

    Returns:
        Dict với schema đầy đủ.
    """
    date_crawled = datetime.now(timezone.utc).isoformat()
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.replace('www.', '') or "unknown"

    # Validate URL trước khi crawl
    if not is_valid_article_url(url):
        return {
            "url": url,
            "title": domain,
            "date_crawled": date_crawled,
            "content_markdown": "",
            "content_length": 0,
            "source": domain,
            "status": "failed",
            "is_valid_article": False,
            "is_relevant_topic": False,
            "error": f"Invalid article URL (category/homepage/search page): {url}"
        }

    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)

            title = extract_title(result, domain)
            content_markdown = extract_markdown_content(result)
            content_length = len(content_markdown)
            source = domain

            # Tạo article dict
            article: dict = {
                "url": url,
                "title": title,
                "date_crawled": date_crawled,
                "content_markdown": content_markdown,
                "content_length": content_length,
                "source": source,
                "status": "success",
                "is_valid_article": False,
                "is_relevant_topic": False,
                "error": None
            }

            # Validate content
            is_valid, reason = validate_article_content(article)
            article["is_valid_article"] = is_valid

            if not is_valid:
                article["status"] = "failed"
                article["error"] = reason
                return article

            # Validate topic relevance
            is_relevant, reason = is_relevant_artist_drug_article(article)
            article["is_relevant_topic"] = is_relevant

            if not is_relevant:
                article["status"] = "failed"
                article["error"] = reason

            return article

    except Exception as e:
        return {
            "url": url,
            "title": domain,
            "date_crawled": date_crawled,
            "content_markdown": "",
            "content_length": 0,
            "source": domain,
            "status": "failed",
            "is_valid_article": False,
            "is_relevant_topic": False,
            "error": f"{type(e).__name__}: {str(e)}"
        }


def save_article(article: dict, index: int) -> tuple[Path, bool]:
    """
    Lưu article vào file JSON.

    Args:
        article: Dict chứa dữ liệu article.
        index: Số thứ tự để đặt tên file.

    Returns:
        Tuple (filepath, is_success) - Path tới file đã lưu và trạng thái.
    """
    title_slug = make_safe_filename(article["title"])
    source_slug = make_safe_filename(article["source"])

    filename = f"{index:03d}_{source_slug}_{title_slug}.json"
    
    # Save to success or failed directory
    # Only success if: valid article AND relevant topic
    if (article["status"] == "success" 
        and article.get("is_valid_article", False) 
        and article.get("is_relevant_topic", False)):
        output_dir = OUTPUT_DIR
        is_success = True
    else:
        output_dir = OUTPUT_DIR_FAILED
        is_success = False

    filepath = output_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(article, f, ensure_ascii=False, indent=2)
    
    return filepath, is_success


async def main() -> list[dict]:
    """
    Main function: crawl toàn bộ ARTICLE_URLS và lưu kết quả.
    """
    setup_directories()

    results: list[dict] = []
    total = len(ARTICLE_URLS)

    print("=" * 70)
    print("TASK 2: CRAWL NEWS ARTICLES ABOUT VIETNAMESE ARTISTS AND DRUGS")
    print(f"Success output: {OUTPUT_DIR}")
    print(f"Failed output:  {OUTPUT_DIR_FAILED}")
    print(f"Total URLs:     {total}")
    print("=" * 70)

    for index, url in enumerate(ARTICLE_URLS, start=1):
        print(f"\n[{index}/{total}] Processing: {url}")

        # Check URL validity before crawl
        if not is_valid_article_url(url):
            article = {
                "url": url,
                "title": urlparse(url).netloc,
                "date_crawled": datetime.now(timezone.utc).isoformat(),
                "content_markdown": "",
                "content_length": 0,
                "source": urlparse(url).netloc.replace('www.', ''),
                "status": "failed",
                "is_valid_article": False,
                "is_relevant_topic": False,
                "error": "Invalid article URL (category/homepage/search page)"
            }
        else:
            article = await crawl_article(url)

        try:
            filepath, is_success = save_article(article, index)
            
            if is_success:
                title = article['title'][:60].encode('ascii', 'replace').decode('ascii')
                print(f"  [SUCCESS] Title: {title}...")
                print(f"  [SUCCESS] Content: {article['content_length']} chars")
            else:
                print(f"  [FAILED] Reason: {article['error']}")
            
            print(f"  [SAVED] {filepath.as_posix()}")
            
        except Exception as e:
            print(f"  [ERROR] Cannot save: {type(e).__name__}: {e}")

        results.append(article)

    # Tổng kết
    success_count = sum(1 for r in results
                       if r["status"] == "success"
                       and r.get("is_valid_article", False)
                       and r.get("is_relevant_topic", False))
    failed_count = total - success_count
    meets_requirement = success_count >= 5

    print("\n" + "=" * 70)
    print("CRAWL SUMMARY")
    print(f"  Total URLs:          {total}")
    print(f"  Success:            {success_count}")
    print(f"  Failed:             {failed_count}")
    print(f"  Meets requirement:  {meets_requirement} (need >= 5)")
    print(f"  Success dir:        {OUTPUT_DIR}")
    print(f"  Failed dir:         {OUTPUT_DIR_FAILED}")
    print("=" * 70)

    if not meets_requirement:
        print(f"\n[WARNING] Only {success_count} article(s) crawled successfully.")
        print(f"[WARNING] Task requires at least 5 valid articles about Vietnamese artists and drugs.")
        print(f"[WARNING] Please check failed articles in: {OUTPUT_DIR_FAILED}")

    return results


if __name__ == "__main__":
    asyncio.run(main())
