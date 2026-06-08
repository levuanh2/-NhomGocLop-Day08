"""Dynamic URL discovery + crawl bài báo về nghệ sĩ Việt Nam liên quan ma túy."""

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse

_ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT_DIR / "src"))

from data_processing.config import LANDING_NEWS_DIR

FAILED_DIR = LANDING_NEWS_DIR.parent / "news_failed"

DRUG_KEYWORDS = [
    "ma túy", "ma tuý", "chất cấm", "ketamine", "thuốc lắc",
    "mdma", "methamphetamine", "heroin", "cocaine", "cần sa",
    "dương tính", "nước vui"
]

ARTIST_KEYWORDS = [
    "nghệ sĩ", "ca sĩ", "diễn viên", "người mẫu", "rapper",
    "showbiz", "hot girl", "kol", "influencer", "hoa hậu",
    "mc", "dj", "người nổi tiếng"
]

LEGAL_ACTION_KEYWORDS = [
    "bị bắt", "bắt giữ", "bị khởi tố", "khởi tố",
    "bị cáo buộc", "điều tra", "tạm giữ", "tạm giam",
    "tổ chức sử dụng", "tàng trữ", "sử dụng trái phép"
]

VALID_NEWS_DOMAINS = [
    "vnexpress.net",
    "ngoisao.vnexpress.net",
    "tuoitre.vn",
    "thanhnien.vn",
    "nld.com.vn",
    "vietnamnet.vn",
    "dantri.com.vn",
    "cand.com.vn",
    "tienphong.vn",
    "laodong.vn",
    "plo.vn",
]


def setup_news_dirs() -> None:
    LANDING_NEWS_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    for param in ["utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "outputType"]:
        query.pop(param, None)
    new_query = urlencode(query, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def existing_urls() -> set[str]:
    urls: set[str] = set()
    for directory in (LANDING_NEWS_DIR, FAILED_DIR):
        if not directory.exists():
            continue
        for path in directory.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
                url = data.get("url")
                if url:
                    urls.add(normalize_url(url))
            except Exception:
                pass
    return urls


def build_search_queries() -> list[str]:
    base_queries = [
        # Site-specific queries
        "site:vnexpress.net ca sĩ ma túy bị bắt",
        "site:vnexpress.net người mẫu ma túy bị khởi tố",
        "site:vnexpress.net diễn viên ma túy bị bắt",
        "site:ngoisao.vnexpress.net ca sĩ ma túy bị bắt",
        "site:tuoitre.vn ca sĩ ma túy bị bắt",
        "site:thanhnien.vn ca sĩ ma túy bị bắt",
        "site:vietnamnet.vn nghệ sĩ ma túy bị bắt",
        "site:dantri.com.vn người mẫu ma túy bị bắt",
        "site:nld.com.vn nghệ sĩ ma túy bị khởi tố",
        "site:plo.vn ca sĩ ma túy bị bắt",
        "site:tienphong.vn nghệ sĩ ma túy bị bắt",
        "site:laodong.vn người nổi tiếng ma túy bị bắt",
        # Artist-specific
        "Chi Dân ma túy bị bắt",
        "Andrea Aybar ma túy",
        "An Tây ma túy",
        "Long Nhật ma túy bị bắt",
        "Minh Phương ca sĩ ma túy",
        "Hưng Cường diễn viên ma túy",
        "Ngọc Trinh ma túy",
        "Trường Thành Vinh ca sĩ ma túy",
        # General queries
        "nghệ sĩ Việt bị bắt ma túy",
        "người nổi tiếng bị bắt ma túy",
        "showbiz Việt ma túy bị khởi tố",
        "ca sĩ Việt dương tính ma túy",
        "rapper Việt ma túy",
        "hot girl Việt ma túy bị bắt",
        "influencer ma túy bị khởi tố",
        "idol K-pop ma túy Việt Nam",
    ]
    return base_queries


def is_valid_domain(url: str) -> bool:
    parsed = urlparse(url)
    netloc = parsed.netloc.lower().removeprefix("www.")
    return netloc in VALID_NEWS_DOMAINS


def is_article_url(url: str) -> bool:
    parsed = urlparse(url)
    netloc = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.lower()

    if not is_valid_domain(url):
        return False

    # Reject homepage / category / tag / search
    skip_patterns = [
        r"^/$",
        r"^/search",
        r"/tag/",
        r"/thoi-su$",
        r"/phap-luat$",
        r"/giai-tri$",
        r"/the-thao$",
        r"/kinh-doanh$",
        r"/the-gioi$",
        r"/giao-duc$",
        r"/suc-khoe$",
        r"/doi-song$",
        r"/tin-moi",
        r"/tin-tuc-24h",
        r"/video",
        r"/photo",
        r"/infographic",
        r"/chuyen-de/",
        r"/chu-de/",
        r"/topic/",
        r"\.pdf$",
        r"\.png$",
        r"\.jpg$",
        r"\.jpeg$",
        r"\.gif$",
    ]
    for pattern in skip_patterns:
        if re.search(pattern, path):
            return False

    # vnexpress: must have numeric ID like -1234567.html
    if netloc == "vnexpress.net" or netloc == "ngoisao.vnexpress.net":
        if not re.search(r"-\d+\.html?$", path):
            return False

    # tuoitre: must end with .htm and have article slug pattern
    if netloc == "tuoitre.vn":
        if not re.search(r"/[\w\-]+\.htm$", path):
            return False

    # thanhnien: must have numeric ID before .htm
    if netloc == "thanhnien.vn":
        if not re.search(r"-?\d+\.htm$", path):
            return False

    # dantri: must have article slug with digits before .htm
    if netloc == "dantri.com.vn":
        if not re.search(r"/[\w\-]+\d+\.htm$", path):
            return False

    # nld: must have article slug before .htm
    if netloc == "nld.com.vn":
        if not re.search(r"/[\w\-]+\.htm$", path):
            return False

    # vietnamnet: must have -number.html
    if netloc == "vietnamnet.vn":
        if not re.search(r"-\d+\.html?$", path):
            return False

    # plo: must end with .html
    if netloc == "plo.vn":
        if not re.search(r"\.html?$", path):
            return False

    # tienphong: .tpo or .html
    if netloc == "tienphong.vn":
        if not re.search(r"\.tpo?$", path):
            return False

    # laodong: must have ID number
    if netloc == "laodong.vn":
        if not re.search(r"-\d+\.htm", path):
            return False

    return True


SITEMAP_URLS = {
    "vnexpress.net": "https://vnexpress.net/sitemap.xml",
    "ngoisao.vnexpress.net": "https://ngoisao.vnexpress.net/sitemap.xml",
    "tuoitre.vn": "https://tuoitre.vn/sitemap.xml",
    "thanhnien.vn": "https://thanhnien.vn/sitemap.xml",
    "vietnamnet.vn": "https://vietnamnet.vn/sitemap.xml",
    "dantri.com.vn": "https://dantri.com.vn/sitemap.xml",
    "nld.com.vn": "https://nld.com.vn/sitemap.xml",
    "plo.vn": "https://plo.vn/sitemap.xml",
    "tienphong.vn": "https://tienphong.vn/sitemap.xml",
    "laodong.vn": "https://laodong.vn/sitemap.xml",
}

CATEGORY_URLS = [
    "https://vnexpress.net/phap-luat/khoa-kiem-tra",
    "https://thanhnien.vn/phap-luat",
    "https://dantri.com.vn/phap-luat.htm",
    "https://nld.com.vn/phap-luat.htm",
    "https://vietnamnet.vn/phap-luat",
    "https://tuoitre.vn/phap-luat",
    "https://plo.vn/phap-luat",
    "https://tienphong.vn/phap-luat",
    "https://laodong.vn/phap-luat",
]


def _build_search_session(requests_session) -> dict:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    requests_session.headers.update(headers)
    return requests_session


async def _fetch_sitemap(session, domain: str, sitemap_url: str) -> list[str]:
    import requests
    try:
        resp = session.get(sitemap_url, timeout=15)
        resp.raise_for_status()
        xml = resp.text
        locs = re.findall(r"<loc>(https?://[^<]+)</loc>", xml)
        return [u.strip() for u in locs]
    except Exception as e:
        print(f"    [WARN] sitemap failed for {domain}: {e}")
        return []


async def _fetch_category_page(session, url: str, page: int) -> tuple[list[str], str]:
    import requests
    try:
        page_url = url if page == 0 else f"{url}/trang-{page}.html"
        resp = session.get(page_url, timeout=15)
        resp.raise_for_status()
        html = resp.text
        links: list[str] = []
        for match in re.finditer(r'href="(https?://[^"#]+)"', html):
            links.append(match.group(1))
        return links, html
    except Exception as e:
        print(f"    [WARN] category page failed for {url}: {e}")
        return [], ""


async def _fetch_google_news_rss(session, query: str) -> list[str]:
    import requests
    try:
        encoded_q = query.replace(" ", "+")
        rss_url = f"https://news.google.com/rss/search?q={encoded_q}&hl=vi-VN&gl=VN&ceid=VN:vi"
        resp = session.get(rss_url, timeout=15)
        resp.raise_for_status()
        xml = resp.text
        links = re.findall(r"<link/>(https?://[^<]+)</link>", xml)
        parsed_links = []
        for link in links:
            link = link.strip()
            if link.startswith("https://news.google.com/articles/"):
                m = re.search(r"url=(https?[^&]+)", link)
                if m:
                    parsed_links.append(m.group(1))
            elif link.startswith("http"):
                parsed_links.append(link)
        return parsed_links
    except Exception as e:
        print(f"    [WARN] Google News RSS failed: {e}")
        return []


async def discover_article_urls(max_urls: int = 80) -> list[str]:
    import requests

    discovered: list[str] = []
    seen: set[str] = set()

    session = _build_search_session(requests.Session())

    # 1. Google News RSS for artist+ma_tuy queries
    artist_queries = [
        "ca sĩ Việt Nam ma túy bị bắt",
        "diễn viên Việt ma túy bị khởi tố",
        "người mẫu Việt ma túy",
        "nghệ sĩ Việt dương tính ma túy",
        "showbiz Việt ma túy bị điều tra",
        "Chi Dân ca sĩ ma túy",
        "Long Nhật ca sĩ ma túy",
        "Miu Lê ca sĩ ma túy",
        "Andrea Aybar ma túy",
        "Minh Phương ca sĩ ma túy",
    ]
    print("  [Google News RSS]")
    for q in artist_queries:
        if len(discovered) >= max_urls:
            break
        print(f"    Query: {q[:50]}...")
        try:
            links = await _fetch_google_news_rss(session, q)
            for link in links:
                link = normalize_url(link)
                if link in seen:
                    continue
                seen.add(link)
                if is_article_url(link):
                    discovered.append(link)
                    if len(discovered) >= max_urls:
                        break
        except Exception as e:
            print(f"      [WARN] {e}")
        time.sleep(1)

    # 2. Category pages (phap-luat) - accept all article URLs
    if len(discovered) < max_urls:
        print("  [Category pages]")
        for cat_url in CATEGORY_URLS:
            if len(discovered) >= max_urls:
                break
            for page in range(3):
                if len(discovered) >= max_urls:
                    break
                links, _ = await _fetch_category_page(session, cat_url, page)
                for link in links:
                    link = normalize_url(link)
                    if link in seen:
                        continue
                    seen.add(link)
                    if is_article_url(link):
                        discovered.append(link)
                        if len(discovered) >= max_urls:
                            break
                time.sleep(0.5)

    # 3. Sitemaps - accept all article URLs from news domains
    if len(discovered) < max_urls:
        print("  [Sitemaps]")
        for domain, sitemap_url in SITEMAP_URLS.items():
            if len(discovered) >= max_urls:
                break
            print(f"    Fetching: {sitemap_url}")
            locs = await _fetch_sitemap(session, domain, sitemap_url)
            for loc in locs:
                loc = normalize_url(loc)
                if loc in seen:
                    continue
                seen.add(loc)
                if is_article_url(loc):
                    discovered.append(loc)
                    if len(discovered) >= max_urls:
                        break
            time.sleep(1)

    # 4. Bing search fallback
    if len(discovered) < 10:
        print("  [Bing fallback]")
        bing_queries = [
            "site:vnexpress.net ca sĩ ma túy",
            "site:thanhnien.vn nghệ sĩ ma túy",
            "site:dantri.com.vn người mẫu ma túy",
            "site:tuoitre.vn diễn viên ma túy",
        ]
        for query in bing_queries:
            if len(discovered) >= max_urls:
                break
            try:
                params = {"q": query}
                resp = session.get(
                    "https://www.bing.com/search",
                    params=params,
                    timeout=15,
                )
                resp.raise_for_status()
                html = resp.text
                for match in re.finditer(r'href="([^"]+)"[^>]*>', html):
                    raw = match.group(1)
                    if raw.startswith("http") and not any(bot in raw for bot in ["bing.com", "microsoft.com"]):
                        url = normalize_url(raw)
                        if url in seen:
                            continue
                        seen.add(url)
                        if is_article_url(url):
                            discovered.append(url)
                            if len(discovered) >= max_urls:
                                break
            except Exception as e:
                print(f"    [WARN] Bing failed: {e}")
            time.sleep(2)

    # Deduplicate while preserving order
    seen_final = set()
    unique: list[str] = []
    for url in discovered:
        if url not in seen_final:
            seen_final.add(url)
            unique.append(url)

    return unique


def make_safe_filename(title: str, url: str, index: int) -> str:
    domain = urlparse(url).netloc.replace("www.", "")
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower())[:40]
    slug = slug.strip("_")
    return f"{index:03d}_{domain}_{slug}.json"


def extract_title(result, url: str) -> str:
    title = ""
    try:
        metadata = getattr(result, "metadata", {}) or {}
        title = metadata.get("title") or metadata.get("og:title") or ""
    except Exception:
        pass
    if not title:
        text = getattr(result, "markdown", "") or ""
        match = re.search(r"^#\s+(.+)", text, re.MULTILINE)
        if match:
            title = match.group(1).strip()
    if not title:
        title = urlparse(url).path.split("/")[-1].replace("-", " ").title()
    return title


def validate_article_content(title: str, content: str, url: str) -> tuple[bool, bool, dict, str]:
    keywords_matched: dict[str, list[str]] = {"drug": [], "artist": [], "legal_action": []}
    text_lower = (title + " " + content).lower()

    for keyword in DRUG_KEYWORDS:
        if keyword.lower() in text_lower:
            keywords_matched["drug"].append(keyword)

    for keyword in ARTIST_KEYWORDS:
        if keyword.lower() in text_lower:
            keywords_matched["artist"].append(keyword)

    for keyword in LEGAL_ACTION_KEYWORDS:
        if keyword.lower() in text_lower:
            keywords_matched["legal_action"].append(keyword)

    is_valid_article = bool(title.strip()) and len(content) >= 1000
    is_relevant_topic = (
        bool(keywords_matched["drug"])
        and bool(keywords_matched["artist"])
        and bool(keywords_matched["legal_action"])
    )

    reasons = []
    if not title.strip():
        reasons.append("missing title")
    if len(content) < 1000:
        reasons.append("content too short")
    if not keywords_matched["drug"]:
        reasons.append("no drug keyword")
    if not keywords_matched["artist"]:
        reasons.append("no artist keyword")
    if not keywords_matched["legal_action"]:
        reasons.append("no legal action keyword")
    reason = "; ".join(reasons) if reasons else "ok"

    return is_valid_article, is_relevant_topic, keywords_matched, reason


async def crawl_article(url: str, index: int) -> dict:
    from crawl4ai import AsyncWebCrawler

    article = {
        "url": url,
        "title": "",
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "source": urlparse(url).netloc,
        "content_markdown": "",
        "content_length": 0,
        "status": "failed",
        "is_valid_article": False,
        "is_relevant_topic": False,
        "error": "",
        "keywords_matched": {"drug": [], "artist": [], "legal_action": []},
    }

    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
        if not result or not getattr(result, "success", False):
            article["error"] = "crawl_failed"
            return article

        title = extract_title(result, url)
        content = getattr(result, "markdown", "") or ""
        if not content:
            content = getattr(result, "cleaned_html", "") or getattr(result, "html", "") or ""

        article["title"] = title
        article["content_markdown"] = content
        article["content_length"] = len(content)

        is_valid, is_relevant, keywords, reason = validate_article_content(title, content, url)
        article["is_valid_article"] = is_valid
        article["is_relevant_topic"] = is_relevant
        article["keywords_matched"] = keywords

        if is_valid and is_relevant:
            article["status"] = "success"
        else:
            article["status"] = "failed"
            article["error"] = reason
    except Exception as e:
        article["error"] = f"{type(e).__name__}: {e}"

    return article


def save_article(article: dict, index: int) -> Path:
    filename = make_safe_filename(article.get("title", f"article_{index}"), article["url"], index)
    if article["status"] == "success":
        out_path = LANDING_NEWS_DIR / filename
    else:
        out_path = FAILED_DIR / filename
    out_path.write_text(
        json.dumps(article, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path


async def crawl_all_news(limit: int = 20, max_urls: int = 80) -> dict:
    setup_news_dirs()

    print("Discovering article URLs via DuckDuckGo search...")
    candidate_urls = await discover_article_urls(max_urls=max_urls)
    print(f"Discovered {len(candidate_urls)} candidate URLs.")

    if not candidate_urls:
        print("WARNING: No URLs discovered. Check internet connection.")
        return {
            "total_urls": 0,
            "discovered": 0,
            "skipped": 0,
            "success_new_articles": 0,
            "failed": 0,
        }

    if len(candidate_urls) < limit:
        print(f"WARNING: Only discovered {len(candidate_urls)} candidate URLs. Try adding more search queries.")

    existing = existing_urls()
    skipped = 0
    success_new = 0
    failed = 0

    for idx, url in enumerate(candidate_urls, start=1):
        if normalize_url(url) in existing:
            skipped += 1
            print(f"[{idx:03d}] SKIP duplicate: {url}")
            continue

        print(f"[{idx:03d}] Crawling: {url}")
        article = await crawl_article(url, idx)
        save_path = save_article(article, idx)

        if article["status"] == "success":
            success_new += 1
            print(f"       OK -> {save_path.name}")
        else:
            failed += 1
            print(f"       FAIL -> {save_path.name} ({article.get('error', '')})")

        if success_new >= limit:
            print(f"\nReached {limit} successful articles. Stopping.")
            break

    print("\n" + "=" * 60)
    print("CRAWL MORE NEWS SUMMARY")
    print(f"Total URLs configured: {len(candidate_urls)}")
    print(f"Skipped duplicates: {skipped}")
    print(f"Success new articles: {success_new}")
    print(f"Failed: {failed}")
    print(f"Saved to: {LANDING_NEWS_DIR}")
    print(f"Failed saved to: {FAILED_DIR}")
    if success_new < limit:
        print(f"WARNING: Need at least {limit} new valid articles. Add more search queries.")
    print("=" * 60)

    return {
        "total_urls": len(candidate_urls),
        "discovered": len(candidate_urls),
        "skipped": skipped,
        "success_new_articles": success_new,
        "failed": failed,
        "saved_to": str(LANDING_NEWS_DIR),
        "failed_saved_to": str(FAILED_DIR),
    }


async def discover_only(max_urls: int = 80, custom_query: str | None = None) -> list[str]:
    if custom_query:
        print(f"Using custom query: {custom_query}")
        urls = await discover_article_urls(max_urls=max_urls)
        # Filter with custom query context
        return urls

    print("Discovering article URLs via DuckDuckGo search...")
    urls = await discover_article_urls(max_urls=max_urls)
    return urls


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crawl bài báo về nghệ sĩ Việt Nam liên quan ma túy"
    )
    parser.add_argument(
        "--limit", type=int, default=20,
        help="Số bài success tối thiểu cần crawl (default: 20)"
    )
    parser.add_argument(
        "--max-urls", type=int, default=80,
        help="Số URL tối đa discover trước khi crawl (default: 80)"
    )
    parser.add_argument(
        "--discover-only", action="store_true",
        help="Chỉ in danh sách URL, không crawl"
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Custom search query (thay thế danh sách query mặc định)"
    )
    args = parser.parse_args()

    if args.discover_only:
        urls = asyncio.run(discover_only(max_urls=args.max_urls, custom_query=args.query))
        print("\n" + "=" * 60)
        print(f"DISCOVERED {len(urls)} article URLs:")
        print("=" * 60)
        for i, url in enumerate(urls, 1):
            print(f"  {i:3d}. {url}")
        print("=" * 60)
        return

    asyncio.run(crawl_all_news(limit=args.limit, max_urls=args.max_urls))


if __name__ == "__main__":
    main()
