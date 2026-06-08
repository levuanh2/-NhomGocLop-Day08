import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from data_processing.config import (
    LANDING_LEGAL_DIR,
    LANDING_NEWS_DIR,
    STANDARDIZED_LEGAL_DIR,
    STANDARDIZED_NEWS_DIR,
)

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

try:
    from markitdown import MarkItDown

    HAS_MARKITDOWN = True
except ImportError:
    HAS_MARKITDOWN = False


def setup_standardized_dirs() -> None:
    STANDARDIZED_LEGAL_DIR.mkdir(parents=True, exist_ok=True)
    STANDARDIZED_NEWS_DIR.mkdir(parents=True, exist_ok=True)


def slugify(text: str, max_length: int = 80) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", "", text)
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")
    if len(text) > max_length:
        text = text[:max_length].rstrip("_")
    return text if text else "untitled"


def _build_yaml_frontmatter(metadata: dict) -> str:
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


FOOTER_MARKERS = [
    "[ Trở lại",
    "Trở lại Pháp luật",
    "Trở lại Hình sự",
    "Trở lại",
    "Gửi email cho tác giả",
    "Về trang chủ",
    "Hotline:",
    "Tải ứng dụng",
    "Điều khoản sử dụng",
    "Tòa soạn",
    "Đăng ký nhận thông báo",
]

# Regex: chỉ xoá nếu TOÀN dòng khớp
_LINE_NOISE = re.compile(
    r"""
    # JavaScript links bất kỳ dạng
    ^\[[\s\S]*?\]\(javascript:  |
    # Author avatar block: [ ![name](img) ](url "name")
    ^\[\s*!\[.*?\]\(https?://.*?\)\s*\]\(  |
    # Google News banner
    news\.google\.com  |
    # Site logo/banner image link
    ^\[\s*!\[.*?(logo|tagline|banner|vne|graphics).*?\]\(https?://  |
    # CDN image lines: các domain ảnh phổ biến của báo VN
    ^!\[.*?\]\(https?://(cdn|static|images?|s\d+|img|thumb|i\d+|icdn)\d*\.  |
    # VNExpress CDN
    ^!\[.*?\]\(https?://[^\s)]*vnecdn\.net  |
    # Empty list items: * (nothing)
    ^\s*\*\s*$  |
    # Social share list items: * [](javascript:; "Chia sẻ...")
    ^\s*\*\s*\[\s*\]\(javascript:  |
    # Social action nav row: [ Nghe ]( [ Chia sẻ ]( ...
    ^\[\s*(Nghe|Chia\s+sẻ|Lưu\s+tin|Bình\s+luận|Trở\s+về)\s*\]\(  |
    # Date-only lines: 20/05/2026 13:56 GMT+7
    ^\d{1,2}/\d{1,2}/\d{4}\s+\d{2}:\d{2}  |
    # Navigation items
    ^\[?\s*(Trang\s+chủ|Mới\s+nhất|Tin\s+theo\s+khu\s+vực|International|
             Đăng\s+nhập|Đăng\s+ký|Thông\s+báo|Ý\s+kiến|Tin\s+đã\s+xem|
             Bật\s+nhận\s+thông\s+báo|Đang\s+tải|Tải\s+ứng\s+dụng|
             Chia\s+sẻ|Bình\s+luận|Copy\s+link|Gửi\s+email|Mail|
             Về\s+trang\s+chủ|Lên\s+đầu\s+trang|Liên\s+hệ\s+quảng\s+cáo|
             Hotline|Copyright|Điều\s+khoản)\s*\]?\(?  |
    # Thứ X, ngày giờ
    ^\[?\s*Thứ\s+\w+,\s*\d+/\d+/\d+  |
    # Topic-only lines (TP HCM, Hà Nội...)
    ^(TP\s+HCM|Hà\s+Nội|Đà\s+Nẵng|Hải\s+Phòng)\s*$  |
    # Topic tag bullets from VNExpress
    ^\s*\*\s*\[.*?\]\(https?://vnexpress\.net/(topic|chu-de)  |
    # Related article bullets with #### heading
    ^\s*\*\s*####\s*\[  |
    # Author email lines
    \S+@\S+\.\w{2,6}\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _strip_before_article_title(content: str, title: str) -> str:
    if not title:
        return content
    lines = content.split("\n")
    # Tìm dòng khớp chính xác tiêu đề
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped in (f"# {title}", f"## {title}", f"#  {title}", f"##  {title}"):
            return "\n".join(lines[i:])
    # Fallback: heading đầu tiên đủ dài
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (stripped.startswith("# ") or stripped.startswith("## ")) and len(stripped) > 20:
            return "\n".join(lines[i:])
    return content


def _strip_after_footer(content: str) -> str:
    lines = content.split("\n")
    cut_index = len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip()
        for marker in FOOTER_MARKERS:
            if marker in stripped:
                cut_index = min(cut_index, i)
                break
    return "\n".join(lines[:cut_index])


def _inline_links_to_text(line: str) -> str:
    """Chuyển internal topic links [text](url "title") → text, giữ external links."""
    # [text](https://site.vn/topic/... "title") → text  (internal topic links)
    line = re.sub(
        r"\[([^\]]+)\]\(https?://[^\s)]+\s+\"[^\"]*\"\)",
        r"\1",
        line,
    )
    return line


def _clean_news_content(content: str, title: Optional[str] = None) -> str:
    if title:
        content = _strip_before_article_title(content, title)

    lines = content.split("\n")
    cleaned: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Dòng trống: giữ tối đa 1 dòng trống liên tiếp
        if not stripped:
            if cleaned and cleaned[-1] == "":
                continue
            cleaned.append("")
            continue

        # Xoá toàn dòng nếu khớp pattern noise
        if _LINE_NOISE.search(stripped):
            continue

        # Chuyển inline topic links thành plain text: [text](url "title") → text
        line = _inline_links_to_text(line)

        # Xoá tiền tố "TPO - " (Tiền Phong Online)
        line = re.sub(r"^TPO\s*[-–]\s*", "", line)

        # Re-compute stripped sau khi convert (important: link text đã được giải ra)
        stripped2 = line.strip()

        # Xoá "và X tác giả khác"
        if re.match(r"^và\s+\d+\s+tác\s+giả\s+khác", stripped2, re.IGNORECASE):
            continue

        # Xoá dòng trống sau convert
        if not stripped2:
            if cleaned and cleaned[-1] == "":
                continue
            cleaned.append("")
            continue

        # Xoá dòng chỉ là tên tác giả (≤3 từ hoặc toàn chữ HOA ≤4 từ, không số/URL)
        words2 = stripped2.split()
        if (not re.search(r'[\d\.,:;!?()\[\]@/]', stripped2)
                and not re.search(r'https?://', stripped2)
                and len(stripped2) < 60):
            if len(words2) <= 2:
                continue
            # Tên viết HOA kiểu báo chí: ĐAN THUẦN, QUỲNH NGUYỄN
            if len(words2) <= 4 and all(w == w.upper() and len(w) > 1 for w in words2):
                continue

        cleaned.append(line)

    result = "\n".join(cleaned)
    result = _strip_after_footer(result)

    # Xoá tiêu đề trùng lặp: nếu 2 heading đầu là substring của nhau → giữ cái ngắn
    _headings = re.findall(r"^#{1,2}\s+(.+)$", result, re.MULTILINE)
    if len(_headings) >= 2:
        t1, t2 = _headings[0].strip(), _headings[1].strip()
        if t1.startswith(t2) or t2.startswith(t1):
            # Xoá heading dài hơn (thường chứa " - Tên trang")
            long_heading = t1 if len(t1) > len(t2) else t2
            result = re.sub(
                r"^#{1,2}\s+" + re.escape(long_heading) + r"\s*$",
                "",
                result,
                count=1,
                flags=re.MULTILINE,
            )
    # Gộp nhiều dòng trống
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


_DOC_TYPE_RE = re.compile(r"^(LUẬT|NGHỊ ĐỊNH|QUYẾT ĐỊNH|THÔNG TƯ|PHÁP LỆNH|HIẾN PHÁP)$", re.IGNORECASE)
_SECTION_RE = re.compile(r"^(CHƯƠNG|Chương|PHẦN|Phần|MỤC|Mục)\s+[IVXLCDM\d]+")
_ARTICLE_RE = re.compile(r"^Điều\s+\d+[a-z]?\s*[\.\:]")
_SKIP_ADMIN_RE = re.compile(r"^(QUỐC HỘI|CỘNG HÒA|Độc lập|Luật số:|Nghị định số:|Quyết định số:|Hà Nội,|TP\.)")


def _convert_doc_win32(path: Path) -> tuple[str, str]:
    """Dùng Word COM automation để đọc file .doc (OLE format) trên Windows."""
    import win32com.client
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    doc = word.Documents.Open(str(path.resolve()))
    paras = [p.Range.Text.strip() for p in doc.Paragraphs]
    doc.Close(False)
    word.Quit()

    # Detect document title: line immediately after "LUẬT"/"NGHỊ ĐỊNH"/etc.
    title = ""
    for idx, text in enumerate(paras):
        if not text:
            continue
        if _DOC_TYPE_RE.match(text):
            for nxt in paras[idx + 1:]:
                if nxt.strip() and not _SKIP_ADMIN_RE.match(nxt):
                    title = nxt.strip()
                    break
            break
    if not title:
        for text in paras:
            if text and text.isupper() and 5 < len(text) < 150 and not _SKIP_ADMIN_RE.match(text):
                title = text
                break
    if not title:
        title = path.stem

    lines = []
    for text in paras:
        if not text or re.match(r"^[-=\s]+$", text):
            lines.append("")
            continue
        if _SKIP_ADMIN_RE.match(text) or _DOC_TYPE_RE.match(text) or text == title:
            continue
        if _SECTION_RE.match(text):
            lines.append(f"\n# {text}")
        elif _ARTICLE_RE.match(text):
            lines.append(f"\n## {text}")
        elif text.isupper() and 4 < len(text) < 150 and not re.match(r"^\d", text):
            lines.append(f"\n# {text}")
        else:
            lines.append(text)

    content = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return title, content


def convert_legal_file(path: Path) -> Optional[Path]:
    try:
        ext = path.suffix.lower()
        source_file = path.name
        converted_at = datetime.now(timezone.utc).isoformat()

        if ext in (".md", ".txt"):
            content = path.read_text(encoding="utf-8")
            lines = content.strip().split("\n")
            title = lines[0].strip().lstrip("#").strip() if lines else path.stem
            if len(title) < 3:
                title = path.stem
            cleaned = re.sub(r"\n{3,}", "\n\n", content.strip())
        elif ext == ".doc":
            # .doc (OLE/Word 97) → markitdown không hỗ trợ, dùng Word COM trên Windows
            try:
                title, cleaned = _convert_doc_win32(path)
            except ImportError:
                print(f"  [WARN] pywin32 not available, skipping {path.name}. Install with: pip install pywin32")
                return None
        else:
            if not HAS_MARKITDOWN:
                print(f"  [WARN] MarkItDown not installed, skipping {path.name}")
                return None
            converter = MarkItDown()
            result = converter.convert(str(path))
            content = result.text_content if hasattr(result, "text_content") else str(result)
            if not content or not content.strip():
                print(f"  [WARN] Empty content in {path.name}")
                return None
            lines = content.strip().split("\n")
            title = lines[0].strip().lstrip("#").strip() if lines else path.stem
            if len(title) < 3:
                title = path.stem
            cleaned = re.sub(r"\n{3,}", "\n\n", content.strip())

        if not cleaned or not cleaned.strip():
            print(f"  [WARN] Empty content in {path.name}")
            return None

        frontmatter = _build_yaml_frontmatter({
            "title": title,
            "source_file": source_file,
            "doc_type": "legal",
            "converted_at": converted_at,
        })

        markdown = f"{frontmatter}\n\n# {title}\n\n{cleaned}\n"
        filename = f"{slugify(title)}.md"
        out_path = STANDARDIZED_LEGAL_DIR / filename
        out_path.write_text(markdown, encoding="utf-8")
        return out_path
    except Exception as e:
        print(f"  [WARN] Cannot convert {path.name}: {type(e).__name__}: {e}")
        return None


def convert_news_file(path: Path) -> Optional[Path]:
    try:
        article = json.loads(path.read_text(encoding="utf-8", errors="replace"))

        title = article.get("title") or article.get("Title") or "Untitled"
        url = article.get("url") or article.get("Url") or ""
        date_crawled = article.get("date_crawled") or article.get("Date") or ""
        content = article.get("content_markdown") or article.get("content") or article.get("Content") or ""
        if not content:
            return None

        cleaned = _clean_news_content(content, title=title)
        if len(cleaned) < 200:
            print(f"  [WARN] Cleaned content too short ({len(cleaned)} chars) for {path.name}")
            return None

        date_str = ""
        if date_crawled:
            try:
                dt = datetime.fromisoformat(date_crawled.replace("Z", "+00:00"))
                date_str = dt.strftime("%Y-%m-%d")
            except Exception:
                date_str = ""

        frontmatter = _build_yaml_frontmatter({
            "title": title,
            "url": url,
            "date": date_str,
            "source": f"{slugify(title)}.md",
            "type": "news",
        })

        markdown = f"{frontmatter}\n\n# {title}\n\n{cleaned}\n"
        filename = f"{slugify(title)}.md"
        out_path = STANDARDIZED_NEWS_DIR / filename
        out_path.write_text(markdown, encoding="utf-8")
        return out_path
    except Exception as e:
        print(f"  [WARN] Cannot convert {path.name}: {type(e).__name__}: {e}")
        return None


def convert_all() -> dict:
    setup_standardized_dirs()

    legal_converted = 0
    news_converted = 0
    failed = []

    legal_files = sorted(
        [p for p in LANDING_LEGAL_DIR.glob("*") if p.is_file() and p.name != ".gitkeep"]
    )
    for path in legal_files:
        expected_out = STANDARDIZED_LEGAL_DIR / f"{slugify(path.stem)}.md"
        if expected_out.exists():
            print(f"  [SKIP] {path.name} -> already exists in standardized")
            continue
        out = convert_legal_file(path)
        if out:
            legal_converted += 1
        else:
            failed.append(str(path))

    news_files = sorted(
        [p for p in LANDING_NEWS_DIR.glob("*.json") if p.is_file() and p.name != ".gitkeep"]
    )
    for path in news_files:
        # Skip if already converted
        expected_out = STANDARDIZED_NEWS_DIR / f"{slugify(path.stem)}.md"
        if expected_out.exists():
            print(f"  [SKIP] {path.name} -> already exists in standardized")
            continue
        out = convert_news_file(path)
        if out:
            news_converted += 1
        else:
            failed.append(str(path))

    return {
        "legal_converted": legal_converted,
        "news_converted": news_converted,
        "failed": failed,
    }
