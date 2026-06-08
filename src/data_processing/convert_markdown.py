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
    "Chia sẻ [",
    "Gửi email cho tác giả",
    "Về trang chủ",
    "Hotline:",
    "Tải ứng dụng",
    "Điều khoản sử dụng",
    "Tòa soạn",
    "Đăng ký nhận thông báo",
]


def _strip_before_article_title(content: str, title: str) -> str:
    if not title:
        return content
    lines = content.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped in (f"# {title}", f"## {title}"):
            return "\n".join(lines[i:])
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("# ") or stripped.startswith("## "):
            if len(stripped) > 20:
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
    if cut_index < len(lines):
        lines = lines[:cut_index]
    return "\n".join(lines)


NOISE_PATTERNS = [
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
    r"^\[?\s*Chia\s+sẻ\s+bài\s+viết\s*\]?\(?.*?\)?\s*$",
    r"^\[?\s*Chia\s+sẻ\s+.*?\s*\]?\(?.*?\)?\s*$",
    r"^\[?\s*Bình\s+luận\s*\]?\(?.*?\)?\s*$",
    r"^\[?\s*In\s*\]?\(?.*?\)?\s*$",
    r"^\[?\s*Copy\s+link.*?\s*\]?\(?.*?\)?\s*$",
    r"^\[?\s*Gửi\s+email.*?\s*\]?\(?.*?\)?\s*$",
    r"^\[?\s*Mail\s*\]?\(?.*?\)?\s*$",
    r"^\[?\s*Trở\s+lại\s+.*?\s*\]?\(?.*?\)?\s*$",
    r"^Về\s+trang\s+chủ\s*$",
    r"^Lên\s+đầu\s+trang\s*$",
    r"^Liên\s+hệ\s+quảng\s+cáo\s*$",
    r"^Hotline\s*:?\s*$",
    r"^Copyright.*?$",
    r"^Điều\s+khoản\s*sử\s+dụng.*?$",
    r"^\[?\s*Thứ\s+\w+,\s*\d+/\d+/\d+\s*\]?\(?.*?\)?\s*$",
    r"^(TP\s+HCM|Hà\s+Nội|Đà\s+Nẵng|Hải\s+Phòng)\s*$",
    r"^\*.*?\[.*?\]\(https?://vnexpress\.net/(topic|chu-de).*?\)\s*$",
    r"^\s*\*\s*\[.*?\]\(https?://vnexpress\.net/(topic|chu-de).*?\)\s*$",
]
COMPILED_NOISE = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]


def _clean_news_content(content: str, title: Optional[str] = None) -> str:
    if title:
        content = _strip_before_article_title(content, title)
    lines = content.split("\n")
    cleaned = []
    skip_next_empty = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if skip_next_empty:
                continue
            skip_next_empty = True
            cleaned.append("")
            continue
        skip_next_empty = False
        if re.match(r"^\[\]\(javascript:", stripped, re.IGNORECASE):
            continue
        if re.match(r"^\[\s*\]\(javascript:", stripped, re.IGNORECASE):
            continue
        if re.match(r"^!\[\s*\]\(https?://.*(logo|menu|graphics).*\)$", stripped, re.IGNORECASE):
            continue
        if re.match(r"^\[\s*\]\(https?://.*(logo|menu).*\)$", stripped, re.IGNORECASE):
            continue
        is_noise = False
        for pattern in COMPILED_NOISE:
            if pattern.match(stripped):
                is_noise = True
                break
        if is_noise:
            continue
        cleaned.append(line)
    result = "\n".join(cleaned)
    result = _strip_after_footer(result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = re.sub(r"(# .+)\n+# \1", r"\1", result, flags=re.IGNORECASE)
    result = re.sub(r"(# .+)\n+## \1", r"\1", result, flags=re.IGNORECASE)
    return result.strip()


def convert_legal_file(path: Path) -> Optional[Path]:
    try:
        ext = path.suffix.lower()
        source_file = path.name
        converted_at = datetime.now(timezone.utc).isoformat()

        if ext in (".md", ".txt"):
            content = path.read_text(encoding="utf-8")
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
