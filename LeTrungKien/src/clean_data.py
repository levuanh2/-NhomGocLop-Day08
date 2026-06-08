"""
Data cleaning module — tập trung vào các file HTML được crawl bởi Crawl4AI.

Vấn đề chính với crawled HTML (data/standardized/news/):
  - Navigation menu: 50–200 dòng bullet-list link ở đầu trang
  - Share buttons: [](javascript:;) và bullet list icon links
  - Image markdown: [![alt](img)](url) không có giá trị ngữ nghĩa
  - Footer boilerplate: copyright, form comment, social widget
  - Inline link URLs: [Hải Phòng](https://...) → URLs gây nhiễu embedding

Legal docs (data/standardized/legal/) không cần cleaning mạnh —
chỉ normalize whitespace và unicode.

Output: data/cleaned/ (không ghi đè data/standardized/).
task4 sẽ đọc từ data/cleaned/.
"""

import re
import unicodedata
from pathlib import Path

INPUT_DIR  = Path(__file__).parent.parent / "data" / "standardized"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "cleaned"


# ===========================================================================
# Shared
# ===========================================================================

def normalize_unicode(text: str) -> str:
    """NFC normalization + xoá zero-width chars (​, ‌, ﻿, ...)."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'[​‌‍﻿]', '', text)
    return text


def normalize_whitespace(text: str) -> str:
    """\xa0 → space, strip trailing spaces, collapse 3+ blank lines thành 2."""
    text = text.replace('\xa0', ' ')
    text = '\n'.join(line.rstrip() for line in text.split('\n'))
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ===========================================================================
# Legal docs — minimal, chỉ fix control chars từ win32com và normalize
# ===========================================================================

def clean_legal(text: str) -> str:
    """
    Normalize nhẹ cho văn bản pháp luật chuyển đổi từ .doc qua win32com.
    Các ký tự điều khiển: \\x0b (line break), \\x07 (table sep), \\x0c (page break).
    """
    text = text.replace('\x0c', '\n\n').replace('\x0b', '\n').replace('\x07', '\t')
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    return text


# ===========================================================================
# News articles — toàn bộ cleaning tập trung ở đây
# ===========================================================================

# --- 1. Xoá phần navigation / header trước nội dung bài viết ---

def extract_article_body(text: str) -> str:
    """
    Crawl4AI lấy toàn bộ trang HTML nên markdown đầu ra chứa:
      • Navigation menu (header site)
      • Breadcrumb
      • Share buttons
    Rồi mới đến nội dung bài viết thực.

    Cách xác định ranh giới:
      - Metadata block (do task3 thêm) kết thúc bằng dòng '---'
      - Sau '---', tìm heading đầu tiên (# hoặc ##) → bắt đầu bài viết thật
      - Phần trước heading đó đều là navigation noise → bỏ
    """
    sep = '\n---\n'
    if sep not in text:
        return text

    meta_end = text.index(sep) + len(sep)
    metadata = text[:meta_end]          # giữ lại header metadata
    rest = text[meta_end:]

    # Heading đầu tiên sau metadata = tiêu đề bài viết
    heading_match = re.search(r'^#{1,3}\s+\S', rest, re.MULTILINE)
    body = rest[heading_match.start():] if heading_match else rest

    return metadata + body


# --- 2. Xoá footer / boilerplate cuối trang ---

# Pattern khởi đầu một "footer section" — cắt từ đây trở đi
_FOOTER_RE = re.compile(
    r'^(Tổng biên tập:'
    r'|© Copyright'
    r'|Đăng ký email'
    r'|Luôn cập nhật tin tức'
    r'|Thông tin của bạn đọc'
    r'|Vui lòng nhập (?:Email|Họ)'
    r'|Gửi bình luận'
    r'|Mã xác nhận'
    r'|Thông tin bạn đọc'
    r'|Tặng sao'
    r'|\s*Chia sẻ\s*$'
    r'|\s*CHIA SẺ\s*$'
    r'|\s*Đọc tiếp\s'
    r'|\s*\*\s*x\d'             # bullet star rating: "  * x1", "  * x5"
    r')',
    re.MULTILINE,
)

def remove_footer(text: str) -> str:
    """Cắt bỏ footer boilerplate từ pattern đầu tiên match được."""
    # Chỉ áp dụng cho phần sau metadata
    sep = '\n---\n'
    if sep not in text:
        m = _FOOTER_RE.search(text)
        return text[:m.start()] if m else text

    meta_end = text.index(sep) + len(sep)
    metadata = text[:meta_end]
    body = text[meta_end:]

    m = _FOOTER_RE.search(body)
    return metadata + (body[:m.start()] if m else body)


# --- 3. Xoá image markdown ---

def remove_images(text: str) -> str:
    """
    Xoá image markdown vì không có giá trị ngữ nghĩa cho RAG:
      [![alt](img-url)](link)  →  (xoá hoàn toàn)
      ![alt](img-url)          →  (xoá hoàn toàn)
    """
    text = re.sub(r'\[!\[[^\]]*\]\([^\)]*\)\]\([^\)]*\)', '', text)
    text = re.sub(r'!\[[^\]]*\]\([^\)]*\)', '', text)
    return text


# --- 4. Xoá block navigation bullets ---

_NAV_LINE_RE = re.compile(r'^\s{0,4}\*\s*\[([^\]]*)\]\([^\)]+\)\s*$')

def remove_nav_blocks(text: str) -> str:
    """
    Xoá block gồm 5+ dòng liên tiếp toàn là bullet-list markdown link —
    dấu hiệu điển hình của menu navigation còn sót sau extract_article_body.
    Giữ lại link lẻ trong nội dung (trích dẫn, tên địa danh, v.v.).
    """
    lines = text.split('\n')
    result: list[str] = []
    i = 0
    while i < len(lines):
        j = i
        while j < len(lines) and _NAV_LINE_RE.match(lines[j]):
            j += 1
        if j - i >= 5:
            i = j       # bỏ cả block nav
        else:
            result.append(lines[i])
            i += 1
    return '\n'.join(result)


# --- 5. Xoá các dòng chỉ chứa javascript: links hoặc ký tự rác ---

_JUNK_LINE_RE = re.compile(
    r'^\s*'
    r'(\[.*?\]\(javascript:[^\)]*\)'   # [text](javascript:;)
    r'|\[\]\([^\)]*\)'                  # [](url) — link không có text
    r'|×'                               # ký tự × từ modal close button
    r')\s*$'
)

def remove_junk_lines(text: str) -> str:
    """Xoá dòng chỉ chứa javascript: links, empty links, hoặc modal chars."""
    lines = text.split('\n')
    return '\n'.join(line for line in lines if not _JUNK_LINE_RE.match(line))


# --- 6. Chuyển inline link thành plain text ---

def strip_link_urls(text: str) -> str:
    """
    [Hải Phòng](https://vnexpress.net/...) → Hải Phòng
    Giữ display text, bỏ URL để embedding model không bị nhiễu.
    """
    return re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)


# --- Pipeline tổng ---

def clean_news(text: str) -> str:
    """Pipeline đầy đủ cho một file news markdown."""
    text = extract_article_body(text)   # cắt nav ở đầu
    text = remove_footer(text)          # cắt footer ở cuối
    text = remove_images(text)          # xoá ảnh
    text = remove_nav_blocks(text)      # xoá nav bullets còn sót
    text = remove_junk_lines(text)      # xoá dòng javascript:/modal
    text = strip_link_urls(text)        # [text](url) → text
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    return text


# ===========================================================================
# Main
# ===========================================================================

def clean_all():
    print("=" * 55)
    print("Data Cleaning")
    print("=" * 55)

    for subdir, clean_fn, label in [
        ("legal", clean_legal,  "Legal docs   (normalize only)"),
        ("news",  clean_news,   "News articles (full HTML cleaning)"),
    ]:
        input_dir  = INPUT_DIR  / subdir
        output_dir = OUTPUT_DIR / subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(input_dir.glob("*.md"))
        print(f"\n--- {label} ({len(files)} files) ---")
        for fp in files:
            raw     = fp.read_text(encoding="utf-8")
            cleaned = clean_fn(raw)
            (output_dir / fp.name).write_text(cleaned, encoding="utf-8")
            pct = (1 - len(cleaned) / max(len(raw), 1)) * 100
            print(f"  {fp.name[:52]:<52} {len(raw):>7,} → {len(cleaned):>6,} chars  (-{pct:.0f}%)")

    print(f"\n✓ Output → {OUTPUT_DIR}")


if __name__ == "__main__":
    clean_all()
