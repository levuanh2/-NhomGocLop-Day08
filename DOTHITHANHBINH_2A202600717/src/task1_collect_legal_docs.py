"""
Task 1 — Thu thập văn bản pháp luật về ma tuý và các chất cấm.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản pháp luật (PDF/DOCX) từ các nguồn chính thống.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, có năm ban hành.

Gợi ý nguồn:
    - https://thuvienphapluat.vn
    - https://vanban.chinhphu.vn
    - https://luatvietnam.vn

Gợi ý văn bản:
    - Luật Phòng, chống ma tuý 2021 (73/2021/QH15)
    - Nghị định 105/2021/NĐ-CP
    - Bộ luật Hình sự 2015 (sửa đổi 2017) - Chương XX
    - Nghị định 57/2022/NĐ-CP về danh mục chất ma tuý
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")

docs_link = [
    "https://vbpl.vn/van-ban/chi-tiet/luat-phong-chong-ma-tuy-so-73-2021-qh14--152501",
    "https://vbpl.vn/van-ban/chi-tiet/nghi-dinh-so-105-2021-nd-cp-quy-dinh-chi-tiet-va-huong-dan-thi-hanh-mot-so-dieu-cua-luat-phong-chong-ma-tuy--154992",
    "https://vbpl.vn/van-ban/chi-tiet/nghi-dinh-so-57-2022-nd-cp-quy-dinh-cac-danh-muc-chat-ma-tuy-va-tien-chat--156060",
    "https://vbpl.vn/van-ban/chi-tiet/phap-lenh-so-01-2022-ubtvqh15-trinh-tu-thu-tuc-toa-an-nhan-dan-xem-xet-quyet-dinh-viec-dua-nguoi-nghien-ma-tuy-tu-du-12-tuoi-den-duoi-18-tuoi-vao-co-so-cai-nghien-bat-buoc--155969"
    ]


FILENAMES = [
    "luat_phong_chong_ma_tuy_2021",
    "nghi_dinh_105_2021_nd_cp",
    "nghi_dinh_57_2022_nd_cp",
    "phap_lenh_01_2022_ubtvqh15",
]


class LinkParser(HTMLParser):
    """Lấy nhanh các link tài liệu từ HTML."""

    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag not in {"a", "iframe", "embed"}:
            return
        attrs = dict(attrs)
        href = attrs.get("href") or attrs.get("src")
        if href:
            self.links.append(href)


def _extension_from_response(url: str, response: requests.Response) -> str:
    content_type = response.headers.get("content-type", "").lower()
    if "pdf" in content_type:
        return ".pdf"
    if "wordprocessingml" in content_type or "docx" in content_type:
        return ".docx"
    if "msword" in content_type or "doc" in content_type:
        return ".doc"

    path = unquote(urlparse(response.url or url).path).lower()
    suffix = Path(path).suffix
    return suffix if suffix in {".pdf", ".docx", ".doc"} else ".pdf"


def _find_document_links(page_url: str, session: requests.Session) -> list[str]:
    response = session.get(page_url, timeout=20)
    response.raise_for_status()

    parser = LinkParser()
    parser.feed(response.text)

    candidates = []
    for link in parser.links:
        absolute_url = urljoin(page_url, link)
        low = absolute_url.lower()
        if re.search(r"\.(pdf|docx?|rtf)(?:$|[?#])", low) or "download" in low or "file" in low:
            candidates.append(absolute_url)
    return candidates


def download_document(page_url: str, stem: str) -> Path:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; legal-doc-collector/1.0)",
    })

    for filepath in DATA_DIR.glob(f"{stem}.*"):
        if filepath.suffix.lower() in {".pdf", ".docx", ".doc"} and filepath.stat().st_size > 1024:
            print(f"✓ Đã có: {filepath}")
            return filepath

    for url in _find_document_links(page_url, session):
        response = session.get(url, timeout=30)
        if response.status_code != 200 or len(response.content) <= 1024:
            continue

        extension = _extension_from_response(url, response)
        filepath = DATA_DIR / f"{stem}{extension}"
        filepath.write_bytes(response.content)
        print(f"✓ Đã tải: {filepath}")
        return filepath

    raise RuntimeError(f"Không tìm thấy file PDF/DOCX: {page_url}")


def download_all_documents():
    with ThreadPoolExecutor(max_workers=min(4, len(docs_link))) as executor:
        futures = [
            executor.submit(download_document, page_url, stem)
            for page_url, stem in zip(docs_link, FILENAMES)
        ]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    setup_directory()
    download_all_documents()
