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

import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "landing" / "legal"

LEGAL_DOC_SUGGESTIONS = [
    {
        "title": "Luat Phoi chong ma tuy 2021",
        "description": "Luật Phòng, chống ma tuý 2021 (73/2021/QH15)",
        "source": "thuvienphapluat.vn",
        "url": "https://thuvienphapluat.vn/van-ban/Vi-pham-hinh-su/Luat-phong-chong-ma-tuy-2021-425435.aspx",
    },
    {
        "title": "Nghi dinh 105_2021_ND_CP",
        "description": "Nghị định 105/2021/NĐ-CP quy định về công tác cai nghiện ma tuý",
        "source": "vanban.chinhphu.vn",
        "url": "https://vanban.chinhphu.vn/home?cmd=detail&docid=39813",
    },
    {
        "title": "Bo luat Hinh su 2015 chuong XX",
        "description": "Bộ luật Hình sự 2015 (sửa đổi 2017) - Chương XX về tội phạm về ma tuý",
        "source": "thuvienphapluat.vn",
        "url": "https://thuvienphapluat.vn/van-ban/Vi-pham-hinh-su/Bo-luat-hinh-su-2015-2718.aspx",
    },
    {
        "title": "Nghi dinh 57_2022_ND_CP",
        "description": "Nghị định 57/2022/NĐ-CP về danh mục các chất ma tuý",
        "source": "vanban.chinhphu.vn",
        "url": "https://vanban.chinhphu.vn/home?cmd=detail&docid=215155",
    },
]


def setup_directory() -> None:
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Directory ready: {DATA_DIR}")


def check_existing_files() -> None:
    """Liệt kê các file đã có trong thư mục legal."""
    if not DATA_DIR.exists():
        print("[INFO] Legal directory does not exist yet.")
        return

    files = sorted(DATA_DIR.iterdir())
    if not files:
        print("[INFO] No legal documents found in data/landing/legal/")
        print("[INFO] Please download legal documents manually or add them to the directory.")
    else:
        print(f"[INFO] Found {len(files)} file(s) in data/landing/legal/:")
        for f in files:
            print(f"  - {f.name}")


def print_guidance() -> None:
    """In hướng dẫn thu thập tài liệu."""
    print("\n" + "=" * 70)
    print("TASK 1: COLLECT LEGAL DOCUMENTS")
    print(f"Target directory: {DATA_DIR}")
    print("=" * 70)
    print("\nSuggested legal documents to download:")
    for i, doc in enumerate(LEGAL_DOC_SUGGESTIONS, 1):
        print(f"\n  [{i}] {doc['description']}")
        print(f"      Source: {doc['source']}")
        print(f"      URL: {doc['url']}")
        print(f"      Filename: {doc['title']}.pdf")

    print("\n" + "-" * 70)
    print("INSTRUCTIONS:")
    print("  1. Download the documents from the URLs above")
    print("  2. Save them to: data/landing/legal/")
    print("  3. Supported formats: .pdf, .docx, .doc, .txt, .md, .html")
    print("  4. Filenames should be ASCII-safe (no Vietnamese diacritics)")
    print("  5. Task 3 will attempt to convert them to Markdown")
    print("  6. If MarkItDown is not available, PDF/DOCX may be skipped with a warning")
    print("-" * 70)


def main() -> None:
    setup_directory()
    check_existing_files()
    print_guidance()
    print(f"\n[TASK 1 COMPLETE] Legal docs directory ready at: {DATA_DIR}")


if __name__ == "__main__":
    main()
