"""
Task 1 — Thu thập văn bản pháp luật về ma tuý và các chất cấm.

Các file đã tải thủ công vào data/landing/legal/:
    - Luật phồng chống ma túy.doc
    - BAN HÀNH CHƯƠNG TRÌNH PHÒNG, CHỐNG MA TÚY TRONG THANH, THIẾU NIÊN.doc
    - QUY ĐỊNH CÁC DANH MỤC CHẤT MA TÚY VÀ TIỀN CHẤT.doc
    - QUY ĐỊNH TIÊU CHUẨN CHẨN ĐOÁN VÀ QUY TRÌNH CHUYÊN MÔN ĐỂ XÁC ĐỊNH TÌNH TRẠNG NGHIỆN MA TÚY.doc
    - XỬ PHẠT VI PHẠM HÀNH CHÍNH.doc
"""

from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc"}
MIN_FILE_SIZE_BYTES = 1024  # 1KB


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def verify_documents() -> list[Path]:
    """
    Kiểm tra các văn bản pháp luật đã có trong data/landing/legal/.

    Returns:
        Danh sách các file hợp lệ (đúng định dạng và đủ kích thước).
    """
    valid_files = []

    for filepath in sorted(DATA_DIR.iterdir()):
        if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        size = filepath.stat().st_size
        if size >= MIN_FILE_SIZE_BYTES:
            valid_files.append(filepath)
            print(f"  ✓ {filepath.name}  ({size / 1024:.1f} KB)")
        else:
            print(f"  ✗ {filepath.name}  ({size} bytes) — quá nhỏ, bỏ qua")

    return valid_files


if __name__ == "__main__":
    setup_directory()
    print("\n--- Kiểm tra văn bản pháp luật ---")
    files = verify_documents()
    print(f"\nTổng: {len(files)} văn bản hợp lệ (yêu cầu tối thiểu 3)")
    if len(files) < 3:
        print("⚠ Chưa đủ — hãy tải thêm file PDF/DOCX vào data/landing/legal/")
    else:
        print("✓ Đã đủ điều kiện Task 1")
