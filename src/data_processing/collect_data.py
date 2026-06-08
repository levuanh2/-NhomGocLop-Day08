from pathlib import Path
from data_processing.config import LANDING_LEGAL_DIR, LANDING_NEWS_DIR


def setup_landing_dirs() -> None:
    LANDING_LEGAL_DIR.mkdir(parents=True, exist_ok=True)
    LANDING_NEWS_DIR.mkdir(parents=True, exist_ok=True)


def list_landing_files() -> dict:
    legal_files = sorted(
        [str(p) for p in LANDING_LEGAL_DIR.glob("*") if p.is_file()]
    )
    news_files = sorted(
        [str(p) for p in LANDING_NEWS_DIR.glob("*") if p.is_file()]
    )
    return {
        "legal_files": legal_files,
        "news_files": news_files,
    }
