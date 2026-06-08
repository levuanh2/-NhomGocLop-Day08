import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))

try:
    from data_processing.crawl_more_news import crawl_all_news
except ImportError:
    crawl_all_news = None

from data_processing.collect_data import setup_landing_dirs, list_landing_files
from data_processing.convert_markdown import convert_all
from data_processing.chunk_and_index import build_chroma_index
from data_processing.verify_output import verify_tv1_output
from data_processing.config import (
    ROOT_DIR,
    DATA_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="TV1 Data Processing Pipeline")
    parser.add_argument(
        "--crawl-more-news",
        action="store_true",
        help="Crawl additional 20 news articles before running the pipeline",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("TV1 DATA PROCESSING PIPELINE")
    print("=" * 60)

    if args.crawl_more_news and crawl_all_news:
        print("\nCrawling additional news articles...")
        crawl_all_news(limit=20)

    setup_landing_dirs()
    file_summary = list_landing_files()
    print(f"Legal files: {len(file_summary['legal_files'])}")
    print(f"News files: {len(file_summary['news_files'])}")

    print("\nConverting landing data to standardized markdown...")
    convert_summary = convert_all()
    print(
        f"Converted: legal={convert_summary['legal_converted']}, "
        f"news={convert_summary['news_converted']}"
    )
    if convert_summary["failed"]:
        print(f"Failed files: {convert_summary['failed']}")

    print("\nBuilding Chroma index...")
    index_summary = build_chroma_index(reset=True)

    print("\nVerifying TV1 output...")
    verify_summary = verify_tv1_output()

    # Write DATA_REPORT.md
    report_path = DATA_DIR / "DATA_REPORT.md"
    lines = [
        "# TV1 Data Report",
        "",
        "## Summary",
        f"- Legal markdown files: {convert_summary['legal_converted']}",
        f"- News markdown files: {convert_summary['news_converted']}",
        f"- ChromaDB path: {CHROMA_DIR}",
        f"- Collection name: {COLLECTION_NAME}",
        f"- Total chunks indexed: {index_summary['chunks_indexed']}",
        f"- Embedding model: {EMBEDDING_MODEL}",
        f"- Embedding provider: OpenAI",
        f"- Embedding dimension: {EMBEDDING_DIM}",
        f"- Vector DB: ChromaDB",
        f"- Distance metric: cosine",
        "",
        "## Metadata Schema",
        "source, type, chunk_index, url, title, date, article_id",
        "",
        "## Sample Query",
        "Query: 'phạt tù'",
    ]
    for meta in verify_summary.get("sample_metadata", []):
        lines.append(str(meta))
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written to {report_path}")
    print("TV1 pipeline completed.")


if __name__ == "__main__":
    main()
