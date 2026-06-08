import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
LANDING_DIR = DATA_DIR / "landing"
LANDING_LEGAL_DIR = LANDING_DIR / "legal"
LANDING_NEWS_DIR = LANDING_DIR / "news"
STANDARDIZED_DIR = DATA_DIR / "standardized"
STANDARDIZED_LEGAL_DIR = STANDARDIZED_DIR / "legal"
STANDARDIZED_NEWS_DIR = STANDARDIZED_DIR / "news"
CHROMA_DIR = DATA_DIR / "vectorstore" / "chroma"

COLLECTION_NAME = "legal_chunks"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 128
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 1536
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
