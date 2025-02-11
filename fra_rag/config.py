from pathlib import Path
from typing import Dict, Any
import os

# Project directory paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# ChromaDB settings
CHROMA_SETTINGS: Dict[str, Any] = {
    "collection_name": "fra_documents",
    "persist_directory": str(PROCESSED_DATA_DIR / "chroma"),
    "anonymized_telemetry": False
}

# Document processing settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Create necessary directories
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
