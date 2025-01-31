import logging
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
import os

def setup_logging(log_level: str = "INFO") -> None:
    """Set up basic logging configuration.
    
    Args:
        log_level: Logging level (default: "INFO")
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def get_document_paths(directory: Path, extensions: Optional[List[str]] = None) -> List[Path]:
    """Get all document paths in a directory with specified extensions.
    
    Args:
        directory: Directory to search for documents
        extensions: List of file extensions to include (default: [".pdf"])
        
    Returns:
        List of Path objects for matching documents
    """
    if extensions is None:
        extensions = [".pdf"]
        
    paths = []
    for ext in extensions:
        paths.extend(directory.glob(f"*{ext}"))
    return sorted(paths)

def load_environment():
    """Load environment variables from the .env file."""
    load_dotenv()
    if not os.getenv("UNSTRUCTURED_API_KEY"):
        logging.warning("UNSTRUCTURED_API_KEY is not set. Some features may not work as expected.")
