# main.py
import logging
from pathlib import Path

from fra_rag.config import RAW_DATA_DIR, CHROMA_SETTINGS
from fra_rag.utils import setup_logging, get_document_paths, load_environment
from fra_rag.ingest import load_documents, split_documents, initialize_chroma, index_documents

def main():
    # Load environment variables
    load_environment()
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Get document paths
        doc_paths = get_document_paths(RAW_DATA_DIR)
        if not doc_paths:
            logger.warning(f"No documents found in {RAW_DATA_DIR}")
            return
        
        logger.info(f"Found {len(doc_paths)} documents to process")
        
        # Load and process documents
        docs = load_documents([str(p) for p in doc_paths])
        logger.info(f"Loaded {len(docs)} documents")
        
        # Split into chunks
        chunks = split_documents(docs)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Initialize and index in ChromaDB
        collection = initialize_chroma(CHROMA_SETTINGS["collection_name"])
        index_documents(collection, chunks)
        logger.info("Document indexing complete")
        
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()