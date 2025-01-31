from typing import List, Dict, Any
import logging
import json

import nltk
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

from fra_rag.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME, CHROMA_SETTINGS

logger = logging.getLogger(__name__)

# Ensure NLTK 'punkt' tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK 'punkt' tokenizer...")
    nltk.download('punkt')

def load_documents(file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Load documents using UnstructuredFileLoader with PDF support.
    
    Args:
        file_paths: List of paths to FRA documents
        
    Returns:
        List of processed documents
    """
    documents = []
    for file_path in file_paths:
        logger.info(f"Loading document: {file_path}")
        try:
            loader = UnstructuredLoader(
                file_path,
                mode="elements",
                strategy="hi_res",
                include_page_breaks=True,
                partition_via_api=True,
                **{
                    # Custom detection settings
                    "detect_titles": True,
                    "detect_lists": True,
                    "detect_tables": True,
                }
            )
            loaded_docs = loader.load()
            if not loaded_docs:
                logger.warning(f"No documents loaded from {file_path}")
            documents.extend(loaded_docs)
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise

    # **Save to JSON for Inspection**
    with open("data/processed/extracted_documents.json", "w", encoding="utf-8") as f:
        json_docs = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            } for doc in documents
        ]
        json.dump(json_docs, f, ensure_ascii=False, indent=4)
        logger.info("Saved extracted documents to data/processed/extracted_documents.json")
    
    return documents

def split_documents(documents: List[Dict[str, Any]], 
                   chunk_size: int = CHUNK_SIZE, 
                   chunk_overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.
    
    Args:
        documents: List of documents
        chunk_size: Maximum chunk size (default: from config)
        chunk_overlap: Overlap between chunks (default: from config)
        
    Returns:
        List of document chunks
    """
    logger.info(f"Splitting documents with chunk_size={chunk_size}, overlap={chunk_overlap}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")
    print(chunks)
    return chunks

def initialize_chroma(collection_name: str) -> chromadb.Collection:
    """
    Initialize ChromaDB client and collection.
    
    Args:
        collection_name: Name of the collection to create/use
        
    Returns:
        ChromaDB collection
    """
    logger.info(f"Initializing ChromaDB collection: {collection_name}")
    client = chromadb.PersistentClient(path=CHROMA_SETTINGS["persist_directory"])
    
    # Delete collection if it already exists
    try:
        client.delete_collection(collection_name)
        logger.info(f"Deleted existing collection: {collection_name}")
    except ValueError:
        pass
        
    return client.create_collection(
        name=collection_name,
        metadata={"description": "Fire Risk Assessment Documents"}
    )

def index_documents(collection: chromadb.Collection, chunks: List[Dict[str, Any]]) -> None:
    """
    Index document chunks in ChromaDB.
    
    Args:
        collection: ChromaDB collection
        chunks: List of document chunks to index
    """
    logger.info("Initializing embedding model")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    
    # Add documents to collection in batches
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1} of {len(chunks)//batch_size + 1}")
        
        # Prepare batch data
        documents = [chunk.page_content for chunk in batch]
        embeddings = embedding_model.embed_documents(documents)
        metadatas = [{
            "source": chunk.metadata.get("source", "unknown"),
            "chunk_id": str(i + idx)
        } for idx, chunk in enumerate(batch)]
        ids = [f"doc_{i + idx}" for idx in range(len(batch))]
        
        # Add batch to collection
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    logger.info(f"Completed indexing {len(chunks)} chunks")