from typing import List, Dict, Any
import logging
import json
import re
from pathlib import Path
from pprint import pformat 
from datetime import datetime
import os
import nltk
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document  # Added explicit Document import
from langchain_unstructured import UnstructuredLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
import glob

from fra_rag.config import (
    CHUNK_SIZE, 
    CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
    EMBEDDING_MODEL_NAME, 
    CHROMA_SETTINGS,
    PROCESSED_DATA_DIR  # Added from config
)

logger = logging.getLogger(__name__)

# Ensure NLTK 'punkt' tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK 'punkt' tokenizer...")
    nltk.download('punkt')

def clean_text(text: str) -> str:
    """Basic text cleaning for FRA documents."""
    # Remove special characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Standardize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove standalone special characters
    text = re.sub(r'^\W+$', '', text)
    return text.strip()

def preprocess_document(doc: Document) -> Document:
    if doc.metadata.get('category') == 'Table':
        # Extract just the text content from tables
        doc.page_content = clean_text(doc.page_content)
    return doc


def load_documents(file_paths: List[str]) -> List[Document]:
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
                    "detection_settings": {  # Added better detection settings
                        "ocr_on": True,
                        "text_formatting": True
                    }
                }
            )
            loaded_docs = loader.load()
            if not loaded_docs:
                logger.warning(f"No documents loaded from {file_path}")
                continue
            
            # Add metadata and preprocess in one step
            loaded_docs = [
                preprocess_document(
                    Document(
                        page_content=doc.page_content,
                        metadata={
                            **doc.metadata,
                            "file_path": file_path,
                            "document_type": "FRA"
                        }
                    )
                ) for doc in loaded_docs
            ]
                
            documents.extend(loaded_docs)
                
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise

    # Save to JSON for Inspection
    output_path = Path(PROCESSED_DATA_DIR) / "extracted_documents.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json_docs = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            } for doc in documents
        ]
        json.dump(json_docs, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved extracted documents to {output_path}")
    
    return documents

def is_valid_chunk(chunk: Document) -> bool:
    # Filter out chunks that are too small or contain no meaningful content
    min_length = 50
    return (
        len(chunk.page_content.strip()) >= min_length and
        not chunk.page_content.isspace() and
        not re.match(r'^[\W\d]+$', chunk.page_content)
    )

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.
    
    Args:
        documents: List of documents
        
    Returns:
        List of document chunks
    """
    logger.info(f"Splitting documents with chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            # Added FRA-specific separators
            "\nSection ", 
            "\nQuestion ",
            "\nAction ID: ",
            # Standard separators
            "\n\n", 
            "\n", 
            ". ", 
            " ", 
            ""
        ],
        keep_separator=True  # Keep separators to maintain context
    )
    chunks = text_splitter.split_documents(documents)

    # Filter invalid chunks
    chunks = [chunk for chunk in chunks if is_valid_chunk(chunk)]
    logger.info(f"Created {len(chunks)} chunks")
    
    # Log sample chunks for inspection
    if chunks:
        logger.debug("Sample first chunk:")
        logger.debug(f"Content: {chunks[0].page_content[:200]}...")
        logger.debug(f"Metadata: {chunks[0].metadata}")
    
    return chunks

# Save chunks to file for inspection
def view_chunks(chunks, output_dir="chunk_outputs"):
    """
    View and save document chunks with both terminal preview and file output.
    
    Args:
        chunks: List of document chunks
        output_dir: Directory to save output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chunks_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)
    
    # Write to file and show preview
    with open(filepath, 'w') as f:
        # Write summary header
        f.write(f"Total Chunks: {len(chunks)}\n")
        f.write("=" * 80 + "\n\n")
        
        # Preview first 3 chunks in terminal
        print(f"\nPreviewing first 3 of {len(chunks)} chunks:")
        print("=" * 80)
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            # Prepare chunk info using pprint
            chunk_info = {
                'chunk_number': i + 1,
                'content_preview': chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content,
                'content_length': len(chunk.page_content),
                'metadata': chunk.metadata
            }
            
            # Write full chunk info to file
            f.write(f"Chunk {i + 1}:\n")
            f.write(pformat(chunk_info))
            f.write("\n\n" + "=" * 80 + "\n\n")
            
            # Show preview in terminal for first 3 chunks
            if i < 3:
                print(f"\nChunk {i + 1}:")
                print(pformat(chunk_info))
                print("=" * 80)
    
    print(f"\nFull chunk details saved to: {filepath}")
    return filepath

def initialize_chroma(collection_name: str = CHROMA_SETTINGS["collection_name"]) -> chromadb.Collection:
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

def index_documents(collection: chromadb.Collection, chunks: List[Document]) -> None:
    """
    Index document chunks in ChromaDB.
    
    Args:
        collection: ChromaDB collection
        chunks: List of document chunks to index
    """
    logger.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    
    # Add documents to collection in batches
    batch_size = 100
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = i // batch_size + 1
        logger.info(f"Processing batch {batch_num} of {total_batches}")
        
        try:
            # Prepare batch data
            documents = [chunk.page_content for chunk in batch]
            embeddings = embedding_model.embed_documents(documents)
            metadatas = [{
                "source": chunk.metadata.get("source", "unknown"),
                "file_path": chunk.metadata.get("file_path", "unknown"),
                "document_type": chunk.metadata.get("document_type", "unknown"),
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
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_num}: {str(e)}")
            raise
    
    logger.info(f"Successfully indexed {len(chunks)} chunks")

if __name__ == "__main__":
    # Get all document paths in the data/raw directory
    file_paths = glob.glob("data/raw/*")
    
    # Load documents
    documents = load_documents(file_paths)
    
    # Split documents into chunks
    chunks = split_documents(documents)
    
    # View and save chunks
    view_chunks(chunks)
    
    # Initialize ChromaDB collection
    collection = initialize_chroma()
    
    # Index document chunks
    index_documents(collection, chunks)