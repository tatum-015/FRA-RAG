import logging
import chromadb
from typing import List, Optional
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid

from fra_rag.config import (
    CHUNK_SIZE, 
    CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME, 
    CHROMA_SETTINGS
)

# Configure logging
logger = logging.getLogger(__name__)

def initialize_chroma(collection_name: str = CHROMA_SETTINGS["collection_name"]) -> chromadb.Collection:
    """
    Initialize ChromaDB client and collection. Clears existing collection if it exists.
    
    Args:
        collection_name (str): Name of the collection to create/use.
        
    Returns:
        chromadb.Collection: Initialized ChromaDB collection.
    """
    logger.info(f"Initializing ChromaDB collection: {collection_name}")
    client = chromadb.PersistentClient(path=CHROMA_SETTINGS["persist_directory"])
    
    # Delete existing collection if it exists
    existing_collections = client.list_collections()
    if any(col.name == collection_name for col in existing_collections):
        logger.info(f"Deleting existing collection: {collection_name}")
        client.delete_collection(collection_name)
    
    # Create a new collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "Fire Risk Assessment Documents"}
    )
    logger.info(f"Created new ChromaDB collection: {collection_name}")
    
    return collection

def split_text(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for efficient embedding and storage.
    
    Args:
        documents (List[Document]): List of Document objects to be split.
        
    Returns:
        List[Document]: List of split Document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    all_chunks = []
    for doc in documents:
        split_texts = text_splitter.split_text(doc.page_content)
        for idx, chunk in enumerate(split_texts):
            new_doc = Document(
                page_content=chunk,
                metadata={
                    **doc.metadata,
                    "chunk_id": f"{doc.metadata.get('document_id', 'doc')}_{idx}"
                }
            )
            all_chunks.append(new_doc)
    logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
    return all_chunks

def index_documents(collection: chromadb.Collection, chunks: List[Document]) -> None:
    """
    Index document chunks into ChromaDB with embedded vectors.
    
    Args:
        collection (chromadb.Collection): The ChromaDB collection to add documents to.
        chunks (List[Document]): List of Document chunks to index.
    """
    logger.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    
    batch_size = 100
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    logger.info(f"Total batches to process: {total_batches}")
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = i // batch_size + 1
        logger.info(f"Processing batch {batch_num} of {total_batches}")
        
        try:
            # Prepare batch data
            documents = [chunk.page_content for chunk in batch]
            metadatas = [{
                "source": chunk.metadata.get("source", ""),
                "file_path": chunk.metadata.get("file_path", ""),
                "document_type": chunk.metadata.get("document_type", "unknown"),
                "section_type": chunk.metadata.get("section_type", ""),
                "parent_section": chunk.metadata.get("parent_section", ""),
                "question_id": chunk.metadata.get("question_id", ""),
                "action_id": chunk.metadata.get("action_id", ""),
                "content_type": chunk.metadata.get("content_type", ""),
                "priority": chunk.metadata.get("priority", ""),
                "due_date": chunk.metadata.get("due_date", ""),
                "client_status": chunk.metadata.get("client_status", ""),
                "chunk_id": chunk.metadata.get("chunk_id", "")
            } for chunk in batch]
            
            embeddings = embedding_model.embed_documents(documents)
            ids = [f"doc_{uuid.uuid4()}" for _ in range(len(batch))]  # Generate unique IDs
            
            # Add batch to collection
            collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Successfully added batch {batch_num} to ChromaDB.")
        
        except Exception as e:
            logger.error(f"Error processing batch {batch_num}: {str(e)}")
            continue  # Continue with next batches instead of halting
    
    logger.info(f"Successfully indexed {len(chunks)} chunks into ChromaDB.") 