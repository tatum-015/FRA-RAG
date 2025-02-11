#ingest.py
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
    EMBEDDING_MODEL_NAME, 
    CHROMA_SETTINGS,
    PROCESSED_DATA_DIR  # Added from config
)

logger = logging.getLogger(__name__)

class DocumentContext:
    """
    Context tracker for document processing.
    """
    def __init__(self):
        self.current_section = None
        self.current_title = None
        self.action_references = {}
        
    def update_context(self, doc: Document):
        """
        Update context based on document metadata.
        
        Args:
            doc: Document object
        """
        if doc.metadata.get('category') == 'Title':
            self.current_title = doc.page_content
            # Reset section when new title found
            self.current_section = None
        elif doc.metadata.get('category') == 'Section':
            self.current_section = doc.page_content
        elif 'Action ID:' in doc.page_content:
            action_id = extract_action_id(doc.page_content)
            question_id = extract_question_id(doc.page_content)
            self.action_references[action_id] = question_id
            
    def get_current_context(self) -> Dict:
        """
        Get current context for metadata processing.
        
        Returns:
            Dict: Current context
        """
        return {
            'current_section': self.current_section,
            'current_title': self.current_title,
            'action_references': self.action_references
        }

# Ensure NLTK 'punkt' tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK 'punkt' tokenizer...")
    nltk.download('punkt')

#1st utility function
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

#2nd utility function
def extract_question_id(text: str) -> str:
    """Extract question ID from text."""
    match = re.search(r'Question[:\s]+([A-Z]\.[0-9]+)', text)
    return match.group(1) if match else ''

#3rd utility function
def extract_field(text: str, pattern: str) -> str:
    """Extract field value using regex pattern."""
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ''

#4th utility function
def convert_table_to_text(html_content: str) -> str:
    """Convert HTML table to structured text."""
    # Simple HTML to text conversion - could be enhanced with BeautifulSoup
    text = re.sub(r'<tr[^>]*>', '\n', html_content)
    text = re.sub(r'<td[^>]*>', ' ', text)
    text = re.sub(r'</td>|</tr>|</tbody>|</thead>|</table>', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    return clean_text(text)

#5th utility function
def extract_priority(text: str) -> str:
    """Extract priority from details section."""
    match = re.search(r'Priority\s+([^\n]+)', text)
    return match.group(1) if match else ''

#6th utility function
def extract_action_id(text: str) -> str:
    """Extract action ID from details section."""
    match = re.search(r'Action ID\s+(\d+)', text)
    return match.group(1) if match else ''

#7th utility function
def is_valid_chunk(chunk: Document, min_length: int = 50) -> bool:
    """Enhanced chunk validation with context awareness."""
    # Don't split action items
    if chunk.metadata.get('content_type') == 'action_item':
        return True
        
    return (
        len(chunk.page_content.strip()) >= min_length and
        not chunk.page_content.isspace() and
        not re.match(r'^[\W\d]+$', chunk.page_content)
    )

#metadata processing function 
def process_metadata(doc: Document, context: Dict) -> Dict:
    """Enhance metadata with context and relationships."""
    metadata = {
        **doc.metadata,
        'section_type': context.get('current_section'),
        'parent_section': context.get('current_title'),
        'question_id': doc.metadata.get('question_id'),
        'action_id': doc.metadata.get('action_id'),
        'has_action': bool(doc.metadata.get('action_id'))
    }
    # Add related question if this is an action
    if doc.metadata.get('action_id'):
        metadata['related_question'] = context.get('action_references', {}).get(
            doc.metadata['action_id']
        )
    
    return metadata

#sanize metadata to ensure all values are strings
def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = {}
    for key, value in metadata.items():
        if value is None:
            sanitized[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        else:
            sanitized[key] = str(value)  # Convert other types to string
    return sanitized

# Core document processing functions
#1.preprocess document with special handling for FRA recommendation tables
def preprocess_document(doc: Document) -> List[Document]:
    if doc.metadata.get('category') == 'Table':
        content = doc.page_content
        findings = []
        
        # Extract content from HTML if present
        if '<table' in content:
            # Convert HTML table to structured text
            content = convert_table_to_text(content)
            
        # Split on new questions
        question_blocks = re.split(r'(?=Question[:\s]+[A-Z]\.[0-9]+)', content)
        
        for block in question_blocks:
            if not block.strip():
                continue

            # Enhanced extraction with more fields
            finding = {
                'question_id': extract_field(block, r'Question[:\s]+([A-Z]\.[0-9]+)'),
                'section': extract_field(block, r'Section[:\s]+(.+?)(?=\n|$)'),
                'action_id': extract_field(block, r'Action ID[:\s]+(\d+)'),
                'priority': extract_field(block, r'Priority[:\s]+([^\n]+)'),
                'quantity_known': extract_field(block, r'Known[:\s]+([^\n]+)'),
                'quantity_potential': extract_field(block, r'Potential[:\s]+([^\n]+)'),
                'due_date': extract_field(block, r'Due Date:[:\s]+([^\n]+)'),
                'client_status': extract_field(block, r'Client Status:[:\s]+([^\n]+)')
            }
            
            # Extract comment and recommendation with improved pattern matching
            comment_match = re.search(r'Comment:[:\s]+(.+?)(?=Recommendation:|$)', block, re.DOTALL)
            recommendation_match = re.search(r'Recommendation:[:\s]+(.+?)(?=Due Date:|$)', block, re.DOTALL)
            
            finding['comment'] = comment_match.group(1).strip() if comment_match else ''
            finding['recommendation'] = recommendation_match.group(1).strip() if recommendation_match else ''
            
            # Enhanced structured text format
            structured_text = f"""
Question ID: {finding['question_id']}
Section: {finding['section']}
Action ID: {finding['action_id']}
Priority: {finding['priority']}
Quantity Known: {finding['quantity_known']}
Quantity Potential: {finding['quantity_potential']}
Comment: {finding['comment']}
Recommendation: {finding['recommendation']}
Due Date: {finding['due_date']}
Client Status: {finding['client_status']}
            """.strip()
            
            # Create new document with enhanced metadata
            new_doc = Document(
                page_content=clean_text(structured_text),
                metadata={
                    **doc.metadata,
                    'content_type': 'action_item',
                    'question_id': finding['question_id'],
                    'section': finding['section'],
                    'action_id': finding['action_id'],
                    'priority': finding['priority'],
                    'due_date': finding['due_date'],
                    'client_status': finding['client_status'],
                    'has_recommendation': bool(finding['recommendation']),
                    'has_quantities': bool(finding['quantity_known'] or finding['quantity_potential'])
                }
            )
            findings.append(new_doc)
            
        return findings
    
    # Handle non-table documents
    elif doc.metadata.get('category') == 'Title':
        # Add section type detection
        if 'Action Plan' in doc.page_content:
            doc.metadata['section_type'] = 'action_plan'
        elif 'Common Area Fire Doors' in doc.page_content:
            doc.metadata['section_type'] = 'fire_doors_section'
    
    return [doc]  # Return original document if not a table
#2. split documents into chunks
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

#Main document loading function
def load_documents(file_paths: List[str]) -> List[Document]:
    documents = []
    context = DocumentContext()  # Initialize context tracker
    
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
                    "detect_titles": True,
                    "detect_lists": True,
                    "detect_tables": True,
                    "detection_settings": {
                        "ocr_on": True,
                        "text_formatting": True
                    }
                }
            )
            loaded_docs = loader.load()
            
            if not loaded_docs:
                logger.warning(f"No documents loaded from {file_path}")
                continue
            
            # Process each document with context
            processed_docs = []
            for doc in loaded_docs:
                # Update context first
                context.update_context(doc)
                
                # Create base document with file info
                base_doc = Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "file_path": file_path,
                        "document_type": "FRA"
                    }
                )
                
                # Process document
                processed = preprocess_document(base_doc)
                
                # Add context metadata to each processed document
                for p_doc in processed:
                    p_doc.metadata = process_metadata(p_doc, context.get_current_context())
                
                processed_docs.extend(processed)
                
            documents.extend(processed_docs)
                
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

#Chroma db functions
#1. Initialize chroma db collection 
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
#2. index document chunks in chroma db
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
            
            # Ensure all metadata values are not None
            metadatas = [{
                "source": chunk.metadata.get("source") or "",
                "file_path": chunk.metadata.get("file_path") or "",
                "document_type": chunk.metadata.get("document_type") or "unknown",
                "section_type": chunk.metadata.get("section_type") or "",
                "parent_section": chunk.metadata.get("parent_section") or "",
                "question_id": chunk.metadata.get("question_id") or "",
                "action_id": chunk.metadata.get("action_id") or "",
                "content_type": chunk.metadata.get("content_type") or "",
                "priority": chunk.metadata.get("priority") or "",
                "due_date": chunk.metadata.get("due_date") or "",
                "client_status": chunk.metadata.get("client_status") or "",
                "chunk_id": str(i + idx)
            } for idx, chunk in enumerate(batch)]
            
            embeddings = embedding_model.embed_documents(documents)
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

#Main function
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