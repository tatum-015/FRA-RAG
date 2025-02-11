#ingest.py
from pathlib import Path
from typing import List, Dict, Any, Optional, Type
import logging
import json
import os
import glob
from google import genai
from pydantic import BaseModel, Field
from langchain.docstore.document import Document

from fra_rag.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    GEMINI_API_KEY 
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Pydantic models for structured output
class AssessmentDetails(BaseModel):
    property_classification: str = Field(description="The classification level of the property")
    bafe_certificate_number: str = Field(description="The BAFE SP205-1 certificate reference number")
    responsible_person: str = Field(description="Organization or person responsible for the property")
    assessment_completed_by: str = Field(description="Name of the assessor who completed the assessment")
    assessment_checked_by: str = Field(description="Name of the person who checked/verified the assessment")
    inspection_date: str = Field(description="Date when the inspection was carried out")
    issue_date: str = Field(description="Date when the assessment was issued to the client")

class Action(BaseModel):
    action_id: str = Field(description="The unique identifier for this action (e.g., M.8)")
    comment: str = Field(description="A comment on the action")
    recommendation: str = Field(description="The full recommendation of what needs to be done")
    priority: str = Field(description="Priority level (High, Medium, or Low)")
    due_date: str = Field(description="When this action needs to be completed by")

class Section(BaseModel):
    section_id: str = Field(description="The section number or identifier")
    title: str = Field(description="The title of this section")
    content: str = Field(description="The main content text of this section")
    questions: List[str] = Field(description="List of questions in this section, including their IDs")
    actions: List[Action] = Field(description="List of actions required in this section")


class FRADocument(BaseModel):
    document_id: str = Field(description="The UPRN number or other unique identifier for this document")
    assessment_date: str = Field(description="The date when this assessment was carried out")
    building_name: str = Field(description="The name or address of the building assessed")
    assessment_details: AssessmentDetails = Field(description="Detailed assessment information")
    sections: List[Section] = Field(description="All sections of the fire risk assessment")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)
model_id = "gemini-2.0-flash"

def extract_structured_data(file_path: str, model: Type[BaseModel]):
    """
    Extract structured data from a PDF file using Gemini's File API.
    
    Args:
        file_path: Path to the PDF file
        model: Pydantic model class to structure the output
        
    Returns:
        Parsed pydantic model instance
    """
    # Step 1: Upload the file to the File API
    file = client.files.upload(
        file=file_path, 
        config={'display_name': os.path.basename(file_path).split('.')[0]}
    )
    
    # Step 2: Extract raw data using Gemini
    prompt = """Please extract the following information from this PDF file and return it as a JSON object:
    - document_id: The UPRN or unique identifier
    - assessment_date: When the assessment was carried out
    - building_name: Name/address of building assessed
    - assessment_details: Object containing property classification, BAFE number, etc.
    - sections: Array of sections with their IDs, titles, content, and actions
    
    Return the data as a clean JSON object with no additional text."""
    
    response = client.models.generate_content(
        model=model_id, 
        contents=[prompt, file], 
        config={'response_mime_type': 'application/json', 'response_schema': model}
    )
    
    # Step 3: Parse the raw response into JSON
    try:
        json_data = json.loads(response.text)
        logger.debug(f"Extracted JSON: {json.dumps(json_data, indent=2)}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini response as JSON: {response.text}")
        raise
        
    # Step 4: Validate against Pydantic model
    try:
        return model.model_validate(json_data)
    except Exception as e:
        logger.error(f"Failed to validate JSON against model: {str(e)}")
        raise

def process_document(file_path: str) -> Optional[Document]:
    """Process a single document through the pipeline."""
    try:
        logger.info(f"Processing: {file_path}")
        
        # Extract structured data
        fra_doc = extract_structured_data(file_path, FRADocument)
        
        # Get token count for the file
        file = client.files.upload(
            file=file_path,
            config={'display_name': os.path.basename(file_path)}
        )
        file_size = client.models.count_tokens(model=model_id, contents=file)
        
        # Create Document object
        return Document(
            page_content=fra_doc.model_dump_json(indent=2),
            metadata={
                "file_path": file_path,
                "document_id": fra_doc.document_id,
                "assessment_date": fra_doc.assessment_date,
                "building_name": fra_doc.building_name,
                "token_count": file_size.total_tokens
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {str(e)}")
        return None

def load_documents(file_paths: List[str]) -> List[Document]:
    """Process multiple documents."""
    documents = []
    
    for file_path in file_paths:
        doc = process_document(file_path)
        if doc:
            documents.append(doc)
            logger.info(f"Successfully processed: {file_path}")
        else:
            logger.warning(f"Failed to process: {file_path}")
    
    if documents:
        output_path = Path(PROCESSED_DATA_DIR) / "extracted_documents.json"
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump([
                    {
                        "content": json.loads(doc.page_content),
                        "metadata": doc.metadata
                    } for doc in documents
                ], f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(documents)} documents to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save documents: {str(e)}")
    
    return documents

if __name__ == "__main__":
    file_paths = glob.glob(os.path.join(RAW_DATA_DIR, "*"))
    logger.info(f"Found {len(file_paths)} files in {RAW_DATA_DIR}")
    
    if not file_paths:
        logger.warning(f"No files found in {RAW_DATA_DIR}")
        exit(1)
    
    documents = load_documents(file_paths)
    
    if not documents:
        logger.warning("No documents were successfully processed")
        exit(1)
    
    logger.info(f"Successfully processed {len(documents)} documents")