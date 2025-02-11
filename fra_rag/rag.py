"""Simple RAG retriever and response chain for FRA documents."""
from typing import Any, Dict

from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
#from langsmith import traceable


from fra_rag.config import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME
from fra_rag.utils import load_environment

def get_vectorstore() -> Chroma:
    """Initialize ChromaDB vectorstore with HuggingFace embeddings."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    
    return Chroma(
        persist_directory=CHROMA_SETTINGS["persist_directory"],
        collection_name=CHROMA_SETTINGS["collection_name"],
        embedding_function=embeddings
    )

def create_rag_chain(vectorstore: Any) -> Any:
    """Create a RAG chain with retriever and response generation."""
    # Enhanced retriever with metadata filtering
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 8,  # Fetch more results for better filtering
            "filter": {
                "document_type": "FRA"  # Only retrieve FRA documents
            }
        }
    )

    template = """You are an expert at analyzing Fire Risk Assessment documents. Answer the question based ONLY on the context provided. Include ONLY information that is supported by the context.

Context:
{context}

Question: {question}

When providing your answer:
- Quote recommendations exactly as given, including Action IDs and Question IDs (e.g., M.8)
- Always include priorities and due dates for actions
- State if there isn't enough context to fully answer
- Present findings in order of priority
- Reference related findings and recommendations together
Answer:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        """Enhanced document formatting with metadata."""
        formatted_docs = []
        for doc in docs:
            # Format based on content type
            if doc.metadata.get('content_type') == 'action_item':
                formatted_docs.append(
                    f"[Action Item {doc.metadata.get('action_id', 'Unknown')}]\n"
                    f"Priority: {doc.metadata.get('priority', 'Unknown')}\n"
                    f"Section: {doc.metadata.get('section_type', 'Unknown')}\n"
                    f"{doc.page_content}"
                )
            else:
                formatted_docs.append(
                    f"[{doc.metadata.get('section_type', 'General')}]\n"
                    f"{doc.page_content}"
                )
        return "\n\n".join(formatted_docs)

    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

def query_by_section(chain: Any, question: str, section_type: str) -> str:
    """Query specifically within a section type."""
    vectorstore = get_vectorstore()
    section_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "filter": {
                "section_type": section_type
            }
        }
    )
    return chain.invoke({"question": question, "retriever": section_retriever})

def query_action_items(chain: Any, priority: str = None) -> str:
    """Query specifically for action items, optionally filtered by priority."""
    vectorstore = get_vectorstore()
    filter_dict = {"content_type": "action_item"}
    if priority:
        filter_dict["priority"] = priority
        
    action_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "filter": filter_dict
        }
    )
    return chain.invoke({"question": "What are the action items?", "retriever": action_retriever})

#@traceable
def do_everything():
     # Load environment variables
    load_environment()
    # Example usage
    vectorstore = get_vectorstore()
    chain = create_rag_chain(vectorstore)
    
    # Example queries
    queries = [
        "What are the main fire safety actions?",
       # "List all high priority action items.",
       # "What are the recommendations for fire doors?"
    ]
    
    for query in queries:
        print(f"\nQuestion: {query}")
        print(f"Answer: {chain.invoke(query)}\n") 


if __name__ == "__main__":
    do_everything()