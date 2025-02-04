"""Simple RAG retriever and response chain for FRA documents."""
from typing import Any

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
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    template = """You are an expert at analyzing Fire Risk Assessment documents. Answer the question based ONLY on the context provided. Include ONLY information that is supported by the context.

Context:
{context}

Question: {question}

When providing your answer:
- If the context doesn't contain enough information to fully answer the question, say so
- If dates are mentioned, include them exactly as given
- For specific requirements or recommendations, quote them directly
- If multiple sections of context have relevant information, combine them logically

Answer:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

#@traceable
def do_everything():
     # Load environment variables
    load_environment()
    # Example usage
    vectorstore = get_vectorstore()
    chain = create_rag_chain(vectorstore)
    
    # Example queries
    queries = [
        "What are the main fire safety recommendations?",
        "Descrbe the layout of the building"
    ]
    
    for query in queries:
        print(f"\nQuestion: {query}")
        print(f"Answer: {chain.invoke(query)}\n") 


if __name__ == "__main__":
    do_everything()