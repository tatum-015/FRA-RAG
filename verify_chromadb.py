import chromadb
import json

from fra_rag.config import CHROMA_SETTINGS

# Initialize Chroma client
client = chromadb.PersistentClient(path=CHROMA_SETTINGS["persist_directory"])

# List all collections
collections = client.list_collections()
print("Available Collections:")
for collection in collections:
    print(f"- {collection}")

# Get a specific collection
collection = client.get_collection("fra_documents")
print(collection)

# Get the number of documents in the collection
num_documents = collection.count()
print(f"Number of documents in the collection: {num_documents}")

results = collection.query(
    query_texts=[" This has a similar scope of work"], # Chroma will embed this for you
    n_results=2, # how many results to return
)
  
print(json.dumps(results, indent=4, ensure_ascii=False))


