import os

from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


# -----------------------------
# Configuration
# -----------------------------

COLLECTION_NAME = "rag_eval"

EMBEDDING_MODEL = "mxbai-embed-large:latest"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROK_API_KEY=os.getenv("GROK_API_KEY")

# -----------------------------
# Embeddings
# -----------------------------

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)


# -----------------------------
# Connect to Qdrant
# -----------------------------

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
)


# -----------------------------
# Retriever
# -----------------------------

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 1
    }
)




# -----------------------------
# Retriever function
# -----------------------------

def retrieve_documents(query):

    documents = retriever.invoke(query)

    return documents


# -----------------------------
# Test Retriever 
# -----------------------------

if __name__=="__main__":
    document=retrieve_documents("rag architecture")
    for doc in document:
        print(doc.page_content)
