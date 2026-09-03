import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient


load_dotenv()


# -----------------------------
# Configuration
# -----------------------------

PDF_FOLDER = Path("data/papers")

COLLECTION_NAME = "rag_eval"

EMBEDDING_MODEL = "mxbai-embed-large:latest"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


# -----------------------------
# Embedding Model
# -----------------------------

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)


# -----------------------------
# Qdrant Client
# -----------------------------

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY if QDRANT_API_KEY else None
)


# -----------------------------
# PDF Loading
# -----------------------------

def load_pdfs():

    documents = []

    pdf_files = list(PDF_FOLDER.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {PDF_FOLDER}"
        )

    for pdf_path in pdf_files:

        print(f"Loading: {pdf_path.name}")

        loader = PyMuPDFLoader(str(pdf_path))

        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = pdf_path.name

        documents.extend(docs)

    print(f"Total pages loaded: {len(documents)}")

    return documents


# -----------------------------
# Chunking
# -----------------------------

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    return chunks


# -----------------------------
# Store in Qdrant
# -----------------------------

def store_documents(chunks):

    print("Creating embeddings and storing in Qdrant...")

    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
        collection_name=COLLECTION_NAME,
    )

    print("Documents successfully stored in Qdrant.")

    return vector_store


# -----------------------------
# Main Pipeline
# -----------------------------

def main():

    documents = load_pdfs()

    chunks = split_documents(documents)

    store_documents(chunks)


if __name__ == "__main__":
    main()