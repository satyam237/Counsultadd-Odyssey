from typing import Annotated

from IPython.display import Image, display  # noqa: F401
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

from langgraph.graph.message import add_messages

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
from langchain.prompts import ChatPromptTemplate
from pathlib import Path
from dotenv import load_dotenv
import os
import re
import traceback

load_dotenv()

# Initialize embeddings and vector stores
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Define paths and load vector stores
data_dir = Path("data")
vector_stores_dir = data_dir / "vector_stores"


def load_vector_store(store_name: str) -> Chroma:
    """Load a vector store."""
    store_path = vector_stores_dir / store_name

    try:
        # Initialize ChromaDB with proper configuration
        return Chroma(
            persist_directory=str(store_path),
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},  # Add default metadata
            collection_name=store_name,  # Explicitly set collection name
        )
    except Exception as e:
        print(f"Error loading vector store: {str(e)}")
        print("Attempting to recreate the collection...")

        # If loading fails, try to create a new collection
        client = chromadb.PersistentClient(path=str(store_path))

        # Delete existing collection if it exists
        try:
            client.delete_collection(store_name)
        except:
            pass

        # Create new collection with proper configuration
        return Chroma(
            persist_directory=str(store_path),
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
            collection_name=store_name,
        )


# Load only eligible_rfp1 vector store
eligible_rfp_store = load_vector_store("eligible_rfp1")

[REST OF THE FILE CONTENT]