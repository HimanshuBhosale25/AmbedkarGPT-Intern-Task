"""
Vector store module for storing and retrieving document embeddings.
Uses ChromaDB for local, persistent vector storage.
"""
from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document


def create_vector_store(chunks: List[Document], embeddings, persist_directory: str = "./chroma_db"):
    """
    Create a ChromaDB vector store from document chunks.
    
    Args:
        chunks: List of document chunks to embed
        embeddings: Embeddings model to use
        persist_directory: Directory to persist the vector store
        
    Returns:
        Chroma vector store object
    """
    print(f"Creating vector store at: {persist_directory}")
    print(f"Embedding {len(chunks)} chunks (this may take a moment)...")
    
    # Create ChromaDB vector store from documents
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="ambedkar_speech"
    )
    
    print(f"Vector store created with {vectorstore._collection.count()} embeddings")
    
    return vectorstore


def load_vector_store(embeddings, persist_directory: str = "./chroma_db"):
    """
    Load existing ChromaDB vector store.
    
    Args:
        embeddings: Embeddings model (must match the one used during creation)
        persist_directory: Directory where vector store is persisted
        
    Returns:
        Chroma vector store object
    """
    print(f"Loading existing vector store from: {persist_directory}")
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="ambedkar_speech"
    )
    
    print(f"Vector store loaded with {vectorstore._collection.count()} embeddings")
    
    return vectorstore
