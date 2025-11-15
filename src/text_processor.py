"""
Text processing module for splitting documents into chunks.
Uses CharacterTextSplitter for simple, efficient chunking.
"""
from langchain_text_splitters import CharacterTextSplitter
from typing import List
from langchain_core.documents import Document


def split_text(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """
    Split documents into smaller chunks for embedding.
    
    Args:
        documents: List of Document objects to split
        chunk_size: Maximum characters per chunk
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of chunked Document objects
    """
    # Initialize text splitter with specified parameters
    text_splitter = CharacterTextSplitter(
        separator="\n",           # Split on newlines
        chunk_size=chunk_size,    # Max characters per chunk
        chunk_overlap=chunk_overlap,  # Overlap for context
        length_function=len,      # Use character count
    )
    
    # Split documents into chunks
    chunks = text_splitter.split_documents(documents)
    
    print(f"Split document into {len(chunks)} chunks")
    print(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    
    return chunks
