"""
Document loader module for loading text files.
Uses LangChain's TextLoader to read the speech document.
"""
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from typing import List
from langchain_core.documents import Document


def load_document(file_path: str) -> List[Document]:
    """
    Load text document from file path.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        List of Document objects containing the loaded text
    """
    # Verify file exists
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load document using TextLoader with UTF-8 encoding
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    
    print(f"Loaded document from {file_path}")
    print(f"Document length: {len(documents[0].page_content)} characters")
    
    return documents
