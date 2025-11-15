"""
Embeddings handler module for creating text embeddings.
Uses HuggingFace's sentence-transformers for local embedding generation.
"""
from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Initialize and return HuggingFace embeddings model.
    
    Args:
        model_name: Name of the sentence-transformers model to use
        
    Returns:
        HuggingFaceEmbeddings object
    """
    print(f"Loading embeddings model: {model_name}")
    print("  (First run will download the model - please wait)")
    
    # Initialize HuggingFace embeddings with the specified model
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},  # Use CPU (change to 'cuda' if GPU available)
        encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity
    )
    
    print("Embeddings model loaded successfully")
    
    return embeddings
