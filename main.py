"""
Q&A System for Dr. B.R. Ambedkar's Speech
Main entry point for the command-line RAG application.
"""
import os
from pathlib import Path

# Import our custom modules
from src.document_loader import load_document
from src.text_processor import split_text
from src.embeddings_handler import get_embeddings_model
from src.vector_store import create_vector_store, load_vector_store
from src.qa_system import create_qa_chain, ask_question


def setup_rag_system(speech_file: str = "data/speech.txt", force_rebuild: bool = False):
    """
    Set up the complete RAG system.
    
    Args:
        speech_file: Path to the speech text file
        force_rebuild: Force rebuild of vector store even if it exists
        
    Returns:
        qa_chain: The RetrievalQA chain ready for questions
    """
    print("=" * 60)
    print("Initializing RAG System")
    print("=" * 60)
    
    # Initialize embeddings model (needed for both creating and loading)
    embeddings = get_embeddings_model()
    
    # Check if vector store already exists
    vector_store_path = "./chroma_db"
    vector_store_exists = os.path.exists(vector_store_path)
    
    if vector_store_exists and not force_rebuild:
        print("\nExisting vector store found - loading...")
        vectorstore = load_vector_store(embeddings, vector_store_path)
    else:
        if force_rebuild:
            print("\nForce rebuild requested - recreating vector store...")
        else:
            print("\nNo existing vector store found - creating new one...")
        
        # Step 1: Load document
        documents = load_document(speech_file)
        
        # Step 2: Split into chunks
        chunks = split_text(documents, chunk_size=500, chunk_overlap=50)
        
        # Step 3: Create vector store
        vectorstore = create_vector_store(chunks, embeddings, vector_store_path)
    
    # Step 4: Create QA chain
    qa_chain = create_qa_chain(vectorstore, model_name="mistral")
    
    print("\n" + "=" * 60)
    print("System Ready! You can now ask questions.")
    print("=" * 60)
    
    return qa_chain


def interactive_mode(qa_chain):
    """
    Run interactive Q&A session.
    
    Args:
        qa_chain: The RetrievalQA chain
    """
    print("\nInteractive Mode - Type your questions (or 'quit'/'exit' to stop)\n")
    
    while True:
        try:
            # Get user input
            question = input("You: ").strip()
            
            # Check for exit commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            # Skip empty questions
            if not question:
                continue
            
            # Ask question and get answer
            response = ask_question(qa_chain, question)
            
            # Optionally show source documents
            if response.get('source_documents'):
                print(f"(Retrieved {len(response['source_documents'])} relevant chunks)")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


def main():
    """
    Main function to run the application.
    """
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Q&A System for Dr. B.R. Ambedkar's Speech"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild the vector store from scratch"
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        help="Ask a single question (non-interactive mode)"
    )
    parser.add_argument(
        "--speech-file",
        type=str,
        default="data/speech.txt",
        help="Path to speech text file (default: data/speech.txt)"
    )
    
    args = parser.parse_args()
    
    try:
        # Set up the RAG system
        qa_chain = setup_rag_system(
            speech_file=args.speech_file,
            force_rebuild=args.rebuild
        )
        
        # Single question mode
        if args.question:
            ask_question(qa_chain, args.question)
        else:
            # Interactive mode
            interactive_mode(qa_chain)
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure 'data/speech.txt' exists!")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
