"""
Question-Answering system using RAG pipeline.
Combines vector retrieval with Ollama LLM for answering questions.
"""
from langchain_ollama import OllamaLLM
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


def create_qa_chain(vectorstore, model_name: str = "mistral"):
    """
    Create a retrieval chain with Ollama LLM.
    
    Args:
        vectorstore: ChromaDB vector store for retrieval
        model_name: Name of Ollama model to use
        
    Returns:
        Retrieval chain object
    """
    print(f"Initializing Ollama with model: {model_name}")
    
    # Initialize Ollama LLM
    llm = OllamaLLM(
        model=model_name,
        temperature=0.2,  # Lower temperature for more focused answers
    )
    
    # Create prompt template
    system_prompt = """Use the following context to answer the question. 
If you cannot find the answer in the context, say "I cannot find this information in the provided text."

Context: {context}"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Create document combining chain
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create retrieval chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    print("QA chain created successfully")
    
    return retrieval_chain


def ask_question(qa_chain, question: str) -> dict:
    """
    Ask a question using the retrieval chain.
    
    Args:
        qa_chain: Retrieval chain
        question: Question to ask
        
    Returns:
        Dictionary with 'answer' and 'context'
    """
    print(f"\nQuestion: {question}")
    
    response = qa_chain.invoke({"input": question})
    
    print(f"Answer: {response['answer']}\n")
    
    return response
