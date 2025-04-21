"""
RAG Pipeline Script

This script implements a Retrieval-Augmented Generation (RAG) pipeline using LangChain, Llama.cpp, and ChromaDB.
It loads configuration from a .env file, supports PDF and TXT documents, and provides a CLI for user interaction.
"""
import os
import sys
import logging
import time
import requests
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables from .env
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Configuration from .env
MODELS_DIR = os.getenv("MODELS_DIR", "models") # Define models directory
MODEL_FILENAME = os.getenv("MODEL_FILENAME") # Get filename from .env, no default
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME) # Construct full model path
MODEL_URL = os.getenv("MODEL_URL")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.7))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 2000))
LLM_N_CTX = int(os.getenv("LLM_N_CTX", 4096))
LLM_VERBOSE = os.getenv("LLM_VERBOSE", "False").lower() == "true"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "./chroma_db")
DOCS_DIRECTORY = os.getenv("DOCS_DIRECTORY", "./docs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
SEARCH_K = int(os.getenv("SEARCH_K", 3))
LLM_N_BATCH = int(os.getenv("LLM_N_BATCH", 512))


def download_model(model_path, model_url):
    """Download the model file if not present."""
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        logging.info(f"Creating directory {model_dir}")
        os.makedirs(model_dir)

    if os.path.exists(model_path):
        logging.info(f"Model file '{model_path}' already exists.")
        return
    if not model_url:
        logging.error("Model URL is not specified in the .env file.")
        sys.exit(1)
    try:
        logging.info(f"Downloading {model_path} from {model_url} ...")
        response = requests.get(model_url, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(model_path, 'wb') as f:
            for data in tqdm(response.iter_content(chunk_size=1024), total=max(total_size//1024, 1)):
                f.write(data)
        logging.info("Download complete!")
    except Exception as e:
        logging.error(f"Failed to download model: {e}")
        sys.exit(1)


def load_documents(docs_dir):
    """Load all PDF and TXT documents from the docs directory."""
    documents = []
    for file in os.listdir(docs_dir):
        path = os.path.join(docs_dir, file)
        try:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(path)
                documents.extend(loader.load())
            elif file.endswith(".txt"):
                loader = TextLoader(path)
                documents.extend(loader.load())
        except Exception as e:
            logging.warning(f"Failed to load {file}: {e}")
    if not documents:
        logging.warning("No documents loaded. The LLM will be used without retrieval context.")
    return documents


def build_vectorstore(chunks):
    """Build or load the Chroma vectorstore from document chunks."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )


def build_llm():
    """Initialize the LlamaCpp LLM."""
    # Ensure the model path is correct before initializing
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model file not found at {MODEL_PATH}. Please ensure it is downloaded.")
        sys.exit(1)
    
    n_gpu_layers = os.getenv("LLM_N_GPU_LAYERS", "0")

    return LlamaCpp(
        model_path=MODEL_PATH,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        n_ctx=LLM_N_CTX,
        n_gpu_layers=int(n_gpu_layers), 
        n_batch=LLM_N_BATCH,
        verbose=LLM_VERBOSE,
        f16_kv=True
    )


def build_rag_pipeline(vectorstore, llm):
    """Build the RAG pipeline."""
    template = """
Answer the question based on the following context:

{context}

Question: {question}
Answer:
"""
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": SEARCH_K}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )


def ask_question(rag_pipeline, question):
    """Ask a question using the RAG pipeline and print the answer and sources."""
    start_time = time.time()
    result = rag_pipeline.invoke({"query": question})
    end_time = time.time()
    print(f"\nQuestion: {question}")
    print(f"Answer: {result['result']}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    source_documents = result.get("source_documents")
    if source_documents:
        print("\nRetrieved documents potentially used for context:")
        unique_sources = set(doc.metadata.get('source', 'Unknown') for doc in source_documents)
        for source in unique_sources:
            print(f"- {source}")
        print() # Add a newline for better formatting
    else:
        print("\nNo documents were retrieved for context.\n")


def main():
    """Main entry point for CLI interaction."""
    download_model(MODEL_PATH, MODEL_URL)
    documents = load_documents(DOCS_DIRECTORY)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents) if documents else []
    vectorstore = build_vectorstore(chunks) if chunks else None
    llm = build_llm()
    rag_pipeline = build_rag_pipeline(vectorstore, llm) if vectorstore else llm

    print("\nRAG pipeline is ready. Type your question and press Enter (Ctrl+C to exit).\n")
    try:
        while True:
            question = input("Your question: ").strip()
            if not question:
                continue
            if vectorstore:
                ask_question(rag_pipeline, question)
            else:
                # Direct LLM call if no documents
                start_time = time.time()
                result = llm.invoke(question)
                end_time = time.time()
                print(f"\nQuestion: {question}")
                print(f"Answer: {result}")
                print(f"Time taken: {end_time - start_time:.2f} seconds\n")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")


if __name__ == "__main__":
    main()

