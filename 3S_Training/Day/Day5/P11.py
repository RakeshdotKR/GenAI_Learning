"""
P11.py — Retrieval Quality Tuning

SUMMARY:
This script compares three ChromaDB retrieval strategies for the same question:
ordinary similarity search, similarity search with a score threshold, and
Maximal Marginal Relevance (MMR), which balances relevance with diversity.
It prints the retrieved chunks for each strategy so their results can be compared.

PREREQUISITES:
- A compatible ChromaDB collection must already exist at ./chroma.
- Configure GEMINI_API_KEY in a .env file.
- Install: pip install python-dotenv langchain-chroma langchain-google-genai
"""

# Access environment variables and operating-system functions.
import os
# Load API credentials and settings from a .env file.
from dotenv import load_dotenv
# Import LangChain's integration with the Chroma vector database.
from langchain_chroma import Chroma
# Import the Gemini embedding integration for LangChain.
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Read key-value settings from the .env file into the process environment.
# Load API credentials such as GEMINI_API_KEY from the .env file.
load_dotenv()

# Configure Gemini's embedding model to convert text into vectors.
embeddings  = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
vectorstore = Chroma(embedding_function=embeddings)  # loads ./chroma (default path)

# Use one question to compare the retrieval strategies fairly.
question = "What tools help build LLM applications?"

# Map each strategy name to a retriever configured with different search settings.
strategies = {
    # Default — return the k most similar chunks
    # Create a retriever that returns the most relevant stored documents.
    "similarity (k=3)": vectorstore.as_retriever(
        search_kwargs={"k": 3}
    ),
    # Score threshold — drop chunks below a relevance cutoff
    # Create a retriever that returns the most relevant stored documents.
    "score_threshold":  vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.4, "k": 3},
    ),
    # MMR — balance relevance AND diversity (avoids near-duplicate chunks)
    # Create a retriever that returns the most relevant stored documents.
    "mmr (k=3)":        vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10},  # fetch 10 candidates, keep 3 diverse
    ),
}

# Process each item in the collection one at a time.
for name, retriever in strategies.items():
    # Execute the chain or retriever using the current input.
    docs = retriever.invoke(question)
    # Print the current result to the terminal.
    print(f"\n--- {name} ({len(docs)} chunks) ---")
    # Process each item in the collection one at a time.
    for d in docs:
        # Print the current result to the terminal.
        print(f"  [{d.metadata.get('source', '?')}] {d.page_content[:70]}...")
