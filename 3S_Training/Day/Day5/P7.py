"""
P7.py — Build KB + Raw RAG

SUMMARY:
This script builds a small knowledge base by splitting text into chunks,
creating Gemini embeddings, and storing the chunks and vectors in a persistent
ChromaDB collection. It then embeds each question, retrieves the most relevant
chunks, sends the retrieved context and question to Gemini, and prints answers.

PREREQUISITES:
- Set GEMINI_API_KEY in a .env file.
- Install: pip install python-dotenv google-genai langchain-text-splitters chromadb
"""

# Step 1: build KB (run once), Step 2: query it — all in one script
# Access environment variables and operating-system functions.
import os

# Load API credentials and settings from a .env file.
from dotenv import load_dotenv

# Import the Google Gen AI SDK client interface.
from google import genai

# Import types used to configure Gemini generation requests.
from google.genai import types

# Import a text splitter for creating overlapping PDF chunks.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import ChromaDB, the vector database used to store and search embeddings.
import chromadb

# Read key-value settings from the .env file into the process environment.
# Load API credentials such as GEMINI_API_KEY from the .env file.
load_dotenv()

# Read the API key from the environment.
api_key = os.getenv("GEMINI_API_KEY")

# Stop execution if the API key is missing.
if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found. Add it to your .env file."
    )

# Create a Gemini API client using the key loaded from the environment.
client = genai.Client(api_key=api_key)

# Store the sample knowledge-base text that will be split, embedded, and searched.
KB_TEXT = """
Python is a high-level, general-purpose programming language.
It was created by Guido van Rossum and first released in 1991.
Python supports object-oriented, functional, and procedural programming.

LangChain is a framework for building LLM-powered applications.
It provides prompts, chains, memory, retrievers, and tool integrations.
LangChain uses LCEL (the pipe | syntax) to compose components.

RAG stands for Retrieval Augmented Generation.
RAG solves the problem of LLMs not knowing about private documents.
The retriever finds relevant chunks; the LLM generates an answer from them.

ChromaDB is an open-source local vector database.
PersistentClient saves the database to disk between runs.
The where= parameter filters results by metadata at query time.
"""

# ── Build: chunk → embed → store ─────────────────────────────────────────────

# Set the target chunk size and overlap between neighboring chunks.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)

# Split the knowledge-base text into smaller, overlapping pieces.
# Split the trimmed knowledge-base text into overlapping chunks.
chunks = splitter.split_text(KB_TEXT.strip())

# Create an empty list to hold one embedding vector for each chunk.
embeddings = []

# Generate embeddings individually to ensure one vector is returned per chunk.
for i, chunk in enumerate(chunks, start=1):

    # Ask Gemini to convert the current text chunk into a numeric vector.
    # Pass a single string instead of a list of strings.
    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=chunk
    )

    # Extract the embedding vector returned for this individual chunk.
    embedding = result.embeddings[0].values

    # Add the vector to the list of embeddings.
    embeddings.append(embedding)

    # Display progress while generating the embeddings.
    print(f"Generated embedding for chunk {i}/{len(chunks)}")

# Verify that every chunk has a corresponding embedding vector.
assert len(chunks) == len(embeddings), (
    f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
)

# Open a ChromaDB database that persists data on disk.
db = chromadb.PersistentClient(path="./rag_db")

# Open the collection if it exists, or create it.
col = db.get_or_create_collection("knowledge")

# Store each text chunk with its corresponding embedding and ID.
# Each of these three lists must have the same length.
# Use upsert so rerunning the script updates existing IDs instead of
# raising duplicate-ID errors when the collection already contains them.
col.upsert(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"c{i}" for i in range(len(chunks))]
)

# Print the current result to the terminal.
print(f"Stored {col.count()} chunks → ./rag_db")

# ── Query: embed question → retrieve → prompt → LLM ──────────────────────────

# Define a function so this operation can be reused.
# Retrieve relevant knowledge-base chunks and ask Gemini to answer the question.
def rag_answer(question: str, top_k: int = 3) -> str:

    # Embed the question the same way we embedded the chunks.
    # Ask the embedding model to convert the supplied text into a numeric vector.
    # gemini-embedding-2: for Advanced cross-modal search, multimodal RAG pipelines, and unified asset indexing.
    q_result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=question
    )

    # Extract the question's embedding vector.
    # Only one question was supplied, so its first vector is appropriate here.
    q_emb = q_result.embeddings[0].values

    # Find the top-k most similar chunks.
    # Retrieve the stored chunks whose vectors are closest to the question vector.
    # Search ChromaDB for the top_k chunks closest to the question embedding.
    results = col.query(
        query_embeddings=[q_emb],
        n_results=min(top_k, col.count())
    )

    # Join the retrieved text chunks into one context block for the LLM.
    context = "\n\n".join(results["documents"][0])

    # Inject context into the prompt and ask the LLM.
    # Ask Gemini to answer using the retrieved context and the user's question.
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=(
            "Use only the context below to answer. "
            "If the answer is not present in the context, say you do not "
            "know based on the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        ),
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=512
        ),
    )

    # Return the computed value to the code that called the function.
    return response.text or "No text response was generated."


# Process each item in the collection one at a time.
for q in [
    "Who created Python?",
    "What does RAG stand for?",
    "What does the where= parameter do?"
]:
    # Print the current result to the terminal.
    print(f"\nQ: {q}\nA: {rag_answer(q)}")
