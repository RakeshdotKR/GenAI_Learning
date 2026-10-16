"""
P3.py — FAQ Search Engine (Mini-Challenge)

SUMMARY:
Loads 10 FAQ question-answer pairs into a ChromaDB in-memory collection,
then runs three natural-language queries and prints the top-2 matches with
their distance scores. Demonstrates semantic search on short Q&A content.

PREREQUISITES:
- pip install chromadb
"""

import chromadb

client = chromadb.Client()
faq = client.create_collection("course_faq")

questions = [
    "What is Python used for?",
    "How do I install Python?",
    "What is a virtual environment?",
    "What is an API?",
    "What is machine learning?",
    "What is the difference between AI and ML?",
    "What is LangChain?",
    "What is RAG?",
    "What is ChromaDB?",
    "How do embeddings work?",
]

# Store answers as documents; link each to its question via metadata
answers = [
    "Python is used for web development, data science, automation, and AI.",
    "Download from python.org and run the installer. Use python --version to verify.",
    "A virtual environment isolates project dependencies so they don't conflict.",
    "An API (Application Programming Interface) lets programs talk to each other.",
    "Machine learning is a method of teaching computers to learn from data.",
    "AI is the broad field; ML is a technique within AI that learns from data.",
    "LangChain is a framework that connects LLMs with prompts, memory, and tools.",
    "RAG retrieves relevant documents and injects them as context into an LLM prompt.",
    "ChromaDB is an open-source vector database for storing and searching embeddings.",
    "Embeddings convert text into a list of numbers that capture semantic meaning.",
]

faq.add(
    documents=answers,
    metadatas=[{"question": q} for q in questions],
    ids=[f"faq_{i}" for i in range(len(questions))],
)

print(f"Loaded {faq.count()} FAQ entries\n")

test_queries = [
    "how do I set up Python?",
    "what is the difference between retrieval and generation?",
    "tell me about vector databases",
]

for query in test_queries:
    results = faq.query(query_texts=[query], n_results=2)
    print(f"Query: {query!r}")
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        print(f"  [{dist:.4f}] Q: {meta['question']}")
        print(f"           A: {doc}")
    print()
