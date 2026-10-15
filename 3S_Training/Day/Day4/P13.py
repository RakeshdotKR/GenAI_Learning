"""
P13.py — ChromaDB In-Memory CRUD: Add, Query, Update, Delete

SUMMARY:
Demonstrates the four core ChromaDB operations on an in-memory collection.
Section A: create a collection and add documents.
Section B: semantic search (query) with natural language.
Section C: update an existing document, then delete it.

PREREQUISITES:
- pip install chromadb
"""

import chromadb

# ── Section A: create collection and add documents ───────────────────────────

# In-memory client — data is lost when the process ends
client = chromadb.Client()

collection = client.create_collection("my_first_collection")

# ChromaDB auto-generates embeddings using its built-in default model
collection.add(
    documents=[
        "Python is a high-level programming language.",
        "ChromaDB stores embeddings for semantic search.",
        "RAG combines retrieval with language generation.",
        "LangChain helps build LLM-powered applications.",
        "Cosine similarity measures the angle between vectors.",
    ],
    metadatas=[
        {"source": "python_intro.txt"},
        {"source": "chroma_docs.txt"},
        {"source": "rag_paper.txt"},
        {"source": "langchain_docs.txt"},
        {"source": "math_notes.txt"},
    ],
    ids=["d1", "d2", "d3", "d4", "d5"],
)

print("Documents in collection:", collection.count())

# ── Section B: semantic query ─────────────────────────────────────────────────

# Natural language query — no exact keyword match needed
# ChromaDB returns distances (lower = more similar, not a similarity score)
results = collection.query(
    query_texts=["how do we search documents using vectors?"],
    n_results=2,
)

print("\nTop matches:")
for doc, dist in zip(results["documents"][0], results["distances"][0]):
    print(f"  [{dist:.4f}] {doc}")

# ── Section C: update and delete ─────────────────────────────────────────────

update_col = client.create_collection("updates_demo")
update_col.add(documents=["Old content about Python."], ids=["doc1"])

# Replace a document's content by its ID
update_col.update(ids=["doc1"], documents=["Updated content: Python is great for AI."])

fetched = update_col.get(ids=["doc1"])
print("\nAfter update:", fetched["documents"])

# Remove a document by its ID
update_col.delete(ids=["doc1"])
print("After delete, count:", update_col.count())  # 0
