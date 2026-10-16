"""
P1.py — ChromaDB: Metadata Filtering + Persistent Storage

SUMMARY:
Section A: filter search results using metadata (the where= parameter) so
queries only scan a specific subset of documents.
Section B: use PersistentClient to save a collection to disk and reload it
in a second run — data survives process restarts.

PREREQUISITES:
- pip install chromadb
"""

import chromadb

# ── Section A: metadata filtering ────────────────────────────────────────────

client = chromadb.Client()
col = client.create_collection("filtered_search")

col.add(
    documents=[
        "Python basics: variables and functions.",
        "Python advanced: decorators and generators.",
        "JavaScript: let, const, and arrow functions.",
        "JavaScript: async/await and promises.",
        "SQL: SELECT, WHERE, and JOIN.",
    ],
    metadatas=[
        {"language": "python",     "level": "beginner"},
        {"language": "python",     "level": "advanced"},
        {"language": "javascript", "level": "beginner"},
        {"language": "javascript", "level": "advanced"},
        {"language": "sql",        "level": "beginner"},
    ],
    ids=["p1", "p2", "js1", "js2", "sql1"],
)

# where= restricts the search to documents matching the metadata condition
results = col.query(
    query_texts=["how to write reusable code?"],
    n_results=2,
    where={"language": "python"},
)

print("Python-only results:")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [{meta['level']}] {doc}")

# ── Section B: persistent storage ────────────────────────────────────────────

DB_PATH = "./my_chroma_db"

# PersistentClient writes data to disk — survives process restarts
persistent_client = chromadb.PersistentClient(path=DB_PATH)

# get_or_create_collection is safe to call on every run
kb = persistent_client.get_or_create_collection("knowledge_base")

kb.add(
    documents=[
        "The Python GIL prevents true multi-threading for CPU-bound tasks.",
        "Use asyncio for I/O-bound concurrency in Python.",
        "Multiprocessing bypasses the GIL by using separate processes.",
    ],
    ids=["gil", "asyncio", "multiprocessing"],
)

print(f"\nSaved {kb.count()} chunks to {DB_PATH}")
print("Restart the script and uncomment the reload block below to verify persistence.")

# --- Reload block: uncomment on a second run to confirm data survived ---
# client2 = chromadb.PersistentClient(path=DB_PATH)
# col2    = client2.get_collection("knowledge_base")
# print("Reloaded. Count:", col2.count())
# results = col2.query(query_texts=["how to handle concurrency?"], n_results=2)
# print(results["documents"])
