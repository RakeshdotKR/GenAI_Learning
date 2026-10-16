"""
P2.py — LangChain + ChromaDB: similarity_search and as_retriever

SUMMARY:
Wraps ChromaDB as a LangChain VectorStore using Chroma.from_documents().
Section A: uses similarity_search() to retrieve matching Document objects.
Section B: wraps the vectorstore in a retriever via as_retriever() — the
interface used inside LangChain RAG chains 

PREREQUISITES:
- pip install langchain-chroma langchain-core
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document

# ── Section A: similarity_search ─────────────────────────────────────────────

docs_a = [
    Document(page_content="Python is a general-purpose programming language.", metadata={"source": "python.txt"}),
    Document(page_content="LangChain simplifies building LLM-powered apps.",   metadata={"source": "lc.txt"}),
    Document(page_content="ChromaDB is a local open-source vector database.",  metadata={"source": "chroma.txt"}),
    Document(page_content="RAG retrieves relevant chunks before generating.",  metadata={"source": "rag.txt"}),
    Document(page_content="Embeddings map text to high-dimensional vectors.",  metadata={"source": "embed.txt"}),
]

# In-memory vectorstore using ChromaDB's default built-in embeddings
vectorstore_a = Chroma.from_documents(docs_a)

# Returns a list of Document objects ranked by relevance
results = vectorstore_a.similarity_search("what is a vector database?", k=2)
print("similarity_search results:")
for r in results:
    print(f"  {r.page_content} | {r.metadata}")

# ── Section B: as_retriever (preview of Day 5 RAG chains) ────────────────────

docs_b = [
    Document(page_content="Python was created by Guido van Rossum in 1991.",                 metadata={"source": "history.txt"}),
    Document(page_content="Python 3 introduced significant improvements over Python 2.",     metadata={"source": "history.txt"}),
    Document(page_content="Django is a Python web framework for rapid development.",         metadata={"source": "web.txt"}),
    Document(page_content="FastAPI is a modern, high-performance Python web framework.",     metadata={"source": "web.txt"}),
    Document(page_content="NumPy provides fast numerical operations on arrays.",             metadata={"source": "data.txt"}),
]

vectorstore_b = Chroma.from_documents(docs_b)

# as_retriever() wraps the vectorstore for use in LangChain chains
retriever = vectorstore_b.as_retriever(search_kwargs={"k": 2})

retrieved = retriever.invoke("which Python web frameworks exist?")
print(f"\nas_retriever results ({len(retrieved)} documents):")
for doc in retrieved:
    print(f"  [{doc.metadata['source']}] {doc.page_content}")
