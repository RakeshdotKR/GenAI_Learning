"""
P6.py — End-to-End RAG Pipeline: Load → Chunk → Embed → Store → Query

SUMMARY:
Section A: full pipeline on a text knowledge base — creates knowledge.txt,
chunks it, embeds with Gemini gemini-embedding-001, stores in ChromaDB, then
answers three questions using the stored vectors.
Section B: same pipeline for a PDF — swap PDF_PATH for a real file to run.
Both sections use an in-memory ChromaDB client (no disk persistence needed
for demonstration).

PREREQUISITES:
- A valid GEMINI_API_KEY in a local .env file.
- pip install google-genai langchain-community langchain-text-splitters chromadb pypdf python-dotenv
"""

import os
from dotenv import load_dotenv
from google import genai
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ── Section A: text knowledge base ───────────────────────────────────────────

# Step 1: write a sample knowledge base file
with open("knowledge.txt", "w", encoding="utf-8") as f:
    f.write("""Python is a high-level, general-purpose programming language.
It supports multiple paradigms including object-oriented and functional programming.
Python was created by Guido van Rossum and released in 1991.
The language is widely used in web development, data science, and artificial intelligence.

LangChain is a framework for building LLM-powered applications.
It provides abstractions for prompts, chains, memory, and tools.
LangChain supports integration with many LLM providers including Gemini and Ollama.
The LCEL pipe syntax makes it easy to compose chains.

RAG stands for Retrieval Augmented Generation.
RAG solves the problem of LLMs not knowing about your private documents.
The retriever fetches relevant chunks; the LLM generates an answer from them.
ChromaDB is a popular vector database used in RAG pipelines.

Embeddings are numerical representations of text.
Similar sentences produce similar embedding vectors.
Cosine similarity measures how close two embedding vectors are.
The Gemini gemini-embedding-001 model produces dense float vectors.
""")

# Step 2: load and chunk
raw_docs = TextLoader("knowledge.txt", encoding="utf-8").load()
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
chunks   = splitter.split_documents(raw_docs)
print(f"[A] Chunks created: {len(chunks)}")

# Step 3: embed all chunks with Gemini (FIXED: updated model)
chunk_texts = [c.page_content for c in chunks]
result      = client.models.embed_content(model="gemini-embedding-001", contents=chunk_texts)
embeddings  = [e.values for e in result.embeddings]
print(f"[A] Embeddings: {len(embeddings)} × {len(embeddings[0])} dims")

# Step 4: store in ChromaDB
db_client  = chromadb.Client()
collection = db_client.create_collection("knowledge_base")
collection.add(
    documents=chunk_texts,
    embeddings=embeddings,
    metadatas=[c.metadata for c in chunks],
    ids=[f"chunk_{i}" for i in range(len(chunks))],
)
print(f"[A] Stored {collection.count()} chunks")

# Step 5: query — embed question, search collection, print top matches
def search(question: str, top_k: int = 3):
    # FIXED: updated model and fixed return mapping array hierarchy
    q_emb    = client.models.embed_content(model="gemini-embedding-001", contents=[question]).embeddings[0].values
    results  = collection.query(query_embeddings=[q_emb], n_results=top_k)
    return results["documents"][0], results["distances"][0]

for q in ["Who created Python?", "What is RAG used for?", "How do embeddings represent text?"]:
    docs, dists = search(q)
    print(f"\n[A] Q: {q}")
    for doc, dist in zip(docs, dists):
        print(f"  [{dist:.4f}] {doc[:80]}")

# ── Section B: PDF pipeline ───────────────────────────────────────────────────

PDF_PATH = r"D:\Training\PESCE_Mandya\GenAI_Learning\3S_Training\me\Day4_new\python_and_langchain_overview.pdf"   # update to a real PDF path before running

if os.path.exists(PDF_PATH):
    pages     = PyPDFLoader(PDF_PATH).load()
    pdf_chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(pages)
    print(f"\n[B] PDF pages: {len(pages)}, chunks: {len(pdf_chunks)}")

    # Skip empty pages (scanned PDFs sometimes produce blank pages)
    texts      = [c.page_content for c in pdf_chunks if c.page_content.strip()]
    # FIXED: updated model
    pdf_result = client.models.embed_content(model="gemini-embedding-001", contents=texts)
    pdf_embeds = [e.values for e in pdf_result.embeddings]

    pdf_col = db_client.create_collection("pdf_search")
    pdf_col.add(
        documents=texts,
        embeddings=pdf_embeds,
        metadatas=[{"page": pdf_chunks[i].metadata.get("page", 0)} for i in range(len(texts))],
        ids=[f"c{i}" for i in range(len(texts))],
    )
    print(f"[B] Stored {pdf_col.count()} PDF chunks")

    for question in [
        "What is the main topic of this document?",
        "What are the key concepts covered?",
    ]:
        # FIXED: updated model and vector extraction array location
        q_vec   = client.models.embed_content(model="gemini-embedding-001", contents=[question]).embeddings[0].values
        results = pdf_col.query(query_embeddings=[q_vec], n_results=2)
        print(f"\n[B] Q: {question}")
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            print(f"  [page {meta['page']}, dist={dist:.4f}] {doc[:80]}")
else:
    print(f"\n[B] Skipped — {PDF_PATH!r} not found. Update PDF_PATH to run the PDF section.")
