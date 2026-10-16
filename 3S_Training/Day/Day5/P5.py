"""
P5.py — RecursiveCharacterTextSplitter: plain text, Documents, size comparison

SUMMARY:
Demonstrates three uses of RecursiveCharacterTextSplitter.
Section A: split a plain string and inspect each chunk.
Section B: split LangChain Document objects — metadata is preserved per chunk.
Section C: compare chunk counts and average sizes across different settings.

PREREQUISITES:
- pip install langchain-text-splitters langchain-core

"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ── Section A: split a plain text string ──────────────────────────────────────

text = """Python is a high-level, general-purpose programming language. Its design
philosophy emphasizes code readability with significant indentation. Python is
dynamically typed and garbage-collected.

Python was created by Guido van Rossum and first released in 1991. Python
consistently ranks as one of the most popular programming languages. It is used
in web development, data science, machine learning, and automation.

LangChain is a framework designed to simplify building applications using large
language models. It supports document loaders, text splitters, vector stores,
retrievers, and chains."""

# chunk_size: max chars per chunk; chunk_overlap: chars repeated between neighbours
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
chunks   = splitter.split_text(text)

print(f"[A] {len(text)} chars → {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    print(f"\n  [{i}] ({len(chunk)} chars)")
    print(f"  {chunk}")

# ── Section B: split Document objects (metadata preserved) ────────────────────

raw_docs = [
    Document(
        page_content="""Python is a versatile programming language.
It is used in data science, web development, AI, and automation.
Python supports OOP, functional, and procedural programming styles.
The language was designed for readability and simplicity.""",
        metadata={"source": "python.txt"},
    ),
    Document(
        page_content="""LangChain is a framework for building LLM-powered applications.
It provides document loaders, text splitters, and vector store integrations.
LangChain chains connect prompts, models, and output parsers using LCEL.
Memory components allow LLMs to remember previous conversation turns.""",
        metadata={"source": "langchain.txt"},
    ),
]

splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
doc_chunks = splitter.split_documents(raw_docs)

print(f"\n[B] {len(raw_docs)} docs → {len(doc_chunks)} chunks (metadata retained)")
for chunk in doc_chunks:
    print(f"  [{chunk.metadata['source']}] ({len(chunk.page_content)} chars) {chunk.page_content[:60]}...")

# ── Section C: compare chunk counts across different sizes ────────────────────

print("\n[C] Size comparison on sample.txt")
try:
    text_c = open("sample.txt", encoding="utf-8").read()
    for size, overlap in [(500, 50), (200, 30), (100, 15)]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
        chunks_c = splitter.split_text(text_c)
        avg = sum(len(c) for c in chunks_c) // len(chunks_c) if chunks_c else 0
        print(f"  chunk_size={size:>4}, overlap={overlap:>3} → {len(chunks_c):>3} chunks, avg {avg} chars")
except FileNotFoundError:
    print("  sample.txt not found")
