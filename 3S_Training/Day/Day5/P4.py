"""
P4.py — LangChain Document Loaders: TextLoader, PyPDFLoader, DirectoryLoader

SUMMARY:
Demonstrates three LangChain loaders that convert source files into Document
objects (page_content + metadata) ready for chunking and embedding.
Section A: TextLoader — a single plain-text file.
Section B: PyPDFLoader — a PDF file, one Document per page.
Section C: DirectoryLoader — all .txt files in a folder.

PREREQUISITES:
- pip install langchain-community pypdf
"""

import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader

# ── Section A: TextLoader ─────────────────────────────────────────────────────

# Create a sample file to load
with open("sample.txt", "w", encoding="utf-8") as f:
    f.write("""Python is a versatile programming language.
It was created by Guido van Rossum and released in 1991.
Python is now one of the most popular languages in the world.
It is used extensively in data science and machine learning.
LangChain is built on top of Python to simplify LLM development.
""")

loader = TextLoader("sample.txt", encoding="utf-8")
docs   = loader.load()  # returns a list of Document objects

print(f"[TextLoader] Documents: {len(docs)}, chars: {len(docs[0].page_content)}")
print(f"  Metadata : {docs[0].metadata}")
print(f"  Preview  : {docs[0].page_content[:80]}")

# ── Section B: PyPDFLoader ────────────────────────────────────────────────────

PDF_PATH = r"D:\Training\PESCE_Mandya\GenAI_Learning\3S_Training\me\Day4_new\python_and_langchain_overview.pdf"  # replace with an actual PDF path before running

if os.path.exists(PDF_PATH):
    loader = PyPDFLoader(PDF_PATH)
    pages  = loader.load()  # one Document per page; metadata includes 'page' number
    print(f"\n[PyPDFLoader] Total pages: {len(pages)}")
    for i, page in enumerate(pages[:3]):
        print(f"  Page {i + 1}: {len(page.page_content)} chars | meta: {page.metadata}")
        print(f"    {page.page_content[:120]}")
else:
    print(f"\n[PyPDFLoader] Skipped — {PDF_PATH!r} not found. Update PDF_PATH to run.")

# ── Section C: DirectoryLoader ────────────────────────────────────────────────

# Create a small folder of text files to demonstrate directory loading
os.makedirs("docs_folder", exist_ok=True)
for name, content in [
    ("python.txt",    "Python is a high-level language used in AI and web."),
    ("langchain.txt", "LangChain builds LLM applications with chains and memory."),
    ("rag.txt",       "RAG stands for Retrieval Augmented Generation."),
]:
    with open(f"docs_folder/{name}", "w", encoding="utf-8") as f:
        f.write(content)

loader = DirectoryLoader(
    "docs_folder",
    glob="**/*.txt",          # match all .txt files recursively
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)
docs = loader.load()

print(f"\n[DirectoryLoader] Loaded {len(docs)} documents")
for d in docs:
    print(f"  {d.metadata['source']} — {len(d.page_content)} chars")
