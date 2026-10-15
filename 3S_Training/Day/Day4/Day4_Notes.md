### Step 1: Ensure virtual env is activated (terminal of vs code)

## [IF NOT ALREADY DONE] To create virtual environment
python -m venv .venv #win
python3 -m venv .venv #mac

## To Activate virtual environment
.venv\Scripts\activate.bat      #windows
source .venv/bin/activate       #Mac

## To deactivate
deactivate

======================================================================================

## Install packages we need for today
*pip install google-genai chromadb langchain-chroma langchain-text-splitters pypdf python-dotenv*

## Run below in cmd prompt
*ollama pull nomic-embed-text*

Install dependencies exactly inside the virtual environment:
"d:\Your_Repository_Folder\.venv\Scripts\python.exe" -m pip install package-name*

=======================================================================================


##Day 4 Notes:
====================================================================
Streamlit usage
====================================================================
Topic: P1.py — Session State — Persisting Data Across Reruns

Topic: P2.py — Chat Components — st.chat_message + st.chat_input

Topic: P3.py — Mini-Challenge — Styled Input/Output Page

Topic: P4.py — Link Gemini + LangChain

Topic: P5.py — Basic Streamlit Chat App

Topic: P6.py — Add System Prompt via Sidebar

Topic: P7.py — Add Role-Based Chat Formatting + Clear Button

Topic: P8.py — Ollama Alternative — Offline Chatbot

===============================================================
RAG: Basics
===============================================================
P9.py — Generating Embeddings with Gemini

P10.py - generates embedding vectors for text using an embedding model

P11.py — Ollama based embedding

P12.py — Cosine Similarity — Measuring Meaning Distance

P13.py — ChromaDB In-Memory CRUD: Add, Query, Update, Delete