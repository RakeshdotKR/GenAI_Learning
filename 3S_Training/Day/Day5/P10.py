"""
P5.py — RAG with Source Citations

SUMMARY:
This example demonstrates RAG answers with source citations. It attaches source
file and page metadata to each document, embeds the documents into ChromaDB,
retrieves relevant context for a question, and returns both the generated answer
and the retrieved documents so their source metadata can be printed.

PREREQUISITES:
- Configure GEMINI_API_KEY in a .env file.
- Install: pip install python-dotenv langchain-chroma langchain-google-genai langchain-core
"""

# Access environment variables and operating-system functions.
import os
# Load API credentials and settings from a .env file.
from dotenv import load_dotenv
# Import LangChain's integration with the Chroma vector database.
from langchain_chroma import Chroma
# Import Gemini chat-model and text-embedding integrations for LangChain.
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# Import the standard LangChain document container with text and metadata.
from langchain_core.documents import Document
# Import the class used to create structured chat prompts.
from langchain_core.prompts import ChatPromptTemplate
# Import a parser that converts model output into a plain string.
from langchain_core.output_parsers import StrOutputParser
# Import pipeline helpers to pass input through and run multiple branches.
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Read key-value settings from the .env file into the process environment.
# Load API credentials such as GEMINI_API_KEY from the .env file.
load_dotenv()

# Each Document carries a 'source' and 'page' so we can cite it later
# Create documents with text plus source/page metadata for later citations.
docs = [
    Document(page_content="Python was created by Guido van Rossum in 1991.",           metadata={"source": "python_history.txt", "page": 1}),
    Document(page_content="Python 3 introduced breaking changes to fix design flaws.",  metadata={"source": "python_history.txt", "page": 2}),
    Document(page_content="LangChain chains connect prompts, LLMs, and parsers.",       metadata={"source": "langchain_intro.txt", "page": 1}),
    Document(page_content="LangChain memory stores conversation history between turns.", metadata={"source": "langchain_intro.txt", "page": 2}),
    Document(page_content="RAG retrieves relevant chunks before the LLM answers.",      metadata={"source": "rag_guide.txt",       "page": 1}),
    Document(page_content="ChromaDB is an open-source local vector database.",          metadata={"source": "rag_guide.txt",       "page": 2}),
]

# Configure Gemini's embedding model to convert text into vectors.
embeddings  = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
# Create a Chroma vector store from LangChain Document objects and their embeddings.
vectorstore = Chroma.from_documents(docs, embedding=embeddings)
# Create a retriever that returns the most relevant stored documents.
retriever   = vectorstore.as_retriever(search_kwargs={"k": 2})

# Define the system instructions and human-message template for the model.
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using only the context. Be concise."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])
# Configure Gemini as the language model used to generate answers.
# Configure Gemini with a low temperature for focused answers.
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.1)

# Define a function so this operation can be reused.
# Combine the page content of retrieved documents into a single context string.
def format_docs(docs):
    # Return the computed value to the code that called this function.
    return "\n\n".join(d.page_content for d in docs)

# RunnableParallel runs both branches simultaneously:
# 'answer'  → the answer string
# 'sources' → the raw Document objects so we can read their metadata
# Run multiple chain branches and collect their results.
chain = RunnableParallel(
    # Pass the original question through unchanged.
    answer={"context": retriever | format_docs, "question": RunnablePassthrough()}
           # Convert the model's response object into plain text.
           | prompt | llm | StrOutputParser(),
    sources=retriever,
)

# Define a function so this operation can be reused.
# Answer a question and print both the response and its source references.
def ask(question: str):
    # Execute the chain or retriever using the current input.
    result = chain.invoke(question)
    # Print the current result to the terminal.
    print(f"\nQ: {question}")
    # Print the current result to the terminal.
    print(f"A: {result['answer']}")
    # Track source references already printed to avoid duplicate citations.
    seen = set()
    # Process each item in the collection one at a time.
    for doc in result["sources"]:
        ref = f"{doc.metadata['source']} (page {doc.metadata.get('page', '?')})"
        # Run the following block only when this condition is true.
        if ref not in seen:
            # Print the current result to the terminal.
            print(f"  • {ref}")
            # Store each text chunk with its corresponding embedding and ID.
            seen.add(ref)

ask("Who created Python?")
ask("What is RAG?")
ask("What does LangChain memory do?")
