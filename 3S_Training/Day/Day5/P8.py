"""
P8.py — RAG Chain with LangChain LCEL

SUMMARY:
This example implements Retrieval-Augmented Generation (RAG) with LangChain's
LCEL pipeline. It loads the persistent ChromaDB knowledge collection, retrieves
relevant documents for each question, joins them into context, formats a chat
prompt, calls Gemini, and converts the response into plain text.

PREREQUISITES:
- Build the ./rag_db/knowledge collection first by running P7.py.
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
# Import the class used to create structured chat prompts.
from langchain_core.prompts import ChatPromptTemplate
# Import a parser that converts model output into a plain string.
from langchain_core.output_parsers import StrOutputParser
# Import a helper that forwards its input unchanged through a chain.
from langchain_core.runnables import RunnablePassthrough

# Read key-value settings from the .env file into the process environment.
# Load API credentials such as GEMINI_API_KEY from the .env file.
load_dotenv()

# Load the persisted ChromaDB we built in 1.3
# Configure Gemini's embedding model to convert text into vectors.
embeddings  = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
# Reopen the persistent knowledge collection created by P2.py.
vectorstore = Chroma(collection_name="knowledge", persist_directory="./rag_db", embedding_function=embeddings)
# Create a retriever that returns the most relevant stored documents.
retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})

# Define the system instructions and human-message template for the model.
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using only the context provided. If unsure, say so."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

# Configure Gemini as the language model used to generate answers.
# Configure Gemini to generate answers with modest variation.
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.2)

# Define a function so this operation can be reused.
# Combine the page content of retrieved documents into a single context string.
def format_docs(docs):
    # Join retrieved Document objects into a single context string
    # Return the computed value to the code that called this function.
    return "\n\n".join(d.page_content for d in docs)

# {context} is filled by the retriever; {question} passes through unchanged
chain = (
    # Pass the original question through unchanged.
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    # Convert the model's response object into plain text.
    | StrOutputParser()
)

# Process each item in the collection one at a time.
for q in ["Who created Python?", "What is LangChain used for?"]:
    # Print the current result to the terminal.
    print(f"\nQ: {q}\nA: {chain.invoke(q)}")
