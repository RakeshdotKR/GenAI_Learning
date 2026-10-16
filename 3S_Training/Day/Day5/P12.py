"""
P12.py — Conversational RAG — Remember Previous Questions

SUMMARY:
This example adds conversation history to a RAG chatbot. A first chain rewrites
follow-up questions into standalone questions using the chat history; a second
chain retrieves relevant documents and generates an answer from their context.
The conversation's user and assistant messages are saved for later turns.

PREREQUISITES:
- A compatible ChromaDB collection must already exist at ./chroma.
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
# Import chat prompt templates and a placeholder for prior conversation messages.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# Import a parser that converts model output into a plain string.
from langchain_core.output_parsers import StrOutputParser
# Import a helper that forwards its input unchanged through a chain.
from langchain_core.runnables import RunnablePassthrough
# Import message classes for storing user and assistant conversation turns.
from langchain_core.messages import HumanMessage, AIMessage

# Read key-value settings from the .env file into the process environment.
# Load API credentials such as GEMINI_API_KEY from the .env file.
load_dotenv()

# Configure Gemini's embedding model to convert text into vectors.
embeddings  = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
# Connect to the existing default Chroma persistence directory.
vectorstore = Chroma(embedding_function=embeddings)
# Create a retriever that returns the most relevant stored documents.
retriever   = vectorstore.as_retriever(search_kwargs={"k": 2})
# Configure Gemini as the language model used to generate answers.
llm         = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.2)

# Chain 1: rewrite the follow-up as a self-contained question
rephrase_chain = (
    # Define the system instructions and human-message template for the model.
    ChatPromptTemplate.from_messages([
        ("system", "Rewrite the follow-up as a standalone question. Return only the question."),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])
    # Convert the model's response object into plain text.
    | llm | StrOutputParser()
)

# Chain 2: standard RAG answer
# Define a function so this operation can be reused.
# Combine the page content of retrieved documents into a single context string.
def format_docs(docs):
    # Return the computed value to the code that called this function.
    return "\n\n".join(d.page_content for d in docs)

answer_chain = (
    # Pass the original question through unchanged.
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    # Define the system instructions and human-message template for the model.
    | ChatPromptTemplate.from_messages([
        ("system", "Answer using only the context. Be concise."),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    # Convert the model's response object into plain text.
    | llm | StrOutputParser()
)

# Keep conversation messages in order so later questions can refer to earlier turns.
history = []

# Define a function so this operation can be reused.
# Rewrite a follow-up question, retrieve context, generate an answer, and save the turn.
def chat(question: str) -> str:
    # Execute the chain or retriever using the current input.
    standalone = rephrase_chain.invoke({"history": history, "question": question})
    # Execute the chain or retriever using the current input.
    answer     = answer_chain.invoke(standalone)
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))
    # Return the computed value to the code that called this function.
    return answer

# Print the current result to the terminal.
print(chat("Tell me about Python."))
# Print the current result to the terminal.
print(chat("Who created it?"))           # "it" → rephrased to "Who created Python?"
# Print the current result to the terminal.
print(chat("When was it first released?"))
