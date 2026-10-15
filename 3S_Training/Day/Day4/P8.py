"""P8.py — Ollama Alternative — Offline Chatbot

SUMMARY:
Provides a local Ollama alternative to the Streamlit chatbot, using conversation memory without requiring a cloud API key.

PREREQUISITES:
- Install: pip install streamlit langchain langchain-ollama. Download phi3:mini with ollama pull phi3:mini and keep Ollama running. Classic ConversationChain and memory APIs may be deprecated in newer LangChain releases.
"""


# Streamlit reruns this script after interactions; session_state preserves values across reruns.
# Import the libraries and framework components required by this example.
# Import the named Python module so this script can use its functionality.
import streamlit as st
# Import the required class or function from this package.
from langchain_ollama import ChatOllama           # swap this line only
# Load environment variables from the local .env file (if required by setup).
from dotenv import load_dotenv

# Modern LangChain imports replacing deprecated legacy components
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables from the local .env file.
load_dotenv()

# Set the name of the local Ollama model to use.
OLLAMA_MODEL = "phi3:mini"
# OLLAMA_MODEL = "gemma2:2b"

# Display the main page title in the Streamlit app.
st.title(f"My Chatbot ({OLLAMA_MODEL})")

# 1. Initialize conversational history backend in session state
# Check this condition and run the indented block only when it is true.
if "chat_history" not in st.session_state:
    # Create the in-memory backend designed to securely hold message lists.
    st.session_state.chat_history = InMemoryChatMessageHistory()

# 2. Build the execution chain only once — store it in session_state so it persists
# Check this condition and run the indented block only when it is true.
if "conversational_chain" not in st.session_state:
    # Configure a LangChain chat model that sends prompts to the local Ollama service.
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.7)  # no API key needed
    
    # Create a prompt with a placeholder for conversational history.
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    
    # Create the base processing chain using LCEL (Pipe operator)
    base_chain = prompt | llm
    
    # Wrap the runnable chain so it dynamically retrieves and updates history
    st.session_state.conversational_chain = RunnableWithMessageHistory(
        base_chain,
        lambda session_id: st.session_state.chat_history,  # Dynamic history target
        input_messages_key="input",
        history_messages_key="history",
    )

# 3. Render existing chat history in the Streamlit UI with role-based formatting
# Loop through stored messages in InMemoryChatMessageHistory (converted to LangChain UI roles)
for msg in st.session_state.chat_history.messages:
    # Identify the sender role type for correct UI block icon matching.
    role = "user" if msg.type == "human" else "assistant"
    # Enter this context manager for the indented block.
    with st.chat_message(role):
        # Render the supplied text as Markdown in the Streamlit app.
        st.markdown(msg.content)

# 4. Handle new input
# Display the chat input and return the submitted message, or None if no message was submitted.
user_input = st.chat_input("Ask me anything...")

# Check this condition and run the indented block only when it is true.
if user_input and user_input.strip():
    # Enter this context manager for the indented block.
    with st.chat_message("user"):
        # Render the supplied text as Markdown in the Streamlit app.
        st.markdown(user_input)

    # Enter this context manager for the indented block.
    with st.spinner("Thinking..."):
        # Configure unique session configuration ID for tracing state context keys.
        config = {"configurable": {"session_id": "ollama_session"}}
        # Send the user input through the conversation chain and return the generated text.
        response = st.session_state.conversational_chain.invoke(
            {"input": user_input}, 
            config=config
        )
    
    # Enter this context manager for the indented block.
    with st.chat_message("assistant"):
        # Render the supplied text as Markdown in the Streamlit app.
        st.markdown(response.content)
