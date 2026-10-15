"""P5.py — Basic Streamlit Chat App

SUMMARY:
Builds a basic Streamlit chatbot backed by Gemini and LangChain conversation memory. 
Session state keeps the chain and visible chat history across reruns.

PREREQUISITES:
- Install: pip install streamlit langchain langchain-google-genai python-dotenv. 
Put GEMINI_API_KEY in .env. Classic ConversationChain and memory APIs may be deprecated in newer LangChain releases.
"""


import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Modern LangChain imports replacing deprecated legacy components
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables from the local .env file.
load_dotenv()

# Display the main page title in the Streamlit app.
st.title("My Chatbot")

# 1. Initialize conversational history backend in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()

# 2. Build the execution chain only once
if "conversational_chain" not in st.session_state:
    # Configure the LangChain integration for the Gemini model.
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.7)
    
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
        lambda session_id: st.session_state.chat_history, # Dynamic history target
        input_messages_key="input",
        history_messages_key="history",
    )

# 3. Render existing chat history in the Streamlit UI
# Loop through stored messages in InMemoryChatMessageHistory (converted to LangChain UI roles)
for msg in st.session_state.chat_history.messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

# 4. Handle new input
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Display the user's message immediately
    with st.chat_message("user"):
        st.write(user_input)

    # Fetch response from the modern chain (history updates implicitly)
    with st.spinner("Thinking..."):
        config = {"configurable": {"session_id": "streamlit_session"}}
        response = st.session_state.conversational_chain.invoke(
            {"input": user_input}, 
            config=config
        )
    
    # Display the assistant's reply
    with st.chat_message("assistant"):
        st.write(response.content)
