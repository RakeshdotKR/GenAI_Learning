"""P6.py — Add System Prompt via Sidebar

SUMMARY:
Adds a sidebar persona editor to a Streamlit Gemini chatbot. Applying a new system prompt clears the old chain and chat so a new chain is built with the updated prompt.

PREREQUISITES:
- Install: pip install streamlit langchain langchain-google-genai python-dotenv. Put GEMINI_API_KEY in .env. Classic ConversationChain and memory APIs may be deprecated in newer LangChain releases.
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

# --- Sidebar - Persona & Controls ---
with st.sidebar:
    st.header("Bot Settings")
    system_prompt = st.text_area(
        "System prompt (persona):",
        value="You are a helpful and friendly assistant.",
        height=100,
    )
    
    # Rebuild the chain whenever the system prompt changes
    if st.button("Apply & Reset Chat"):
        # Clear legacy and updated state keys to rebuild fresh
        for key in ["conversational_chain", "chat_history"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()  # trigger an immediate rerun

# 1. Initialize conversational history backend in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()

# 2. Build the execution chain only once (or whenever reset by the sidebar)
if "conversational_chain" not in st.session_state:
    # Configure the LangChain integration for the Gemini model.
    # Note: Using gemini-2.5-flash as gemini-3.8-flash does not exist.
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.7)
    
    # Create a prompt using the custom system prompt from the sidebar
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
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

# 3. Render existing chat history in the Streamlit UI
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
