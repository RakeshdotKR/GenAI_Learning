"""P7.py — Add Role-Based Chat Formatting + Clear Button

SUMMARY:
Adds role-based chat rendering, a configurable persona, a clear-chat button, and basic error display to the Streamlit Gemini chatbot.

PREREQUISITES:
- Install: pip install streamlit langchain langchain-google-genai python-dotenv. Put GEMINI_API_KEY in .env. Classic ConversationChain and memory APIs may be deprecated in newer LangChain releases.
"""


# Streamlit reruns this script after interactions; session_state preserves values across reruns.
# Import the libraries and framework components required by this example.
# Import the named Python module so this script can use its functionality.
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
        "Persona / system prompt:",
        value="You are a helpful and friendly assistant.",
        height=120,
    )
    
    # Rebuild the chain whenever the system prompt changes
    if st.button("Apply & Reset Chat"):
        for key in ["conversational_chain", "chat_history"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    # Clear chat without changing the persona
    if st.button("Clear Chat"):
        if "chat_history" in st.session_state:
            del st.session_state["chat_history"]
        st.rerun()

    st.divider()
    st.caption("Model: gemini-3.8-flash")

# 1. Initialize conversational history backend in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()

# 2. Build the execution chain only once (or whenever reset by the sidebar)
if "conversational_chain" not in st.session_state:
    # Configure the LangChain integration for the Gemini model.
    # Note: Updated to stable 'gemini-2.5-flash' since 'gemini-3.8-flash' does not exist.
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

# 3. Render existing chat history in the Streamlit UI with role-based formatting
for msg in st.session_state.chat_history.messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# 4. Handle new input
user_input = st.chat_input("Type a message...")

if user_input:
    # Reject empty inputs or whitespace strings
    if not user_input.strip():
        st.warning("Please type a message first.")
    else:
        # Display the user's message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # Fetch response from the modern chain (history updates implicitly inside invoke)
        with st.spinner("Generating response..."):
            try:
                config = {"configurable": {"session_id": "streamlit_session"}}
                response = st.session_state.conversational_chain.invoke(
                    {"input": user_input}, 
                    config=config
                )
                reply = response.content
            except Exception as e:
                # Surface API/runtime errors in the UI, and log to history
                reply = f"Error: {e}"
                # Manually sync the conversation fallback when invoke fails midway
                st.session_state.chat_history.add_user_message(user_input)
                st.session_state.chat_history.add_ai_message(reply)

        # Display the assistant's reply if it was generated successfully or caught an error
        # (Skip rendering if the engine already appended the response inside the try block)
        if not any(msg.content == reply for msg in st.session_state.chat_history.messages[-1:]):
            with st.chat_message("assistant"):
                st.markdown(reply)
        else:
            with st.chat_message("assistant"):
                st.markdown(reply)
