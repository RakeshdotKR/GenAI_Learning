"""P2.py — Chat Components — st.chat_message + st.chat_input

SUMMARY:
Demonstrates Streamlit chat components and session state by showing prior messages and echoing each new user message as a placeholder assistant reply.

PREREQUISITES:
- Install: pip install streamlit. Run with streamlit run P2.py.
"""


# Streamlit reruns this script after interactions; session_state preserves values across reruns.
# Import the libraries and framework components required by this example.
# Import the named Python module so this script can use its functionality.
import streamlit as st

# Display the main page title in the Streamlit app.
st.title("Chat UI Demo")

# Initialise message history in session_state
# Check this condition and run the indented block only when it is true.
if "messages" not in st.session_state:
    # Assign the computed value or object to this variable.
    st.session_state.messages = []

# Render all past messages from history on every rerun
# Loop over each item in the supplied sequence.
for msg in st.session_state.messages:
    # Enter this context manager for the indented block.
    with st.chat_message(msg["role"]):   # role = "user" or "assistant"
        # Display the supplied text or value in the Streamlit app.
        st.write(msg["content"])

# Chat input box — fixed at the bottom of the page
# Display the chat input and return the submitted message, or None if no message was submitted.
user_input = st.chat_input("Type a message...")

# Check this condition and run the indented block only when it is true.
if user_input:
    # Save and immediately display user message
    # Add this message to the chat history stored for the current browser session.
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Enter this context manager for the indented block.
    with st.chat_message("user"):
        # Display the supplied text or value in the Streamlit app.
        st.write(user_input)

    # Placeholder echo response (we'll replace with LLM in Session 3)
    # Store the assistant reply generated for the current user input.
    reply = f"Echo: {user_input}"
    # Add this message to the chat history stored for the current browser session.
    st.session_state.messages.append({"role": "assistant", "content": reply})
    # Enter this context manager for the indented block.
    with st.chat_message("assistant"):
        # Display the supplied text or value in the Streamlit app.
        st.write(reply)
