"""P16.py — ChatMessageHistory with LCEL — keeps full history

SUMMARY:
Demonstrates session-based chat history using modern LangChain 
Expression Language (LCEL), keeping earlier exchanges available to later turns.

PREREQUISITES:
- Install: pip install langchain-ollama langchain-core
- Ollama must be running with phi3:mini downloaded.
"""

# Import the modern framework components required by this example.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_ollama import ChatOllama

# Set the name of the local Ollama model to use.
OLLAMA_MODEL = "phi3:mini"

# Configure a LangChain chat model that sends prompts to the local Ollama service.
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.5)

# Create a modern prompt template that structures the chat and holds history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant having a conversation with a human."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# Combine prompt and language model into a clean runnable chain
chain = prompt | llm

# Dictionary to store our chat histories in-memory using session IDs
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Helper function to fetch or initialize history for a specific session."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Wrap our chain with message history management capabilities
conversation_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Configuration dictionary to specify which chat session we are interacting with
config = {"configurable": {"session_id": "ravi_session"}}

# 1. First interaction: Provide a name
response1 = conversation_with_history.invoke({"input": "My name is Ravi."}, config=config)
print(f"AI: {response1.content}\n")

# 2. Second interaction: Ask a general question
response2 = conversation_with_history.invoke({"input": "What is Python?"}, config=config)
print(f"AI: {response2.content}\n")

# 3. Third interaction: Test the memory
response3 = conversation_with_history.invoke({"input": "What did I tell you my name was?"}, config=config)
print(f"AI: {response3.content}\n")
