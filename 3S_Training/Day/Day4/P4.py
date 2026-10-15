"""P4.py — Wire Up Gemini + LangChain

SUMMARY:
Connects Gemini to LangChain ConversationChain and ConversationBufferMemory, then 
checks whether a second turn can recall details from the first.

PREREQUISITES:
- Install: pip install langchain langchain-google-genai python-dotenv. 
Put GEMINI_API_KEY in .env. 
"""


# Import the libraries and framework components required by this example.
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Modern LangChain imports replacing deprecated legacy components
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables from the local .env file.
load_dotenv()

# Configure the LangChain integration for the Gemini model.
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.7)

# Create a prompt with a placeholder for conversational history.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# Create the base processing chain using LCEL (Pipe operator)
chain = prompt | llm

# Dictionary to hold the in-memory chat histories for different sessions
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Wrap the runnable chain so it dynamically injects and updates conversation history
conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Test two turns — second reply will reference the first
# Provide a 'session_id' in the configuration to persist memory across calls
config = {"configurable": {"session_id": "ravi_session"}}

reply1 = conversational_chain.invoke({"input": "My name is Ravi and I like Python."}, config=config)
reply2 = conversational_chain.invoke({"input": "What did I just tell you?"}, config=config)

# Display values in the terminal (Extracting text content from the AI message object)
print("Turn 1:", reply1.content)
print("Turn 2:", reply2.content)

