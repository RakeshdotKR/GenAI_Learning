"""
P9.py — Mini-Challenge: Customer Support Bot Template

SUMMARY:
This program builds a customer support prompt that accepts a query type and
customer message. The prompt is connected to a local Ollama model through LCEL.
A loop tests billing, technical, and general questions and prints each response.
The query type guides the model but does not strictly enforce classification.

PREREQUISITES:
- Ollama running with phi3:mini downloaded.
- Install: pip install langchain-core langchain-ollama
"""

# Import the chat prompt template class from LangChain.
from langchain_core.prompts import ChatPromptTemplate
# Import the Ollama chat-model integration.
from langchain_ollama import ChatOllama
# Optional Gemini alternative; uncomment and configure credentials to use it.
# from langchain_google_genai import ChatGoogleGenerativeAI

# Select the local model.
MODEL = "phi3:mini"
# Alternative local model; uncomment if downloaded.
# MODEL = "gemma2:2b"

# Create the local chat model wrapper with a moderate, relatively predictable temperature.
llm = ChatOllama(model=MODEL, temperature=0.3)

# Define the system instructions and the customer's message template.
template = ChatPromptTemplate.from_messages([
    # Explain how the assistant should handle each query category.
    ("system", """You are a support agent for Nova Phones.
Handle these query types:
- billing: explain charges clearly, offer to escalate if needed
- technical: give step-by-step troubleshooting instructions
- general: answer politely and briefly

Query type: {query_type}"""),
    # Insert the user's message into the human chat message.
    ("human", "{user_message}"),
])

# Chain prompt construction and model generation into one runnable pipeline.
chain = template | llm

# Prepare sample inputs covering the three supported query categories.
queries = [
    ("billing", "I was charged twice for my last bill."),       # Billing example.
    ("technical", "My phone won't connect to Wi-Fi."),          # Technical example.
    ("general", "What are your support hours?"),                 # General example.
]

# Process each query type and message in the sample list.
for query_type, message in queries:
    # Print a heading identifying the category being tested.
    print(f"\n--- {query_type.upper()} ---")
    # Fill the prompt placeholders and ask the model to generate a response.
    r = chain.invoke({"query_type": query_type, "user_message": message})
    # Print the generated response text.
    print(r.content)
