"""
P6.py — LangChain ChatPromptTemplate with Ollama

SUMMARY:
This program combines a ChatPromptTemplate with a locally running Ollama model.
The template receives company and question values, then the LangChain Expression
Language (LCEL) pipe sends the formatted messages to the model. The response text
is printed to the terminal.

PREREQUISITES:
- Ollama must be running and the selected model downloaded (ollama pull phi3:mini).
- Install dependencies: pip install langchain-core langchain-ollama
"""

# Import the LangChain wrapper for chat models served by Ollama.
from langchain_ollama import ChatOllama
# Import the class used to build structured chat prompts with placeholders.
from langchain_core.prompts import ChatPromptTemplate

# Choose the local Ollama model to use.
MODEL = "phi3:mini"
# Alternative model; uncomment only if it has been downloaded in Ollama.
# MODEL = "gemma2:2b"

# Configure the model wrapper; temperature controls response variability.
llm = ChatOllama(model=MODEL, temperature=0.3)

# Create a prompt with a system message and a human message.
# {company} and {question} are filled when the chain is invoked.
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant for {company}."),  # Assistant role/context.
    ("human", "{question}"),                                  # User's question.
])

# Connect the prompt to the model using LCEL's pipe operator.
# Input flows through the template first and then to the language model.
chain = template | llm

# Fill the placeholders, run the chain, and store the model's response.
response = chain.invoke({"company": "Nova Phones", "question": "How do I reset my phone?"})
# Print the generated text from the model response object.
print(response.content)
