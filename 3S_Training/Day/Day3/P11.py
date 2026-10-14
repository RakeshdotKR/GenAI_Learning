"""
P11.py — LangChain Output Parser: String Output

SUMMARY:
This program connects a chat prompt, an Ollama model, and StrOutputParser in a
LangChain pipeline. The parser converts the model's message output into a plain
Python string, so the result can be printed or used without accessing .content.

PREREQUISITES:
- Ollama running with phi3:mini downloaded.
- Install: pip install langchain-core langchain-ollama
"""

# Import the class used to define chat prompts with placeholders.
from langchain_core.prompts import ChatPromptTemplate
# Import a parser that extracts plain text from a model response.
from langchain_core.output_parsers import StrOutputParser
# Import the integration for a local Ollama chat model.
from langchain_ollama import ChatOllama
# Optional Gemini alternative; configure credentials before using it.
# from langchain_google_genai import ChatGoogleGenerativeAI

# Select the local model.
MODEL = "phi3:mini"
# Alternative model; uncomment if available in Ollama.
# MODEL = "gemma2:2b"

# Create the model wrapper and set response variability.
llm = ChatOllama(model=MODEL, temperature=0.3)

# Define a human message whose {text} placeholder will contain the source passage.
template = ChatPromptTemplate.from_messages([
    ("human", "Summarise this in one sentence: {text}"),
])

# Build a pipeline: format prompt, generate model response, extract plain text.
chain = template | llm | StrOutputParser()

# Run the pipeline with the passage to summarize and store the resulting string.
result = chain.invoke({"text": "LangChain is a framework for building applications powered by large language models. It provides tools for chaining prompts, managing memory, and connecting to external data sources."})
# Print the plain-string summary; there is no .content attribute on this string.
print(result)
# Print the Python type to demonstrate that the parser returned str.
print(type(result))
