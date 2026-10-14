"""
P12.py — LangChain Output Parser: JSON Output

SUMMARY:
This program asks an Ollama model to extract contact details in JSON format.
JsonOutputParser parses the model response into Python data, usually a dictionary.
The program prints the full result and then accesses individual fields by key.

PREREQUISITES:
- Ollama running with phi3:mini downloaded.
- Install: pip install langchain-core langchain-ollama
"""

# Import LangChain's chat prompt template class.
from langchain_core.prompts import ChatPromptTemplate
# Import a parser that converts JSON text from the model into Python data.
from langchain_core.output_parsers import JsonOutputParser
# Import the Ollama chat-model integration.
from langchain_ollama import ChatOllama

# Choose the local model.
MODEL = "phi3:mini"
# Alternative model; uncomment only if installed locally.
# MODEL = "gemma2:2b"

# Create the model wrapper; temperature 0.0 aims for consistent output.
llm = ChatOllama(model=MODEL, temperature=0.0)
# Create the parser that will turn JSON-formatted text into a Python object.
parser = JsonOutputParser()

# Ask for three specific fields and restrict intent to the listed categories.
template = ChatPromptTemplate.from_messages([
    ("human", """Extract from the text and return as JSON with keys: name, email, intent.
intent must be one of: complaint, inquiry, feedback.
Return only the JSON object.

Text: {text}"""),
])

# Chain prompt formatting, model generation, and JSON parsing.
chain = template | llm | parser

# Execute the chain with a sample message that includes a name, email, and complaint.
result = chain.invoke({
    "text": "Hi, I'm Arjun (arjun@email.com). My order hasn't arrived after 2 weeks."
})

# Print the complete parsed result, typically a Python dictionary.
print(result)
# Read and print the value associated with the name key.
print(result["name"])
# Read and print the value associated with the intent key.
print(result["intent"])
