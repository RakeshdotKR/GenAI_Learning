"""
P14.py — Putting It Together: Template → LLM → Parser Pipeline

SUMMARY:
This example combines three stages into one LangChain pipeline: a chat prompt
template, an Ollama language model, and StrOutputParser. For each topic, the
program inserts the topic into the prompt, asks the model for a two-sentence
beginner explanation, converts the response into a plain string, and prints it.

PREREQUISITES:
- Ollama running with phi3:mini downloaded.
- Install: pip install langchain-core langchain-ollama
"""

# Import the class for building structured chat prompts.
from langchain_core.prompts import ChatPromptTemplate
# Import a parser that extracts the model's response as plain text.
from langchain_core.output_parsers import StrOutputParser
# Import LangChain's Ollama chat-model integration.
from langchain_ollama import ChatOllama
# Optional Gemini alternative; configure credentials before using it.
# from langchain_google_genai import ChatGoogleGenerativeAI

# Select the model available through the local Ollama service.
MODEL = "phi3:mini"
# Alternative local model; uncomment only if downloaded.
# MODEL = "gemma2:2b"

# Create the chat model wrapper and configure response variability.
llm = ChatOllama(model=MODEL, temperature=0.3)

# Build a pipeline with three sequential stages.
chain = (
    # First, format the system and human messages; {topic} is filled per iteration.
    ChatPromptTemplate.from_messages([
        ("system", "You are a concise technical writer."),  # Set assistant behavior.
        ("human", "Explain {topic} in exactly 2 sentences for a beginner."),  # Topic prompt.
    ])
    # Second, pass the formatted messages to the language model.
    | llm
    # Third, convert the model response object into a plain string.
    | StrOutputParser()
)

# Loop over three AI-related topics to reuse the same pipeline.
for topic in ["API", "RAG", "embeddings"]:
    # Print the current topic as a heading, with a preceding blank line.
    print(f"\n{topic}:")
    # Run the chain with the current topic and print its plain-text response.
    print(chain.invoke({"topic": topic}))
