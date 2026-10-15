"""
P10.py — embed_single.py

SUMMARY:
This example generates embedding vectors for text using an embedding model.
The resulting numeric vectors can be inspected, compared for semantic
similarity, or used by a retrieval system.

PREREQUISITES:
- A valid GEMINI_API_KEY configured in a local .env file, where required.
"""

# Import operating-system utilities, including environment and file-path access.
import os
# Load helpers for reading environment variables from a .env file.
from dotenv import load_dotenv
# Import the Google Gemini SDK used to call embedding or language-model services.
from google import genai

# Load environment variables from the local .env file.
load_dotenv()

# Create a Gemini client using the API key stored in the environment.
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Define the input text, documents, or file path used by this example.
sentences = [
    "Python is a programming language.",
    "A snake is a reptile.",
    "Machine learning uses data to train models.",
    "Deep learning is a subset of machine learning.",
    "The sun rises in the east.",
]

# Request embedding vectors for the supplied text using the updated Gemini embedding model.
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=sentences,
)

# Display the result or status information in the terminal.
print(f"Got {len(result.embeddings)} embeddings")
# Repeat the indented block for each item in the iterable.
for i, emb in enumerate(result.embeddings):
    # Display the result or status information in the terminal.
    print(f"  [{i}] '{sentences[i][:40]}' → {len(emb.values)} dims")
