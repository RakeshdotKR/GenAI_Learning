"""
P9.py — Generating Embeddings with Gemini

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

# Request embedding vectors for the supplied text using the supported Gemini embedding model.
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents="Python is a high-level programming language.",
)

# Extract the values array from the first returned embedding object.
vector = result.embeddings[0].values
# Display the result or status information in the terminal.
print(f"Embedding length : {len(vector)}")      # 768 dimensions
# Display the result or status information in the terminal.
print(f"First 5 values   : {vector[:5]}")
# Display the result or status information in the terminal.
print(f"Type             : {type(vector[0])}")  # float

