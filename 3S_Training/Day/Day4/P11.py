"""
P11.py — embed_batch.py

SUMMARY:
It defines the required inputs, performs the main operation, and prints or stores the result
so the behavior can be inspected.

PREREQUISITES:
- Ollama installed and running, with the referenced model downloaded.
"""

# Install: pip install ollama
# Pull model first: ollama pull nomic-embed-text
# Import the Ollama client for interacting with locally running models.
import ollama
# Save the result returned by the model, loader, or database operation.
response = ollama.embed(
    model="nomic-embed-text",
    input="Python is a programming language.",
)

# Set `vector` to the value calculated by the expression on this line.
vector = response.embeddings[0]
# Display the result or status information in the terminal.
print(f"Embedding length : {len(vector)}")
# Display the result or status information in the terminal.
print(f"First 5 values   : {vector[:5]}")
