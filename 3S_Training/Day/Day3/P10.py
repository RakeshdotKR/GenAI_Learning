"""
P10.py — Prompting for JSON Output with Ollama

SUMMARY:
This program asks a local language model to extract a person's name, age, city,
and skills as JSON. It prints the raw response, parses it with json.loads(), and
uses the resulting Python dictionary to display selected fields. Invalid JSON is
handled with JSONDecodeError so the program can show a helpful error message.

PREREQUISITES:
- Ollama running with phi3:mini downloaded.
- Install: pip install openai
"""

# Import the OpenAI-compatible client used to send requests to Ollama.
from openai import OpenAI
# Import Python's built-in JSON module for parsing the model's output.
import json

# Select the local model to use.
MODEL = "phi3:mini"
# Alternative model; uncomment only if it is installed in Ollama.
# MODEL = "gemma2:2b"

# Configure the API client to use Ollama's local OpenAI-compatible endpoint.
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Write instructions asking the model to extract specific fields and return JSON only.
# The text inside the prompt is data for the model to process.
prompt = """Extract information from the text below and return it as JSON.
Use exactly these keys: name, age, city, skills (list of strings).
Return only the JSON object, no explanation.

Text: "My name is Priya. I am 22 years old, from Bengaluru.
I know Python, SQL, and machine learning." """

# Send the prompt to the selected model and store the API response.
r = client.chat.completions.create(
    model=MODEL,  # Select the local model.
    messages=[{"role": "user", "content": prompt}],  # Send the prompt as a user message.
    max_tokens=100,  # Limit the generated output length.
    temperature=0.0,  # Use low randomness for more consistent structured output.
)

# Extract the generated text and remove whitespace around it.
raw = r.choices[0].message.content.strip()
# Display the raw model output so it can be inspected if parsing fails.
print("Raw output:", raw)

# Try to parse the generated text as JSON and use it as a Python dictionary.
try:
    # json.loads() converts a valid JSON string into Python values such as dicts and lists.
    data = json.loads(raw)
    # Print the name value from the dictionary.
    print(f"\nName: {data['name']}")
    # Print the city value from the dictionary.
    print(f"City: {data['city']}")
    # Join the list of skills with commas and print the result.
    print(f"Skills: {', '.join(data['skills'])}")
# Handle the case where the model response is not valid JSON.
except json.JSONDecodeError:
    # Show a clear error message rather than allowing the parsing error to crash the program.
    print("Model did not return valid JSON — try rephrasing the prompt")
