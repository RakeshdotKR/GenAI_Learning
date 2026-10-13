# Gemini API Key usage for prompting
import os
from dotenv import load_dotenv

from google import genai
from google.genai import types

# Load the API key from your local .env file
load_dotenv()

# 2. Initialize the client (it automatically picks up GEMINI_API_KEY from environment if not passed explicitly)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 3. Call the model directly via the client object
result = client.models.generate_content(
    model="gemini-3.8-flash",  # Update to a valid model version if gemini-3.8-flash returns an error
    contents="What is a Vector database?",
    config=types.GenerateContentConfig(
        temperature=0.7,
        max_output_tokens=500,
        system_instruction="You are a concise AI tutor. Answer in 2-3 sentences."
    )
)

# 4. Print the response output and token data
print(result.text)
print(f"Tokens: {result.usage_metadata.total_token_count}")
