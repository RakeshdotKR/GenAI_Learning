# First API call wrt Gemini
import google.generativeai as genai
import os 
from dotenv import load_dotenv

# Load the API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create a model instance with config
# 2.0, 2.5, 3.5 
model = genai.GenerativeModel(model_name="gemini-2.5-flash",
                              generation_config={"temperature":0.7,
                                                 "max_output_tokens":150},
                                                 system_instruction="You" \
                                                 "are a concise AI tutor. " \
                                                 "Answer in 2-3 sentences.",)

# Call the model
result = model.generate_content("What is a Vector database?")
print(result.text)
print(f"Tokens:{result.usage_metadata.total}")