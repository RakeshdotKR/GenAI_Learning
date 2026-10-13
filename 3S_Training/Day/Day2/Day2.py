# This file gives you daywise info
# Install Python Packages
# openai: ollama & Openai api key
# google-generativeai: gemini api key
pip install openai google-generativeai

# Code 1:
# First API Call to ollama
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1",api_key="ollama")
r = client.chat.completions.create(model="gemma2:2b",
                                   messages=[{"role":"user",
                                              "content":"What is an LLM? "
                                              "Answer in 3 sentences"}],)

print(r.choices[0].message.content)
print(f"Tokens used:{r.usage.total_tokens}")

# code 2:
#### Add a system prompt and generation parameters
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1",api_key="ollama")

r = client.chat.completions.create(
    model="gemma2:2b",
    messages=[
        {"role": "system", "content": "You are a concise AI tutor. Answer in 2-3 sentences."},
        {"role": "user",   "content": "What is a vector database?"},
    ],
    temperature=0.1,   # controls randomness — 0.0 is deterministic, 1.5 is very random
    max_tokens=150,    # caps the response length so it doesn't run on forever
)
print(r.choices[0].message.content)
print(f"Tokens used:{r.usage.total_tokens}")