# The model does not change. The prompt does. The same LLM gives dramatically different results depending on how you ask it.
from openai import OpenAI

MODEL = "phi3:mini"    # 4 GB RAM
# MODEL = "gemma2:2b"  # 8 GB RAM

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def ask(prompt):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
    )
    return r.choices[0].message.content.strip()

bad  = ask("Tell me about sorting.")
good = ask(
    "Explain bubble sort to a first-year CS student in 3 bullet points. "
    "Use simple language. No code."
)

print("--- BAD PROMPT ---")
print(bad)
print("\n--- GOOD PROMPT ---")
print(good)