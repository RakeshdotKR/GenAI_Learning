# Zero-shot = give the task directly, no examples. The model uses only what it learned during training.
from openai import OpenAI

MODEL = "phi3:mini"    # 4 GB RAM
# MODEL = "gemma2:2b"  # 8 GB RAM

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

prompt = """Classify the sentiment of this review as Positive, Negative, or Neutral.
Reply with one word only.

Review: "The camera quality is great but the battery drains too fast." """

r = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=5,
    temperature=0.0,   # deterministic — classification should always give the same answer
)
print(r.choices[0].message.content.strip())