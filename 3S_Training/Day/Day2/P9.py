# Few-shot = give the model 2–4 examples of input → output before asking your actual question. The model learns the pattern from the examples.

from openai import OpenAI

MODEL = "phi3:mini"    # 4 GB RAM
# MODEL = "gemma2:2b"  # 8 GB RAM

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Examples teach the model the exact format you want
prompt = """Extract the product name and price from each line. Format: name | price

Input: "Buy the UltraSound Pro headphones for just Rs2,499 today."
Output: UltraSound Pro | Rs2,499

Input: "The Zephyr 4K Monitor is available at Rs18,999."
Output: Zephyr 4K Monitor | Rs18,999

Input: "Order the CoolBreeze portable fan for Rs899."
Output:"""

r = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=20,
    temperature=0.0,
)
print(r.choices[0].message.content.strip())