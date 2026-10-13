# Chain-of-thought (CoT) = ask the model to reason step by step before giving the final answer. 
# This works because each reasoning token improves the probability distribution for the next token.
from openai import OpenAI

MODEL = "phi3:mini"    # 4 GB RAM
# MODEL = "gemma2:2b"  # 8 GB RAM

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Without CoT — model jumps to answer, often wrong on logic problems
without_cot = """A store sells apples for ₹10 each and oranges for ₹15 each.
Ravi buys 3 apples and 2 oranges. How much does he pay in total?
Reply with just the number."""

# With CoT — model works through each step first
with_cot = """A store sells apples for ₹10 each and oranges for ₹15 each.
Ravi buys 3 apples and 2 oranges. How much does he pay in total?
Think step by step, then give the final answer."""

def ask(prompt):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.0,
    )
    return r.choices[0].message.content.strip()

print("Without CoT:", ask(without_cot))
print("\nWith CoT:")
print(ask(with_cot))