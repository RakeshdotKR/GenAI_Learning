# Role prompting = tell the model *who it is* before asking the question. 
# This shifts the vocabulary, tone, depth, and assumptions it makes.
from openai import OpenAI

MODEL = "phi3:mini"    # 4 GB RAM
# MODEL = "gemma2:2b"  # 8 GB RAM

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

question = "What is a neural network?"

roles = {
    "no role":      question,
    "professor":    f"You are a university professor teaching AI to final-year engineering students. {question}",
    "school teacher": f"You are a school teacher explaining technology to 10-year-olds. {question}",
    "interviewer":  f"You are a technical interviewer. Ask the candidate to explain: {question}",
}

def ask(prompt):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.3,
    )
    return r.choices[0].message.content.strip()

for role, prompt in roles.items():
    print(f"\n--- {role.upper()} ---")
    print(ask(prompt))


# Assignment for 13.10.26
# Mini-Challenge — Three Techniques, One Task (all in the same file)
# Task: Get the model to explain what a "API" is.
# Write three separate prompts for this same task — one using each technique — and compare the outputs.