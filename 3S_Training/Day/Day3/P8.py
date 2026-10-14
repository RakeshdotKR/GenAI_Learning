"""
P8.py — Multi-Turn Conversations: Building Chat History Manually

SUMMARY:
This program keeps a list of chat messages so each model call can see previous
turns. The chat() function appends the user's message, calls the model with the
full history, stores the assistant's reply, and returns it. Three calls simulate
a conversation and the final message count shows how history grows.

PREREQUISITES:
- Ollama must be running and phi3:mini downloaded.
- Install the OpenAI SDK: pip install openai
"""

# Import the OpenAI SDK client, which can communicate with Ollama's compatible API.
from openai import OpenAI

# Select the locally available language model.
MODEL = "phi3:mini"
# Alternative model; uncomment if it is installed in Ollama.
# MODEL = "gemma2:2b"

# Configure the client to send requests to Ollama running on this computer.
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Start the conversation with a system message that sets the tutor's behavior.
# This list is updated after every user message and assistant reply.
history = [
    {"role": "system", "content": "You are a helpful Python tutor. Keep answers brief."}
]

# Define a function that sends one user message and returns the assistant's reply.
# user_message is expected to be a string; -> str documents the return type.
def chat(user_message: str) -> str:
    # Add the new user message so the model can respond to it.
    history.append({"role": "user", "content": user_message})
    # Send the entire conversation history, not just the latest question.
    r = client.chat.completions.create(
        model=MODEL,             # Select the model for this request.
        messages=history,         # Provide system instructions and all prior turns.
        max_tokens=150,           # Limit the length of the generated response.
        temperature=0.3,          # Use relatively low response variability.
    )
    # Extract the first response's text and remove surrounding whitespace.
    reply = r.choices[0].message.content.strip()
    # Save the assistant reply so the next call has the full conversation context.
    history.append({"role": "assistant", "content": reply})
    # Return the reply to the code that called chat().
    return reply

# Send the first question and print the answer.
print(chat("What is a list in Python?"))
# Ask a follow-up question; the previous exchange remains in history.
print(chat("How is it different from a tuple?"))
# Ask another follow-up that depends on the preceding topic.
print(chat("Give me one example of when I'd use a tuple instead of a list."))

# Print the number of stored messages, including the system message and each turn.
print(f"\nTotal messages in history: {len(history)}")
