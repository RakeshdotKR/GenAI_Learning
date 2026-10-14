"""
P5.py — LangChain ChatPromptTemplate

SUMMARY:
This program defines a reusable chat prompt with system and human messages.
It fills the placeholders with a support role, company, tone, and question,
then prints each formatted message with its role. It formats messages only;
it does not invoke a language model.
"""

# Import LangChain's chat-prompt template class.
from langchain_core.prompts import ChatPromptTemplate

# Build a template from two structured chat messages: system instructions and user input.
# The {role}, {company}, {tone}, and {question} values will be supplied later.
template = ChatPromptTemplate.from_messages([
    # The system message describes the assistant's role, company, and communication style.
    ("system", "You are a {role} for {company}. Be {tone}."),
    # The human message represents the end user's question.
    ("human", "{question}"),
])

# Replace the template variables with values for this support scenario.
# format_messages() returns a list of LangChain message objects, not one plain string.
messages = template.format_messages(
    role="technical support specialist",  # Set the assistant's role.
    company="Nova Phones",                # Set the company name.
    tone="concise and professional",      # Set the desired communication style.
    question="How do I reset my phone to factory settings?",  # Supply the user's question.
)

# Iterate through each formatted message in the returned list.
for msg in messages:
    # Print the message role in uppercase, followed by its content.
    # Examples of roles are SYSTEM and HUMAN.
    print(f"{msg.type.upper()}: {msg.content}")
