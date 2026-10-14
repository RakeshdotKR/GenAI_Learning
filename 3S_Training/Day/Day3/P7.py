"""
P7.py — LangChain ChatPromptTemplate with Gemini

SUMMARY:
This program uses LangChain's ChatPromptTemplate with Google's Gemini chat model.
It loads environment variables, creates a model wrapper, and connects the prompt
to the model with LCEL. Invoking the chain supplies the company and question and
returns a response, which is printed.

PREREQUISITES:
- Install: pip install langchain-core langchain-google-genai python-dotenv
- Set GEMINI_API_KEY in a .env file or the environment.
"""

# Import LangChain's wrapper for Google's Gemini chat models.
from langchain_google_genai import ChatGoogleGenerativeAI
# Import the chat prompt template class.
from langchain_core.prompts import ChatPromptTemplate
# Import os to access environment variables if needed elsewhere in this script.
import os
# Import the helper that loads variables from a .env file into the environment.
from dotenv import load_dotenv

# Read environment variables from the local .env file, if present.
load_dotenv()

# Create the Gemini model wrapper and set response variability.
# The Google API key is normally read from GEMINI_API_KEY in the environment.
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.3)

# Define a system instruction and a human message with placeholders.
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant for {company}."),  # Assistant context.
    ("human", "{question}"),                                  # User question.
])

# Connect prompt formatting to model generation using LCEL.
chain = template | llm
# Supply values for both placeholders and execute the chain.
response = chain.invoke({"company": "Nova Phones", "question": "How do I reset my phone?"})
# Print the generated response text.
print(response.content)
