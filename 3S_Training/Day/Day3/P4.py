
"""
P4.py — LangChain PromptTemplate

SUMMARY:
This program uses LangChain's PromptTemplate to create a reusable
customer support prompt.

The template contains placeholders for a product name and a customer issue.
The template.format() method replaces those placeholders with actual values
and returns the completed prompt as a Python string.

The generated prompt can later be supplied to a language model.
"""

# Import PromptTemplate from LangChain's core prompts module.
# PromptTemplate helps define reusable prompts with named placeholders.
from langchain_core.prompts import PromptTemplate

# Define a reusable prompt template using the from_template() class method.
# LangChain identifies {product} and {issue} as template variables.
# Their values will be provided later when the template is formatted.
template = PromptTemplate.from_template(

    # Define the prompt as a multiline string.
    # {product} will be replaced with a product or company name.
    # {issue} will be replaced with the customer's reported problem.
    # The final sentence instructs the AI to respond in 2-3 sentences.
    """You are a customer support agent for {product}.
A customer reports: "{issue}"
Provide a helpful response in 2-3 sentences."""
)

# Fill in the template variables with actual values.
# product="Nova Phones" replaces the {product} placeholder.
# issue="My battery drains in 3 hours." replaces the {issue} placeholder.
# The format() method returns the completed prompt as a string.
# The completed string is stored in the variable filled.
filled = template.format(
    product="Nova Phones",
    issue="My battery drains in 3 hours."
)

# Print the completed prompt to the terminal.
# The placeholders have already been replaced with their supplied values.
print(filled)

# NEXT STEP:
# The variable filled now contains a normal Python string.
# You can pass it to an LLM API call or use it in a LangChain chain.
# No AI model is called anywhere in this program.
