
"""
P3.py — Prompt Templates with f-strings

SUMMARY:
This program creates reusable customer support prompts using a Python
function and f-strings.

The support_prompt() function accepts a product/company name, a customer
issue, and an optional response tone. It returns a customized prompt.

The function is called twice with different inputs to demonstrate reuse.
"""

# Define a function named support_prompt.
# product: str means the product parameter is expected to be a string.
# issue: str means the issue parameter is expected to be a string.
# tone: str = "formal" means tone is a string with a default value of "formal".
# -> str indicates that the function is expected to return a string.
# Note: Python type hints describe expectations but do not enforce types.
def support_prompt(product: str, issue: str, tone: str = "formal") -> str:

    # Return a customized prompt using a multiline f-string.
    # The f before the triple quotes enables variable interpolation.
    # Triple quotes allow the prompt to span multiple lines.
    # {tone} is replaced with the value passed through the tone parameter.
    # {product} is replaced with the product or company name.
    # {issue} is replaced with the customer's issue.
    # The final sentence instructs the AI to respond in 2–3 sentences.
    return f"""You are a {tone} customer support agent for {product}.
A customer says: "{issue}"
Respond helpfully in 2-3 sentences."""

# Call support_prompt() with a company name and a customer issue.
# Since tone is not provided, its default value "formal" is used.
# The generated prompt is stored in the variable p1.
p1 = support_prompt("Nova Phones", "My screen cracked after one drop.")

# Call the same function again with different input values.
# "SpeedNet WiFi" replaces the product parameter.
# The internet disconnection message replaces the issue parameter.
# tone="friendly" overrides the default tone of "formal".
# The generated prompt is stored in the variable p2.
p2 = support_prompt(
    "SpeedNet WiFi",
    "My internet keeps disconnecting.",
    tone="friendly"
)

# Print the first generated prompt.
print(p1)

# Print an empty line to separate the two prompts in the output.
# print() without arguments prints a newline.
print()

# Print the second generated prompt.
print(p2)
