"""P15.py — .invoke() runs the full chain

SUMMARY:
Builds a reusable LCEL chain whose prompt variables control the teacher style, concept, and audience, then invokes it with concrete values.

PREREQUISITES:
- Install: pip install langchain-core langchain-ollama. Ollama must be running with phi3:mini downloaded.
"""


# LCEL uses the pipe operator to pass each component's output to the next component.
# Import the libraries and framework components required by this example.
# Import the required class or function from this package.
from langchain_core.prompts import ChatPromptTemplate
# Import the required class or function from this package.
from langchain_core.output_parsers import StrOutputParser
# Import the required class or function from this package.
from langchain_ollama import ChatOllama

# Set the name of the local Ollama model to use.
OLLAMA_MODEL = "phi3:mini"
# OLLAMA_MODEL = "gemma2:2b"

# Configure a LangChain chat model that sends prompts to the local Ollama service.
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.5)

# Build the chain by connecting prompt, model, and output-processing stages.
chain = (
    # Create a chat prompt template from role-message pairs.
    ChatPromptTemplate.from_messages([
        # Begin the grouped expression or collection used by the statement above.
        ("system", "You are a {style} teacher."),
        # Begin the grouped expression or collection used by the statement above.
        ("human", "Explain {concept} to a {audience}."),
    # Close the function call, collection, or grouped expression opened above.
    ])
    # Pipe the previous component output into the next stage of the LCEL chain.
    | llm
    # Convert the model output message into a plain Python string.
    | StrOutputParser()
# Close the function call, collection, or grouped expression opened above.
)

# Display this value in the terminal.
print(chain.invoke({
    # Perform this step as part of the example.
    "style": "funny",
    # Perform this step as part of the example.
    "concept": "recursion",
    # Perform this step as part of the example.
    "audience": "10-year-old",
# Perform this step as part of the example.
}))
