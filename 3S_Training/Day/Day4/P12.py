"""
P12.py — Cosine Similarity — Measuring Meaning Distance

SUMMARY:
This example calculates cosine similarity between numeric embedding vectors.
It demonstrates how vector direction can be used to estimate how closely two
pieces of text are related in meaning.

PREREQUISITES:
- Install any third-party packages imported by this script, if they are not already installed.
"""

# Import mathematical functions used for vector calculations.
import math

# Define a reusable function for the operation described by its name and parameters.
def cosine_similarity(a: list[float], b: list[float]) -> float:
    # Iterate over corresponding items from multiple collections together.
    dot     = sum(x * y for x, y in zip(a, b))
    # Calculate the vector magnitude using the square root of the sum of squared values.
    mag_a   = math.sqrt(sum(x ** 2 for x in a))
    # Calculate the vector magnitude using the square root of the sum of squared values.
    mag_b   = math.sqrt(sum(x ** 2 for x in b))
    # Check this condition and run the indented block only when it is true.
    if mag_a == 0 or mag_b == 0:
        # Return the calculated value to the caller.
        return 0.0
    # Return the calculated value to the caller.
    return dot / (mag_a * mag_b)

# Simple 3D example to build intuition
# Set `v1` to the value calculated by the expression on this line.
v1 = [1, 0, 1]   # meaning: "some concept"
# Set `v2` to the value calculated by the expression on this line.
v2 = [1, 0, 0]   # partially overlapping
# Set `v3` to the value calculated by the expression on this line.
v3 = [0, 1, 0]   # completely different direction

# Display the result or status information in the terminal.
print("v1 vs v2 (partial):", round(cosine_similarity(v1, v2), 4))  # 0.7071
# Display the result or status information in the terminal.
print("v1 vs v3 (unrelated):", round(cosine_similarity(v1, v3), 4))  # 0.0
# Display the result or status information in the terminal.
print("v1 vs v1 (identical):", round(cosine_similarity(v1, v1), 4))  # 1.0
