"""P3.py — Mini-Challenge — Styled Input/Output Page

SUMMARY:
Builds a small text transformer with options for uppercase, lowercase, title case, and reversing text, with a warning for empty input.

PREREQUISITES:
- Install: pip install streamlit. Run with streamlit run P3.py.
"""


# Import the libraries and framework components required by this example.
# Import the named Python module so this script can use its functionality.
import streamlit as st

# Display the main page title in the Streamlit app.
st.title("Text Transformer")

# Sidebar options
# Enter this context manager for the indented block.
with st.sidebar:
    # Display a dropdown and store the selected option.
    transform = st.selectbox("Transform:", ["UPPERCASE", "lowercase", "Title Case", "Reverse"])

# Text input area — larger than st.text_input
# Display a multiline text field and store the entered value.
text = st.text_area("Enter text here:")

# Check this condition and run the indented block only when it is true.
if st.button("Transform"):
    # Check this condition and run the indented block only when it is true.
    if not text.strip():
        # Show a warning message to the user.
        st.warning("Please enter some text first.")  # non-blocking warning
    # Run this alternative block when the preceding conditions are false.
    else:
        # Apply the chosen transform
        # Check this condition and run the indented block only when it is true.
        if transform == "UPPERCASE":
            # Store the result returned by the model or operation.
            result = text.upper()
        # Check this alternative condition if the preceding condition was false.
        elif transform == "lowercase":
            # Store the result returned by the model or operation.
            result = text.lower()
        # Check this alternative condition if the preceding condition was false.
        elif transform == "Title Case":
            # Store the result returned by the model or operation.
            result = text.title()
        # Run this alternative block when the preceding conditions are false.
        else:
            # Store the result returned by the model or operation.
            result = text[::-1]  # reverse

        # st.success renders a green box
        # Show a success message to the user.
        st.success("Transformed output:")
        # Display the result in a formatted code block.
        st.code(result)  # monospace code block
