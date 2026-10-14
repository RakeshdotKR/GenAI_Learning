"""P17.py — Streamlit Basics

SUMMARY:
Introduces Streamlit basics: a page title, static text, text input, a button, and sidebar controls for model choice and temperature.

PREREQUISITES:
- Install: pip install streamlit. Run with streamlit run P17.py.

To Run:
".\.venv\Scripts\streamlit.exe" run "d:\Training\PESCE_Mandya\GenAI_Learning\3S_Training\me\Day3_\P10.py"
"""


# Import the libraries and framework components required by this example.
# Import the named Python module so this script can use its functionality.
import streamlit as st

# Page title — always the first Streamlit call
# Display the main page title in the Streamlit app.
st.title("My First Streamlit App")

# Static text
# Display the supplied text or value in the Streamlit app.
st.write("This text is rendered on the page.")

# Text input — returns the current value as a string
# Display a single-line text field and store the entered value.
name = st.text_input("Enter your name:")

# Button — returns True only on the turn it is clicked
# Check this condition and run the indented block only when it is true.
if st.button("Say Hello"):
    # Display the supplied text or value in the Streamlit app.
    st.write(f"Hello, {name}!")  # only runs when button is clicked

# Sidebar — side panel for settings
# Enter this context manager for the indented block.
with st.sidebar:
    # Display a section heading in the Streamlit interface.
    st.header("Settings")
    # Display a dropdown and store the selected option.
    model_choice = st.selectbox("Choose model:", ["phi3:mini", "gemma2:2b"])
    # Display a slider and store its selected numeric value.
    temperature = st.slider("Temperature:", 0.0, 1.0, 0.5)

# Display the supplied text or value in the Streamlit app.
st.write(f"Model: {model_choice}, Temperature: {temperature}")
