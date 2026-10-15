"""P1.py — Session State — Persisting Data Across Reruns

SUMMARY:
Builds a counter app using Streamlit session state so the count survives the script reruns caused by button interactions.

PREREQUISITES:
- Install: pip install streamlit. Run with streamlit run P11.py.
"""


# Streamlit reruns this script after interactions; session_state preserves values across reruns.
# Import the libraries and framework components required by this example.
# Import the named Python module so this script can use its functionality.
import streamlit as st

# Display the main page title in the Streamlit app.
st.title("Counter App")

# Initialise counter if it doesn't exist yet — runs only on first load
# Check this condition and run the indented block only when it is true.
if "count" not in st.session_state:
    # Set the saved counter to the specified value.
    st.session_state.count = 0

# Buttons modify session_state — the updated value persists on next rerun
# Check this condition and run the indented block only when it is true.
if st.button("Increment"):
    # Increase the saved counter by the specified amount.
    st.session_state.count += 1

# Check this condition and run the indented block only when it is true.
if st.button("Reset"):
    # Set the saved counter to the specified value.
    st.session_state.count = 0

# Display the supplied text or value in the Streamlit app.
st.write(f"Count: {st.session_state.count}")
