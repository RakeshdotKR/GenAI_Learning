"""
P13.py — PDF Chat App — Full Application

SUMMARY:
This Streamlit application lets a user upload a PDF, splits its pages into
overlapping chunks, embeds them into ChromaDB, and creates a RAG chain for
question answering. It keeps chat messages in Streamlit session state and
displays source page numbers alongside answers.

PREREQUISITES:
- Configure GEMINI_API_KEY in a .env file.
- Install: pip install pypdf streamlit python-dotenv langchain-community langchain-text-splitters langchain-chroma langchain-google-genai
- Run with: streamlit run P13.py
"""

# Import operating-system utilities and temporary-file support.
import os, tempfile
# Import Streamlit to build the interactive web application.
import streamlit as st
# Load API credentials and settings from a .env file.
from dotenv import load_dotenv
# Import a PDF loader that creates LangChain documents from PDF pages.
from langchain_community.document_loaders import PyPDFLoader
# Import a text splitter for creating overlapping PDF chunks.
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Import LangChain's integration with the Chroma vector database.
from langchain_chroma import Chroma
# Import Gemini chat-model and text-embedding integrations for LangChain.
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# Import the class used to create structured chat prompts.
from langchain_core.prompts import ChatPromptTemplate
# Import a parser that converts model output into a plain string.
from langchain_core.output_parsers import StrOutputParser
# Import pipeline helpers to pass input through and run multiple branches.
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Read key-value settings from the .env file into the process environment.
# Load API credentials such as GEMINI_API_KEY from the .env file.
load_dotenv()

# Configure the Streamlit page title and icon.
st.set_page_config(page_title="PDF Chat", page_icon="📄")
# Display the main heading in the Streamlit app.
st.title("📄 Chat with your PDF")

# ── Sidebar — upload and process ──────────────────────────────────────────────
# Enter this Streamlit UI section or safely manage the temporary resource.
with st.sidebar:
    # Display a file picker that accepts PDF files.
    # Let the user select a PDF from their computer.
    uploaded = st.file_uploader("Choose a PDF file", type="pdf")

    # Run the following block only when this condition is true.
    if uploaded and st.button("Process PDF"):
        # Write the upload to a temp file so PyPDFLoader can open it by path
        # Enter this Streamlit UI section or safely manage the temporary resource.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        # Enter this Streamlit UI section or safely manage the temporary resource.
        with st.spinner("Indexing PDF…"):
            # Load source content into LangChain Document objects.
            pages    = PyPDFLoader(tmp_path).load()
            # Set the target chunk size and overlap between neighboring chunks.
            chunks   = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(pages)

            # Configure Gemini's embedding model to convert text into vectors.
            embeddings  = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
            # Create a Chroma vector store from LangChain Document objects and their embeddings.
            vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
            # Create a retriever that returns the most relevant stored documents.
            retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
            # Configure Gemini as the language model used to generate answers.
            llm         = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.2)

            # Define a function so this operation can be reused.
            # Combine the page content of retrieved documents into a single context string.
            def format_docs(docs):
                # Return the computed value to the code that called this function.
                return "\n\n".join(d.page_content for d in docs)

            # Define the system instructions and human-message template for the model.
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Answer using only the context. Say 'I don't know' if unsure."),
                ("human", "Context:\n{context}\n\nQuestion: {question}"),
            ])

            # RunnableParallel returns both the answer string and the source docs
            # Run multiple chain branches and collect their results.
            st.session_state.chain = RunnableParallel(
                # Pass the original question through unchanged.
                answer={"context": retriever | format_docs, "question": RunnablePassthrough()}
                       # Convert the model's response object into plain text.
                       | prompt | llm | StrOutputParser(),
                sources=retriever,
            )
            # Store this value in session state so it survives Streamlit reruns.
            # Start a fresh chat history for the newly processed PDF.
            st.session_state.messages = []
            # Store this value in session state so it survives Streamlit reruns.
            # Remember the uploaded PDF's filename for display in the sidebar.
            st.session_state.pdf_name = uploaded.name

        # Display a status message in the Streamlit interface.
        st.success(f"Ready — {len(chunks)} chunks from {uploaded.name}")
        # Delete the temporary file after processing.
        os.unlink(tmp_path)

    # Run the following block only when this condition is true.
    if "pdf_name" in st.session_state:
        # Display a status message in the Streamlit interface.
        st.caption(f"Active: {st.session_state.pdf_name}")

    # Run the following block only when this condition is true.
    if st.button("Clear Chat"):
        # Store this value in session state so it survives Streamlit reruns.
        # Start a fresh chat history for the newly processed PDF.
        st.session_state.messages = []
        # Rerun the Streamlit script immediately to refresh the cleared chat UI.
        st.rerun()

# ── Chat area ─────────────────────────────────────────────────────────────────
# Run the following block only when this condition is true.
if "chain" not in st.session_state:
    # Display a status message in the Streamlit interface.
    st.info("Upload and process a PDF from the sidebar to start chatting.")
# Run this alternative block when the preceding condition is false.
else:
    # Run the following block only when this condition is true.
    if "messages" not in st.session_state:
        # Store this value in session state so it survives Streamlit reruns.
        # Start a fresh chat history for the newly processed PDF.
        st.session_state.messages = []

    # Process each item in the collection one at a time.
    for msg in st.session_state.messages:
        # Enter this Streamlit UI section or safely manage the temporary resource.
        with st.chat_message(msg["role"]):
            # Display this message as Markdown in the Streamlit chat interface.
            st.markdown(msg["content"])

    # Display the chat input and read the user's question.
    # Display the chat input and capture the user's question.
    question = st.chat_input("Ask a question about the PDF…")

    # Run the following block only when this condition is true.
    if question and question.strip():
        st.session_state.messages.append({"role": "user", "content": question})
        # Enter this Streamlit UI section or safely manage the temporary resource.
        with st.chat_message("user"):
            # Display this message as Markdown in the Streamlit chat interface.
            st.markdown(question)

        # Enter this Streamlit UI section or safely manage the temporary resource.
        with st.spinner("Searching PDF…"):
            # Start a try block so runtime errors can be handled gracefully.
            try:
                # Execute the chain or retriever using the current input.
                result = st.session_state.chain.invoke(question)
                answer = result["answer"]

                # Collect unique page citations from source metadata
                # Track pages already cited and collect unique citation labels.
                seen, citations = set(), []
                # Process each item in the collection one at a time.
                for doc in result["sources"]:
                    page = doc.metadata.get("page", "?")
                    key  = str(page)
                    # Run the following block only when this condition is true.
                    if key not in seen:
                        citations.append(f"page {int(page) + 1}")  # 0-indexed → 1-indexed
                        # Store each text chunk with its corresponding embedding and ID.
                        seen.add(key)

                # Start the displayed reply with the generated answer.
                full_reply = answer
                # Run the following block only when this condition is true.
                if citations:
                    full_reply += "\n\n**Sources:** " + ", ".join(citations)
            # Handle an error raised while processing the question.
            except Exception as e:
                full_reply = f"Error: {e}"

        st.session_state.messages.append({"role": "assistant", "content": full_reply})
        # Enter this Streamlit UI section or safely manage the temporary resource.
        with st.chat_message("assistant"):
            # Display this message as Markdown in the Streamlit chat interface.
            st.markdown(full_reply)
