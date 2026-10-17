# app.py — PDF Chat App
import os, tempfile
import streamlit as st
# from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Use os.environ first (great for Docker/Local setups), then fallback safely to st.secrets
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# If it's not found in the environment variables, check Streamlit secrets carefully
if not GEMINI_API_KEY:
    try:
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass # If secrets don't exist at all, ignore the error and keep checking below


# Pass the key explicitly so the SDK never looks for an env var
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

st.set_page_config(page_title="PDF Chat", page_icon="📄")
st.title("📄 Chat with your PDF")

# ── Sidebar — upload and process ──────────────────────────────────────────────
with st.sidebar:
    uploaded = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded and st.button("Process PDF"):
        # Write the upload to a temp file so PyPDFLoader can open it by path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner("Indexing PDF…"):
            pages    = PyPDFLoader(tmp_path).load()
            chunks   = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(pages)

            embeddings  = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
            vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
            retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
            llm         = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.2)

            def format_docs(docs):
                return "\n\n".join(d.page_content for d in docs)

            prompt = ChatPromptTemplate.from_messages([
                ("system", "Answer using only the context. Say 'I don't know' if unsure."),
                ("human", "Context:\n{context}\n\nQuestion: {question}"),
            ])

            # RunnableParallel returns both the answer string and the source docs
            st.session_state.chain = RunnableParallel(
                answer={"context": retriever | format_docs, "question": RunnablePassthrough()}
                       | prompt | llm | StrOutputParser(),
                sources=retriever,
            )
            st.session_state.messages = []
            st.session_state.pdf_name = uploaded.name

        st.success(f"Ready — {len(chunks)} chunks from {uploaded.name}")
        os.unlink(tmp_path)

    if "pdf_name" in st.session_state:
        st.caption(f"Active: {st.session_state.pdf_name}")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ── Chat area ─────────────────────────────────────────────────────────────────
if "chain" not in st.session_state:
    st.info("Upload and process a PDF from the sidebar to start chatting.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about the PDF…")

    if question and question.strip():
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.spinner("Searching PDF…"):
            try:
                result = st.session_state.chain.invoke(question)
                answer = result["answer"]

                # Collect unique page citations from source metadata
                seen, citations = set(), []
                for doc in result["sources"]:
                    page = doc.metadata.get("page", "?")
                    key  = str(page)
                    if key not in seen:
                        citations.append(f"page {int(page) + 1}")  # 0-indexed → 1-indexed
                        seen.add(key)

                full_reply = answer
                if citations:
                    full_reply += "\n\n**Sources:** " + ", ".join(citations)
            except Exception as e:
                full_reply = f"Error: {e}"

        st.session_state.messages.append({"role": "assistant", "content": full_reply})
        with st.chat_message("assistant"):
            st.markdown(full_reply)