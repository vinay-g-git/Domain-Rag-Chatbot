"""
app.py
------
Module 7 (Streamlit Interface).

- PDF uploader + domain/audience selector in the sidebar
- "Process Documents" button that runs the full ingest pipeline
- Chat input + chat history
- Source (document + page) shown below every answer
- "Clear Chat" button
"""

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from document_loader import validate_file, DocumentValidationError
from prompt import DOMAIN_HINTS
from rag_pipeline import RAGChatbot

load_dotenv()

st.set_page_config(page_title="Domain-Specific RAG Chatbot", page_icon="📄", layout="wide")

DOMAINS = list(DOMAIN_HINTS.keys())

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts: role, content, sources
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# ---------------------------------------------------------------------------
# Sidebar: upload + process
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📄 Domain-Specific RAG Chatbot")
    st.caption("Upload PDFs. Ask questions. Get grounded, source-cited answers.")

    domain = st.selectbox(
        "Who is this for? (tunes answer style, not the grounding rules)",
        DOMAINS,
        index=0,
    )

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files", type=["pdf"], accept_multiple_files=True
    )

    top_k = st.slider("Chunks to retrieve per question", min_value=2, max_value=8, value=4)

    process_clicked = st.button("Process Documents", type="primary", use_container_width=True)

    if uploaded_files:
        st.write("**Uploaded files:**")
        for f in uploaded_files:
            st.write(f"- {f.name} ({f.size / 1024:.1f} KB)")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    provider = os.environ.get("LLM_PROVIDER", "groq")
    st.caption(f"LLM provider: `{provider}` (set `LLM_PROVIDER` in `.env` to change)")

# ---------------------------------------------------------------------------
# Process documents
# ---------------------------------------------------------------------------
if process_clicked:
    if not uploaded_files:
        st.sidebar.error("Please upload at least one PDF first.")
    else:
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                pdf_paths = []
                for f in uploaded_files:
                    validate_file(f.name, f.size)
                    path = Path(tmp_dir) / f.name
                    path.write_bytes(f.getvalue())
                    pdf_paths.append(str(path))

                with st.spinner("Extracting text, chunking, and building the vector index..."):
                    chatbot = RAGChatbot(domain=domain, top_k=top_k)
                    num_chunks = chatbot.ingest(pdf_paths)
                    st.session_state.chatbot = chatbot
                    st.session_state.processed_files = [f.name for f in uploaded_files]

                st.sidebar.success(f"Indexed {num_chunks} chunks from {len(pdf_paths)} file(s).")
        except DocumentValidationError as e:
            st.sidebar.error(str(e))
        except Exception as e:
            st.sidebar.error(f"Failed to process documents: {e}")

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.header("Chat")

if st.session_state.processed_files:
    st.caption("Answering from: " + ", ".join(st.session_state.processed_files))
else:
    st.info("Upload PDFs and click **Process Documents** in the sidebar to get started.")

for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])
        if turn.get("sources"):
            with st.expander("Sources"):
                for s in turn["sources"]:
                    st.write(f"- **{s.document}**, page {s.page} (similarity {s.score:.2f})")
                    st.caption(s.snippet + "...")

question = st.chat_input("Ask a question about the uploaded documents...")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question, "sources": None})
    with st.chat_message("user"):
        st.write(question)

    if st.session_state.chatbot is None:
        answer_text = "Please upload and process at least one PDF before asking questions."
        sources = []
    else:
        with st.spinner("Retrieving relevant passages and generating an answer..."):
            result = st.session_state.chatbot.ask(question)
            answer_text = result.answer
            sources = result.sources

    with st.chat_message("assistant"):
        st.write(answer_text)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.write(f"- **{s.document}**, page {s.page} (similarity {s.score:.2f})")
                    st.caption(s.snippet + "...")
        else:
            st.caption("⚠️ Always verify high-stakes information against the original document.")

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer_text, "sources": sources}
    )
