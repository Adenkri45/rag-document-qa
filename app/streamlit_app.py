import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG Document Q&A", layout="wide")
st.title("📄 RAG Document Q&A System")
st.caption("Upload a PDF and ask questions — answers grounded in your document")

# ── Sidebar — Upload ──────────────────────────────────────────────────────
st.sidebar.header("📂 Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Uploading and indexing..."):
        response = requests.post(
            f"{API_URL}/upload",
            files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
        )
    if response.status_code == 200:
        data = response.json()
        st.sidebar.success(f"✅ Indexed {data['chunks_created']} chunks")
        st.sidebar.info(f"📄 {data['filename']}")
    else:
        st.sidebar.error("Upload failed")

# ── Health check ──────────────────────────────────────────────────────────
health = requests.get(f"{API_URL}/health").json()
if health["document_loaded"]:
    st.sidebar.success(f"🟢 {health['chunks_indexed']} chunks ready")
else:
    st.sidebar.warning("🟡 No document loaded yet")

# ── Main — Q&A ────────────────────────────────────────────────────────────
st.subheader("💬 Ask a Question")

question = st.text_input("Enter your question about the document:")
top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=5, value=3)

if st.button("Ask") and question:
    if not health["document_loaded"]:
        st.error("Please upload a document first.")
    else:
        with st.spinner("Searching document and generating answer..."):
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question, "top_k": top_k}
            )

        if response.status_code == 200:
            data = response.json()

            st.markdown("### 📝 Answer")
            st.success(data["answer"])

            st.markdown("### 📚 Sources Used")
            for i, source in enumerate(data["sources"]):
                with st.expander(f"Source {i+1}"):
                    st.write(source)

            st.caption(f"Model: `{data['model']}` | Chunks retrieved: `{data['chunks_retrieved']}`")
        else:
            st.error(f"Error: {response.json()['detail']}")