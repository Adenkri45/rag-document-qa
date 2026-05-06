# 📄 RAG Document Q&A System

A production-style Retrieval Augmented Generation (RAG) system that lets you upload any PDF and ask questions — answers are grounded in your document with source citations.

## 🎯 What It Does

> "Stop asking LLMs from memory — ask from your own documents"

Instead of relying on an LLM's training data, this system retrieves relevant passages from your uploaded PDF and generates answers strictly from that content.

---

## 🏗️ Architecture
PDF Upload
↓
Document Chunking (500 words, 50 word overlap)
↓
Sentence-BERT Embeddings (all-MiniLM-L6-v2, 384 dimensions)
↓
FAISS Vector Store (cosine similarity search)
↓
User Question → Embedding → Top-K Retrieval
↓
Groq LLM (Llama 3.3 70B) → Grounded Answer + Sources

---

## 📁 Project Structure
├── src/
│   ├── document_loader.py   # PDF parsing + text chunking
│   ├── embedder.py          # Sentence-BERT embeddings
│   ├── vector_store.py      # FAISS index + similarity search
│   └── generator.py         # Groq LLM answer generation
├── api/
│   └── main.py              # FastAPI REST endpoints
├── app/
│   └── streamlit_app.py     # Chat UI with source citations
├── learn/                   # Concept experiments
│   ├── embeddings.py
│   ├── chunking.py
│   └── faiss_demo.py
└── requirements.txt

---

## 🚀 Setup

```bash
git clone https://github.com/Adenkri45/rag-document-qa.git
cd rag-document-qa
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set your Groq API key:
```bash
export GROQ_API_KEY="your-key-here"
```

---

## 🧪 Running the Project

### Start FastAPI (Terminal 1)
```bash
python -m uvicorn api.main:app --reload
# Visit http://localhost:8000/docs
```

### Start Streamlit UI (Terminal 2)
```bash
streamlit run app/streamlit_app.py
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/upload` | Upload and index a PDF |
| POST | `/ask` | Ask a question, get grounded answer |
| GET | `/document` | Get current document info |
| GET | `/health` | Health check |

---

## 🔍 Key Concepts

- **Chunking** — PDF split into 500 word overlapping passages so LLM context limits aren't exceeded
- **Embeddings** — Sentence-BERT converts text to 384-dimensional vectors capturing semantic meaning
- **FAISS** — Vector database for fast cosine similarity search across all chunks
- **RAG** — Retrieved chunks + question fed to LLM for grounded, citation-backed answers

---

## 🛠️ Tech Stack

`Python` `Sentence-BERT` `FAISS` `Groq` `Llama 3.3 70B` `FastAPI` `Streamlit` `PyMuPDF`