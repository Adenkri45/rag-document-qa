import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from src.document_loader import load_and_chunk_pdf
from src.embedder import embed_chunks, embed_query
from src.vector_store import VectorStore

# ── Config ───────────────────────────────────────────────────────────────
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────

app = FastAPI(title="RAG Document Q&A API", version="1.0")

# One vector store per session (in memory)
store = VectorStore()
current_doc = None


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 3


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    model: str
    chunks_retrieved: int


@app.get("/health")
def health():
    return {
        "status": "ok",
        "document_loaded": current_doc is not None,
        "chunks_indexed": store.index.ntotal
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global store, current_doc

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Process PDF
    chunks = load_and_chunk_pdf(file_path)
    embeddings = embed_chunks(chunks)

    # Reset store and index new document
    store = VectorStore()
    store.add(chunks, embeddings)
    current_doc = file.filename

    return {
        "message": "Document uploaded and indexed successfully",
        "filename": file.filename,
        "chunks_created": len(chunks),
        "words_extracted": sum(c["word_count"] for c in chunks)
    }


@app.post("/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest):
    global store, current_doc

    if current_doc is None:
        raise HTTPException(status_code=400, detail="No document uploaded yet. Use /upload first.")

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Retrieve relevant chunks
    query_emb = embed_query(req.question)
    results = store.search(query_emb, top_k=req.top_k)

    if not results:
        raise HTTPException(status_code=404, detail="No relevant chunks found")

    # Generate answer
    from src.generator import generate_answer
    response = generate_answer(req.question, results)

    return AnswerResponse(
        question=response["question"],
        answer=response["answer"],
        sources=response["sources"],
        model=response["model"],
        chunks_retrieved=len(results)
    )


@app.get("/document")
def get_current_document():
    if current_doc is None:
        return {"document": None, "chunks_indexed": 0}
    return {
        "document": current_doc,
        "chunks_indexed": store.index.ntotal
    }