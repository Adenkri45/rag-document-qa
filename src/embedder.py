from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"

# Load once, reuse everywhere
_model = None

def get_model():
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_chunks(chunks: list[dict]) -> np.ndarray:
    """
    Convert list of chunk dicts to embeddings.
    Returns numpy array of shape (num_chunks, 384)
    """
    model = get_model()
    texts = [chunk["text"] for chunk in chunks]

    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}")

    return embeddings


def embed_query(query: str) -> np.ndarray:
    """Convert a single question to embedding."""
    model = get_model()
    return model.encode([query])


if __name__ == "__main__":
    from src.document_loader import load_and_chunk_pdf
    import sys

    pdf_path = sys.argv[1]
    chunks = load_and_chunk_pdf(pdf_path)
    embeddings = embed_chunks(chunks)

    print(f"\nFirst chunk: {chunks[0]['text'][:100]}...")
    print(f"Its embedding (first 5 numbers): {embeddings[0][:5]}")
    