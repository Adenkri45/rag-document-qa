import faiss
import numpy as np
import pickle
import os


class VectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # IP = Inner Product (cosine similarity)
        self.chunks = []  # store original chunk dicts alongside embeddings

    def add(self, chunks: list[dict], embeddings: np.ndarray):
        """Add chunks and their embeddings to the store."""
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings.astype('float32'))
        self.index.add(embeddings.astype('float32'))
        self.chunks.extend(chunks)
        print(f"Added {len(chunks)} chunks. Total: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[dict]:
        """Find top_k most similar chunks to a query."""
        query = query_embedding.astype('float32')
        faiss.normalize_L2(query)

        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # -1 means no result found
                results.append({
                    "chunk": self.chunks[idx],
                    "score": float(score)
                })
        return results

    def save(self, path: str):
        """Save index and chunks to disk."""
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"Saved vector store to: {path}")

    def load(self, path: str):
        """Load index and chunks from disk."""
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)
        print(f"Loaded {self.index.ntotal} chunks from: {path}")


if __name__ == "__main__":
    from src.document_loader import load_and_chunk_pdf
    from src.embedder import embed_chunks, embed_query
    import sys

    pdf_path = sys.argv[1]

    # Full pipeline test
    chunks = load_and_chunk_pdf(pdf_path)
    embeddings = embed_chunks(chunks)

    store = VectorStore()
    store.add(chunks, embeddings)

    # Test a search
    question = "What is the main topic of this document?"
    query_emb = embed_query(question)
    results = store.search(query_emb, top_k=3)

    print(f"\nQuestion: '{question}'")
    print(f"\nTop 3 relevant chunks:")
    for i, r in enumerate(results):
        print(f"\nRank {i+1} (score: {r['score']:.4f}):")
        print(r['chunk']['text'][:200])