document = """
Artificial intelligence is transforming industries worldwide.
Companies are investing billions in AI research and development.
Machine learning models require large amounts of training data.
Deep learning has revolutionized computer vision and NLP tasks.
Natural language processing enables computers to understand human text.
Transformers are the backbone of modern NLP models like BERT and GPT.
RAG combines retrieval systems with language model generation.
Vector databases store embeddings for fast similarity search.
FastAPI is a modern Python framework for building REST APIs.
Streamlit allows rapid development of data science web applications.
"""

def chunk_text(text, chunk_size=30, overlap=10):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


chunks = chunk_text(document)

print(f"Document words: {len(document.split())}")
print(f"Total chunks: {len(chunks)}")
print(f"\n--- Chunks ---")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1} ({len(chunk.split())} words):")
    print(chunk)