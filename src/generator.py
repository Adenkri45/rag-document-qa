import os
from groq import Groq


def generate_answer(question: str, retrieved_chunks: list[dict]) -> dict:
    """
    Given a question and retrieved chunks, generate a grounded answer using Groq.
    """
    # Build context from retrieved chunks
    context = ""
    for i, result in enumerate(retrieved_chunks):
        context += f"\n[Source {i+1}]:\n{result['chunk']['text']}\n"

    # Build prompt
    prompt = f"""You are a helpful assistant that answers questions based strictly on the provided document context.

Context from document:
{context}

Question: {question}

Instructions:
- Answer based only on the context provided above
- If the answer is not in the context, say "I couldn't find this information in the document"
- Mention which source(s) you used in your answer
- Be concise and clear

Answer:"""

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )

    return {
        "question": question,
        "answer": response.choices[0].message.content,
        "sources": [r['chunk']['text'][:150] for r in retrieved_chunks],
        "model": response.model
    }


if __name__ == "__main__":
    from src.document_loader import load_and_chunk_pdf
    from src.embedder import embed_chunks, embed_query
    from src.vector_store import VectorStore
    import sys

    pdf_path = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) > 2 else "What is the main topic of this document?"

    # Full RAG pipeline
    print("\n=== RAG Pipeline ===")
    chunks = load_and_chunk_pdf(pdf_path)
    embeddings = embed_chunks(chunks)

    store = VectorStore()
    store.add(chunks, embeddings)

    query_emb = embed_query(question)
    results = store.search(query_emb, top_k=3)

    print(f"\nRetrieved {len(results)} chunks, generating answer...")
    response = generate_answer(question, results)

    print(f"\n{'='*50}")
    print(f"Question: {response['question']}")
    print(f"\nAnswer: {response['answer']}")
    print(f"\nSources used:")
    for i, src in enumerate(response['sources']):
        print(f"  [{i+1}] {src}...")
    print(f"\nModel: {response['model']}")