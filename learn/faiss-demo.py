import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Simulate document chunks
chunks = [
    "Refunds are processed within 5 to 7 business days.",
    "To return a product, visit our returns portal online.",
    "Our products are available in red, blue, and green colors.",
    "Shipping takes 3 to 5 business days within the US.",
    "You can track your order using the tracking number in your email.",
    "Refund eligibility requires the product to be unused and in original packaging.",
]

# Step 1: Convert chunks to embeddings
embeddings = model.encode(chunks)
print(f"Embeddings shape: {embeddings.shape}")  # (6, 384)

# Step 2: Build FAISS index
dimension = embeddings.shape[1]  # 384
index = faiss.IndexFlatL2(dimension)  # L2 = euclidean distance
index.add(embeddings.astype('float32'))
print(f"Chunks indexed in FAISS: {index.ntotal}")

# Step 3: Search with a question
question = "How long does a refund take?"
question_embedding = model.encode([question]).astype('float32')

# Get top 3 most similar chunks
k = 3
distances, indices = index.search(question_embedding, k)

print(f"\nQuestion: '{question}'")
print(f"\nTop {k} relevant chunks:")
for i, idx in enumerate(indices[0]):
    print(f"\n  Rank {i+1} (distance: {distances[0][i]:.4f}):")
    print(f"  {chunks[idx]}")