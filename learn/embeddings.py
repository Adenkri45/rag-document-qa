from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load a small, fast embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Some example sentences
sentences = [
    "I love playing football",
    "Soccer is my favorite sport",
    "The stock market crashed today",
    "I enjoy watching cricket matches",
    "Bitcoin prices dropped significantly"
]

# Convert to embeddings
embeddings = model.encode(sentences)

print(f"Each sentence becomes a vector of {len(embeddings[0])} numbers\n")

# Compare first sentence against all others
query = "I love playing football"
query_embedding = model.encode([query])

print(f"Query: '{query}'\n")
print("Similarity scores:")
for i, sentence in enumerate(sentences):
    score = cosine_similarity(query_embedding, [embeddings[i]])[0][0]
    print(f"  {score:.4f} → '{sentence}'")
