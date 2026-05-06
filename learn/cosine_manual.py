import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

s1 = "I love playing football"
s2 = "Soccer is my favorite sport"
s3 = "The stock market crashed"

e1, e2, e3 = model.encode([s1, s2, s3])

def cosine_sim(a, b):
    dot_product = np.dot(a, b)
    magnitude = np.linalg.norm(a) * np.linalg.norm(b)
    return dot_product / magnitude

print(f"'{s1}' vs '{s2}'")
print(f"Similarity: {cosine_sim(e1, e2):.4f}\n")

print(f"'{s1}' vs '{s3}'")
print(f"Similarity: {cosine_sim(e1, e3):.4f}")