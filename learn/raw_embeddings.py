from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

sentence = "I love playing football"
embedding = model.encode(sentence)

print(f"Type: {type(embedding)}")
print(f"Shape: {embedding.shape}")
print(f"\nFirst 10 numbers: {embedding[:10]}")
print(f"Last 10 numbers: {embedding[-10:]}")
print(f"\nMin value: {embedding.min():.4f}")
print(f"Max value: {embedding.max():.4f}")