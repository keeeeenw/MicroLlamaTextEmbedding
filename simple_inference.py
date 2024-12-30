from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("keeeeenw/MicroLlama-text-embedding")
# Run inference
sentences = [
    'How do I attract a girl?',
    'How can I attract girls?',
    "Why isn't my iPhone 5 charging?",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 1024]
print(embeddings)

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
print(similarities)
