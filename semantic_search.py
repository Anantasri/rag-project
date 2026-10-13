from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

chunks = [
    "Docker is a container platform",
    "Linux is an operating system",
    "DevOps uses automation and CI/CD"
]

# Create embeddings
chunk_embeddings = model.encode(chunks)

# User question
question = input("Ask a question: ")

question_embedding = model.encode(question)

# Calculate similarity
scores = cos_sim(question_embedding, chunk_embeddings)

# Find best match
best_index = scores.argmax()

print("\nBest Match:")
print(chunks[best_index])
