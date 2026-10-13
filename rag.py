import os
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load docs
documents = []

for file in os.listdir("data"):

    path = os.path.join("data", file)

    with open(path, "r") as f:
        text = f.read()

    documents.append(
        {
            "filename": file,
            "content": text
        }
    )

# Chunk docs
chunks = []

for doc in documents:

    words = doc["content"].split()

    chunk_size = 5

    for i in range(0, len(words), chunk_size):

        chunk_text = " ".join(words[i:i+chunk_size])

        chunks.append(
            {
                "source": doc["filename"],
                "text": chunk_text
            }
        )

# Create embeddings
chunk_texts = [chunk["text"] for chunk in chunks]

chunk_embeddings = model.encode(chunk_texts)

# Ask question
question = input("Question: ")

question_embedding = model.encode(question)

# Similarity search
scores = cos_sim(question_embedding, chunk_embeddings)

scores = scores[0]

top_indices = scores.argsort(descending=True)[:3]

print("\nTop Matches:\n")

for idx in top_indices:

    print("-------------------")
    print("Source:", chunks[idx]["source"])
    print("Text:", chunks[idx]["text"])
    print("Score:", float(scores[idx]))

