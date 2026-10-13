import json
import os

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load documents
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

# Chunk documents
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

# Load evaluation questions
with open("eval_questions.json", "r") as f:
    evaluation_set = json.load(f)

passed = 0

for test in evaluation_set:

    question = test["question"]
    expected_source = test["expected_source"]

    question_embedding = model.encode(question)

    scores = cos_sim(question_embedding, chunk_embeddings)
    scores = scores[0]

    best_index = scores.argmax()

    returned_source = chunks[best_index]["source"]

    if returned_source == expected_source:
        print(f"PASS: {question}")
        passed += 1
    else:
        print(
            f"FAIL: {question} | Expected={expected_source} | Got={returned_source}"
        )

accuracy = (passed / len(evaluation_set)) * 100

print("\n====================")
print(f"Accuracy: {accuracy:.2f}%")
print("====================")