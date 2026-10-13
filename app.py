import os

# LOAD DOCUMENTS
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

# CHUNK DOCUMENTS
chunks = []

for doc in documents:

    words = doc["content"].split()

    chunk_size = 5

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(words[i:i+chunk_size])

        chunks.append(
            {
                "source": doc["filename"],
                "text": chunk
            }
        )

# ASK QUESTION
question = input("Ask a question: ")

# SIMPLE RETRIEVAL
results = []

for chunk in chunks:

    if question.lower() in chunk["text"].lower():

        results.append(chunk)

print("\nResults:\n")

for result in results:
    print(result)