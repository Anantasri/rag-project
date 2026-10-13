def chunk_documents(documents, chunk_size=5):

    chunks = []

    for doc in documents:

        words = doc["content"].split()

        for i in range(0, len(words), chunk_size):

            chunks.append({
                "source": doc["filename"],
                "text": " ".join(words[i:i+chunk_size])
            })

    return chunks
