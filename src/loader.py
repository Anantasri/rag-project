import os

def load_documents(folder="data"):

    documents = []

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        with open(path, "r") as f:
            text = f.read()

        documents.append({
            "filename": file,
            "content": text
        })

    return documents