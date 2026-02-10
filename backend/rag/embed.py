from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def build_index(text_path):
    with open(text_path, "r") as f:
        docs = [line.strip() for line in f if line.strip()]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(docs)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return index, docs, model