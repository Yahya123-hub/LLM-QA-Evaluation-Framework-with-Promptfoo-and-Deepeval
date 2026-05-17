from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from rag.docs import documents

model = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = model.encode(documents)

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))


def retrieve_context(query, top_k=1):
    query_vector = model.encode([query])
    distances, indices = index.search(np.array(query_vector), top_k)

    return [documents[i] for i in indices[0]]