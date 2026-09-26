from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(texts):
    return model.encode(texts).tolist()


def create_query_embedding(text):
    return model.encode([text])[0].tolist()