from app.embeddings import create_query_embedding
from app.vectorstore import VectorStore


vector_store = VectorStore()


def add_chunks_to_vectorstore(chunks, embeddings):
    vector_store.add(
        embeddings,
        chunks
    )


def retrieve_relevant_chunks(question, top_k=3):

    query_embedding = create_query_embedding(question)

    results = vector_store.search(
        query_embedding,
        top_k
    )

    return results
def generate_answer(question, context):
    import requests

    prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]