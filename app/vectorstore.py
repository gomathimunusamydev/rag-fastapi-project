import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension=384):
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []

    def add(self, embeddings, texts):

        if not embeddings or not texts:
            return

        vectors = np.array(embeddings).astype("float32")

        self.index.add(vectors)

        self.texts.extend(texts)

    def search(self, query_embedding, top_k=3):

        if len(self.texts) == 0:
            return []

        query_vector = np.array(
            [query_embedding]
        ).astype("float32")

        k = min(top_k, len(self.texts))

        distances, indices = self.index.search(
            query_vector,
            k
        )

        results = []

        for index in indices[0]:

            if 0 <= index < len(self.texts):
                results.append(self.texts[index])

        return results