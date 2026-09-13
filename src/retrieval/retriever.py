from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder


class LocalRetriever:
    """Moteur de recherche hybride combinant similarité vectorielle et Reranking."""

    def __init__(
        self,
        collection_name: str = "financial_docs",
        storage_path: str = "qdrant_storage",
        embedding_model: str = "BAAI/bge-m3",
        reranker_model: str = "BAAI/bge-reranker-base",
    ):
        self.collection_name = collection_name

        # 1. Bi-Encoder pour le retrieval initial rapide
        self.encoder = SentenceTransformer(embedding_model, device = "cpu")
        self.client = QdrantClient(path=storage_path)

        # 2. Cross-Encoder pour affiner et reclasser précisément les résultats
        print(f"Chargement du Reranker : {reranker_model}...")
        self.reranker = CrossEncoder(reranker_model, device = "cpu")

    def search(self, query: str, top_k: int = 3, retrieve_k: int = 10) -> list[dict]:
        """Récupère un Top-K initial large dans Qdrant puis applique le Reranker."""
        # Étape 1 : Récupération large (Top-10) par similarité cosinus
        query_vector = self.encoder.encode(query).tolist()
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=retrieve_k,
        )

        initial_docs = []
        for point in response.points:
            initial_docs.append(
                {
                    "page_number": point.payload.get("page_number"),
                    "chunk_type": point.payload.get("chunk_type"),
                    "text_content": point.payload.get("text_content"),
                    "image_path": point.payload.get("image_path"),
                    "source_document": point.payload.get("source_document"),
                }
            )

        if not initial_docs:
            return []

        # Étape 2 : Préparation des paires (Requête, Passage) pour le Cross-Encoder
        pairs = [[query, doc["text_content"] or ""] for doc in initial_docs]
        rerank_scores = self.reranker.predict(pairs)

        # Étape 3 : Attribution du nouveau score et tri
        for idx, doc in enumerate(initial_docs):
            doc["rerank_score"] = float(rerank_scores[idx])

        # Tri décroissant selon la pertinence calculée par le reranker
        sorted_docs = sorted(initial_docs, key=lambda x: x["rerank_score"], reverse=True)

        return sorted_docs[:top_k]

    def close(self):
        self.client.close()