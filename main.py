from src.retrieval.retriever import LocalRetriever
from src.generation.generator import MultimodalFinancialGenerator

if __name__ == "__main__":
    query = "Quels sont les frais de gestion et autres coûts administratifs ou d'exploitation ?"
    print(f"\n[1] Question : {query}")

    # 1. Retrieval + Reranking
    print("\n[2] Recherche sémantique et reranking dans Qdrant...")
    retriever = LocalRetriever()
    results = retriever.search(query=query, top_k=2, retrieve_k=10)
    retriever.close()

    for idx, r in enumerate(results, 1):
        img_info = f" | Image: {r['image_path']}" if r.get("image_path") else ""
        print(f"    -> Chunk retenu #{idx} (Page {r['page_number']}, Score {r['rerank_score']:.4f}{img_info})")

    # 2. Génération VLM locale
    print("\n[3] Génération de la réponse via Qwen2.5-VL (Ollama)...")
    generator = MultimodalFinancialGenerator()
    answer = generator.generate_response(query=query, retrieved_context=results)

    print("\n" + "=" * 60)
    print("RÉPONSE DU SYSTÈME RAG :")
    print("=" * 60)
    print(answer)