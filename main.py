from src.retrieval.retriever import LocalRetriever
from src.generation.generator import MultimodalFinancialGenerator

if __name__ == "__main__":
    collection_name = "prospectus_sg_v3"

    query = (
        "Quelles sont l'ensemble des contraintes et limites d'exposition par classe d'actifs "
        "(actions, obligations, devises, dérivés) décrites dans la stratégie d'investissement ? "
        "Pour chaque contrainte, donne l'intervalle d'acceptation lorsqu'il est énoncé."
    )
    print(f"--- 1. QUESTION ANALYSTE ---\n{query}\n")

    # 1. Recherche et reranking directs (sans toucher à l'indexation)
    print("--- 2. RECHERCHE & RERANKING DANS QDRANT ---")
    retriever = LocalRetriever(collection_name=collection_name)
    results = retriever.search(query=query, top_k=4, retrieve_k=35)
    retriever.close()

    print("--- 3. CHUNKS RETENUS ---")
    for idx, r in enumerate(results, 1):
        img_flag = " [TABLEAU/IMAGE]" if r.get("image_path") else ""
        print(f"  -> #{idx} | Page {r['page_number']} | Score: {r['rerank_score']:.4f}{img_flag}")
        print(f"     {r['text_content'][:140]}...\n")

    # 2. Inférence VLM locale
    print("--- 4. SYNTHÈSE VIA QWEN2.5-VL ---")
    generator = MultimodalFinancialGenerator()
    answer = generator.generate_response(query=query, retrieved_context=results)

    print("=" * 60)
    print("RÉPONSE DU SYSTÈME :")
    print("=" * 60)
    print(answer)