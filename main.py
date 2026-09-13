from src.parsing.pdf_parser import FinancialPDFParser
from src.indexing.indexer import LocalVectorIndexer
from src.retrieval.retriever import LocalRetriever

if __name__ == "__main__":
    pdf_path = "data/samples/sample_prospectus.pdf"

    # 1. Parsing enrichi
    print("--- 1. PARSING & EXTRACTION DU PDF ---")
    parser = FinancialPDFParser()
    chunks = parser.extract_chunks(pdf_path)
    
    tables = [c for c in chunks if c.chunk_type.value == "table"]
    print(f"Total chunks : {len(chunks)} | Tableaux détectés : {len(tables)}")
    for t in tables:
        print(f"-> Tableau page {t.page_number} | Image : {t.image_path}")

    # 2. Réindexation
    print("\n--- 2. INDEXATION DANS QDRANT ---")
    indexer = LocalVectorIndexer()
    indexer.index_chunks(chunks)
    indexer.client.close()

    # 3. Retrieval & Reranking
    print("\n--- 3. TEST DE RECHERCHE SÉMANTIQUE ---")
    retriever = LocalRetriever()
    query = "Quels sont les frais de gestion et autres coûts administratifs ou d'exploitation ?"
    print(f"Requête : '{query}'\n")

    results = retriever.search(query=query, top_k=3, retrieve_k=10)

    for i, res in enumerate(results, start=1):
        print(f"--- Résultat #{i} (Score Reranker : {res['rerank_score']:.4f}) ---")
        print(f"Page : {res['page_number']} | Type : {res['chunk_type']}")
        if res["image_path"]:
            print(f"Image liée : {res['image_path']}")
        print(f"Contenu : {res['text_content'][:200]}...\n")

    retriever.close()