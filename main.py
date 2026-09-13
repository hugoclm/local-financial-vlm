from src.parsing.pdf_parser import FinancialPDFParser
from src.indexing.indexer import LocalVectorIndexer

if __name__ == "__main__":
    pdf_file = "data/samples/sample_prospectus.pdf"

    # 1. Parsing du document
    print("--- 1. PARSING DU PDF ---")
    parser = FinancialPDFParser()
    chunks = parser.extract_chunks(pdf_file)
    print(f"Total chunks générés : {len(chunks)}")

    # 2. Vectorisation et indexation dans Qdrant
    print("\n--- 2. INDEXATION DANS QDRANT ---")
    indexer = LocalVectorIndexer()
    indexer.index_chunks(chunks)

    # 3. Vérification de la persistance
    total_in_db = indexer.client.count(collection_name="financial_docs").count
    print(f"\nVérification : {total_in_db} enregistrements trouvés dans la base locale.")