from src.parsing.pdf_parser import FinancialPDFParser

if __name__ == "__main__":
    parser = FinancialPDFParser()
    chunks = parser.extract_chunks("data/samples/sample_prospectus.pdf")
    
    tables = [c for c in chunks if c.chunk_type.value == "table"]
    texts = [c for c in chunks if c.chunk_type.value == "text"]

    print(f"Total chunks : {len(chunks)}")
    print(f"Blocs texte : {len(texts)}")
    print(f"Tableaux extraits : {len(tables)}")

    for t in tables:
        print(f"\n[TABLEAU] Page {t.page_number} | Image : {t.image_path}")
        print(t.text_content[:200] + "...\n")