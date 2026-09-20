"""Point d'entrée en ligne de commande.

Exemples :
    uv run python main.py --pdf data/samples/prospectus.pdf
    uv run python main.py --pdf data/samples/prospectus.pdf --query "Quels sont les frais de gestion ?" --mode free
    uv run python main.py --collection financial_docs_v2 --doc-id 3fa2c1d9e0b7a412
"""
import argparse
import logging
import sys
import time
from pathlib import Path

from src.generation.generator import MultimodalFinancialGenerator
from src.indexing.indexer import LocalVectorIndexer
from src.parsing.dispatcher import DocumentDispatcher
from src.retrieval.retriever import LocalRetriever
from src.utils import Timings, file_sha256

DEFAULT_QUERY = (
    "Quelles sont l'ensemble des contraintes et limites d'exposition par classe d'actifs "
    "(actions, obligations, devises, dérivés) décrites dans la stratégie d'investissement ? "
    "Pour chaque contrainte, donne l'intervalle d'acceptation lorsqu'il est énoncé."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG financier local : indexation, recherche et synthèse VLM.")
    p.add_argument("--pdf", help="PDF à indexer (ignoré s'il l'est déjà, sauf --reindex)")
    p.add_argument("--doc-id", help="Identifiant d'un document déjà indexé (sinon : toute la collection)")
    p.add_argument("--collection", default="financial_docs_v2")
    p.add_argument("--query", default=DEFAULT_QUERY)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--retrieve-k", type=int, default=30)
    p.add_argument("--decompose", choices=["off", "rules", "llm"], default=None)
    p.add_argument("--neighbors", type=int, default=None, help="Chunks voisins ajoutés au contexte")
    p.add_argument("--mode", choices=["constraints", "free"], default="constraints")
    p.add_argument("--images", choices=["auto", "always", "never"], default="auto")
    p.add_argument("--reindex", action="store_true", help="Force la ré-indexation du PDF")
    p.add_argument("--retrieval-only", action="store_true", help="Affiche les passages sans appeler le VLM")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    timings = Timings()

    indexer = LocalVectorIndexer(collection_name=args.collection)
    doc_id = args.doc_id

    if args.pdf:
        pdf_path = Path(args.pdf)
        doc_id = file_sha256(pdf_path)
        if args.reindex or not indexer.is_indexed(doc_id):
            print(f"--- 0. INDEXATION de {pdf_path.name} (doc_id={doc_id}) ---")
            with timings.measure("parsing"):
                chunks = DocumentDispatcher().process(str(pdf_path), doc_id=doc_id)
            with timings.measure("vectorisation"):
                indexer.index_chunks(chunks, replace=True)
            print(f"{len(chunks)} chunks indexés.\n")
        else:
            print(f"Document déjà indexé (doc_id={doc_id}), indexation ignorée.\n")

    print(f"--- 1. QUESTION ANALYSTE ---\n{args.query}\n")

    print("--- 2. RECHERCHE HYBRIDE & RERANKING ---")
    retriever = LocalRetriever(collection_name=args.collection)
    hits = retriever.search(
        args.query,
        top_k=args.top_k,
        retrieve_k=args.retrieve_k,
        doc_ids=[doc_id] if doc_id else None,
        decompose=args.decompose,
    )
    timings.durations.update(retriever.last_timings.durations)
    if len(retriever.last_queries) > 1:
        print(f"Sous-requêtes : {retriever.last_queries[1:]}")
    if not hits:
        print("Aucun passage trouvé.")
        return 1

    print("--- 3. CHUNKS RETENUS ---")
    for idx, r in enumerate(hits, 1):
        flag = " [TABLEAU]" if r["chunk_type"] == "table" else ""
        flag += " [IMAGE]" if r.get("needs_image") else ""
        pages = f"{r['page_number']}-{r['page_end']}" if r.get("page_end") else r["page_number"]
        print(f"  -> #{idx} | Page {pages} | Score: {r['rerank_score']:.4f}{flag}")
        if r.get("section"):
            print(f"     Section : {r['section']}")
        print(f"     {(r['text_content'] or '')[:140]!r}...\n")

    if args.retrieval_only:
        print("⏱", timings.summary())
        return 0

    with timings.measure("contexte"):
        context = retriever.build_context(hits, window=args.neighbors)
    generator = MultimodalFinancialGenerator()
    prepared = generator.prepare(args.query, context, mode=args.mode, image_policy=args.images)
    print(
        f"Contexte : {len(prepared.extracts)} extraits, {len(prepared.image_extract_numbers)} image(s), "
        f"~{prepared.estimated_tokens} tokens\n"
    )

    print("--- 4. SYNTHÈSE VIA QWEN2.5-VL ---")
    print("=" * 60)
    start = time.perf_counter()
    answer_parts: list[str] = []
    first = True
    for piece in generator.stream(prepared):
        if first:
            timings.durations["1er token"] = time.perf_counter() - start
            first = False
        answer_parts.append(piece)
        print(piece, end="", flush=True)
    timings.durations["génération"] = time.perf_counter() - start
    print("\n" + "=" * 60)

    unsupported = generator.unsupported_figures("".join(answer_parts), prepared)
    if unsupported:
        print(f"⚠️  Chiffres non retrouvés dans le texte extrait (à vérifier) : {', '.join(unsupported)}")
    print("⏱", timings.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
