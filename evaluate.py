"""Évaluation reproductible : mesurer avant/après chaque changement de chunking ou de retrieval.

Format du fichier de questions (voir eval/questions.example.json) :
[
  {
    "question": "Quelle est la limite d'exposition aux actions ?",
    "pdf": "data/samples/prospectus.pdf",          # ou "doc_id": "3fa2c1d9e0b7a412"
    "expected_pages": [12, 13],                     # pages où se trouve la réponse
    "expected_terms": ["35 %", "actions"]           # termes/chiffres que la réponse doit contenir
  }
]

Métriques (retrieval seul, rapide) :
  - page_hit@k : au moins un passage retenu provient d'une page attendue ;
  - MRR        : 1/rang du premier passage venant d'une page attendue ;
  - term_recall: part des termes attendus présents dans le contexte envoyé au VLM.
Avec --generate, `answer_recall` mesure la même chose sur la réponse finale du VLM.

Usage :
    uv run python evaluate.py eval/questions.json --top-k 4 --retrieve-k 30
    uv run python evaluate.py eval/questions.json --generate --out eval/results.json
"""
import argparse
import json
import statistics
from pathlib import Path

from src.generation.generator import MultimodalFinancialGenerator
from src.indexing.indexer import LocalVectorIndexer
from src.parsing.dispatcher import DocumentDispatcher
from src.retrieval.retriever import LocalRetriever
from src.utils import Timings, file_sha256, normalize_text


def term_recall(terms: list[str], text: str) -> float:
    if not terms:
        return 1.0
    haystack = normalize_text(text).replace(" %", "%")
    found = sum(1 for t in terms if normalize_text(t).replace(" %", "%") in haystack)
    return found / len(terms)


def pages_of(hit: dict) -> set[int]:
    start = hit.get("page_number") or 0
    end = hit.get("page_end") or start
    return set(range(start, end + 1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("questions", help="Fichier JSON de questions")
    p.add_argument("--collection", default="financial_docs_v2")
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--retrieve-k", type=int, default=30)
    p.add_argument("--decompose", choices=["off", "rules", "llm"], default=None)
    p.add_argument("--neighbors", type=int, default=None)
    p.add_argument("--generate", action="store_true", help="Appelle aussi le VLM (lent)")
    p.add_argument("--out", help="Écrit le détail des résultats en JSON")
    args = p.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    indexer = LocalVectorIndexer(collection_name=args.collection)
    retriever = LocalRetriever(collection_name=args.collection)
    generator = MultimodalFinancialGenerator() if args.generate else None
    dispatcher = DocumentDispatcher()

    rows = []
    for q in questions:
        doc_id = q.get("doc_id")
        if q.get("pdf"):
            doc_id = file_sha256(q["pdf"])
            if not indexer.is_indexed(doc_id):
                print(f"Indexation de {q['pdf']}...")
                indexer.index_chunks(dispatcher.process(q["pdf"], doc_id=doc_id))

        timings = Timings()
        hits = retriever.search(
            q["question"],
            top_k=args.top_k,
            retrieve_k=args.retrieve_k,
            doc_ids=[doc_id] if doc_id else None,
            decompose=args.decompose,
        )
        timings.durations.update(retriever.last_timings.durations)
        context = retriever.build_context(hits, window=args.neighbors)

        expected_pages = set(q.get("expected_pages", []))
        first_rank = next((i for i, h in enumerate(hits, 1) if pages_of(h) & expected_pages), None)
        row = {
            "question": q["question"],
            "page_hit": bool(first_rank) if expected_pages else None,
            "mrr": (1.0 / first_rank if first_rank else 0.0) if expected_pages else None,
            "term_recall": term_recall(q.get("expected_terms", []), "\n".join(c.get("text_content") or "" for c in context)),
            "retrieval_s": sum(timings.durations.values()),
        }
        if generator:
            prepared = generator.prepare(q["question"], context)
            with timings.measure("génération"):
                answer = generator.generate_response(q["question"], context)
            row["answer_recall"] = term_recall(q.get("expected_terms", []), answer)
            row["unsupported_figures"] = generator.unsupported_figures(answer, prepared)
            row["generation_s"] = timings.durations["génération"]
        rows.append(row)
        print(f"- {q['question'][:70]!r}: page_hit={row['page_hit']} mrr={row['mrr']} term_recall={row['term_recall']:.2f}")

    def avg(key: str) -> float | None:
        values = [r[key] for r in rows if r.get(key) is not None]
        return statistics.mean(values) if values else None

    print("\n=== RÉSUMÉ ===")
    for key in ("page_hit", "mrr", "term_recall", "answer_recall", "retrieval_s", "generation_s"):
        value = avg(key)
        if value is not None:
            print(f"{key:>14}: {value:.3f}")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Détail écrit dans {args.out}")


if __name__ == "__main__":
    main()
