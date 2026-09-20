"""Tests du retriever et du générateur avec de faux clients (ni Qdrant, ni modèles, ni Ollama requis).

Lancer : `uv run pytest tests -q`  ou  `python tests/test_retrieval_generation.py`
"""
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

# Si qdrant-client n'est pas installé, on fournit de quoi importer les classes de filtre.
try:  # pragma: no cover
    import qdrant_client.models  # noqa: F401
except ImportError:  # pragma: no cover
    import types

    class _Obj:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    qc = types.ModuleType("qdrant_client")
    qm = types.ModuleType("qdrant_client.models")
    for _name in ("Filter", "FieldCondition", "MatchValue", "MatchAny", "Range"):
        setattr(qm, _name, type(_name, (_Obj,), {}))
    qc.models = qm
    sys.modules["qdrant_client"] = qc
    sys.modules["qdrant_client.models"] = qm

from src.config import Settings  # noqa: E402
from src.generation.generator import MultimodalFinancialGenerator  # noqa: E402
from src.retrieval import retriever as retriever_module  # noqa: E402
from src.retrieval.lexical import tokenize  # noqa: E402


# ---------------------------------------------------------------- faux composants
class FakeEncoder:
    """Sac de mots haché sur 64 dimensions, normalisé (suffisant pour tester la mécanique)."""

    def encode(self, text, normalize_embeddings=True, show_progress_bar=False, **_):
        vec = np.zeros(64)
        for tok in tokenize(text):
            vec[hash(tok) % 64] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec


class FakeReranker:
    def predict(self, pairs, batch_size=16, show_progress_bar=False):
        scores = []
        for query, passage in pairs:
            q, p = set(tokenize(query)), set(tokenize(passage))
            scores.append(len(q & p) / max(len(q), 1))
        return scores


class FakeClient:
    def __init__(self, payloads):
        self.points = []
        for i, payload in enumerate(payloads):
            vec = FakeEncoder().encode(payload.get("retrieval_text") or payload["text_content"])
            self.points.append(SimpleNamespace(id=f"p{i}", payload=payload, vector=vec))

    @staticmethod
    def _ok(payload, flt):
        if flt is None:
            return True
        for cond in flt.must:
            value = payload.get(cond.key)
            match = getattr(cond, "match", None)
            rng = getattr(cond, "range", None)
            if match is not None and hasattr(match, "value") and value != match.value:
                return False
            if match is not None and hasattr(match, "any") and value not in match.any:
                return False
            if rng is not None and (value is None or not (rng.gte <= value <= rng.lte)):
                return False
        return True

    def _filtered(self, flt):
        return [p for p in self.points if self._ok(p.payload, flt)]

    def count(self, collection, count_filter=None, exact=True):
        return SimpleNamespace(count=len(self._filtered(count_filter)))

    def scroll(self, collection, scroll_filter=None, limit=100, offset=None, **_):
        pts = self._filtered(scroll_filter)
        start = offset or 0
        page = pts[start : start + limit]
        return page, (start + limit if start + limit < len(pts) else None)

    def query_points(self, collection, query, limit, query_filter=None, with_payload=True):
        q = np.array(query)
        pts = sorted(self._filtered(query_filter), key=lambda p: float(q @ p.vector), reverse=True)
        return SimpleNamespace(points=pts[:limit])


def make_payload(doc, idx, page, text, chunk_type="text", section=None, needs_image=False, image=None):
    return {
        "doc_id": doc,
        "source_document": f"{doc}.pdf",
        "chunk_index": idx,
        "page_number": page,
        "page_end": None,
        "chunk_type": chunk_type,
        "section": section,
        "text_content": text,
        "retrieval_text": f"{section}\n{text}" if section else text,
        "image_path": image,
        "needs_image": needs_image,
    }


CORPUS = [
    make_payload("D1", 0, 1, "Présentation générale du fonds et de sa société de gestion."),
    make_payload("D1", 1, 4, "Le compartiment actions est limité à 35 % de l'actif net.", section="Limites"),
    make_payload("D1", 2, 4, "Suite des règles de composition et de surveillance quotidienne du portefeuille."),
    make_payload("D1", 3, 5, "Les obligations et titres de taux représentent entre 65 % et 100 % de l'actif.", section="Limites"),
    make_payload("D1", 4, 5, "Le risque de change sur les devises peut atteindre 20 % de l'actif net.", section="Limites"),
    make_payload("D1", 5, 6, "Les instruments dérivés sont limités à un engagement de 100 % de l'actif net.", section="Limites"),
    make_payload("D1", 6, 9, "| Frais | Taux |\n| Frais de gestion annuels | 1,5 % |", "table", section="Frais"),
    # quasi-doublon du chunk 3 (ex. répétition d'un en-tête ou d'un tableau)
    make_payload("D1", 7, 11, "Les obligations et titres de taux représentent entre 65 % et 100 % de l'actif.", section="Limites"),
    make_payload("D2", 0, 2, "Le compartiment actions est limité à 50 % dans cet autre document.", section="Limites"),
]

QUERY = "Quelles limites par classe d'actifs (actions, obligations, devises, dérivés) ?"


def make_retriever(corpus=None, **overrides):
    client = FakeClient(corpus or CORPUS)
    retriever_module.get_client = lambda *_a, **_k: client
    retriever_module.get_encoder = lambda *_a, **_k: FakeEncoder()
    retriever_module.get_reranker = lambda *_a, **_k: FakeReranker()
    return retriever_module.LocalRetriever("test", settings=Settings(**overrides))


# ---------------------------------------------------------------- retriever
def test_doc_filter_isolates_documents():
    r = make_retriever()
    hits = r.search("actions limite", top_k=5, retrieve_k=10, doc_ids=["D1"], decompose="off")
    assert hits and all(h["doc_id"] == "D1" for h in hits)
    both = r.search("actions limite", top_k=5, retrieve_k=10, doc_ids=["D1", "D2"], decompose="off")
    assert {h["doc_id"] for h in both} == {"D1", "D2"}


def test_multi_query_covers_each_asset_class():
    r = make_retriever()
    hits = r.search(QUERY, top_k=4, retrieve_k=12, doc_ids=["D1"], decompose="rules")
    assert len(r.last_queries) == 5  # question + 4 sous-requêtes
    covered = {h["chunk_index"] for h in hits}
    assert {1, 4, 5} <= covered  # actions, devises, dérivés
    assert covered & {3, 7}  # obligations


def test_each_subquery_gets_a_guaranteed_slot():
    # Trois passages « actions » écrasent le classement global ; sans place garantie, « devises » disparaît.
    corpus = [
        make_payload("D1", 0, 1, "Limites par classe d'actifs : actions plafonnées à 35 %."),
        make_payload("D1", 1, 2, "Limites par classe d'actifs : actions, seuil de 30 % maximum."),
        make_payload("D1", 2, 3, "Limites classe d'actifs actions 25 % de l'actif net."),
        make_payload("D1", 3, 4, "Limites classe : devises, exposition de 20 %."),
    ]
    r = make_retriever(corpus)
    hits = r.search("Limites par classe d'actifs (actions, devises) ?", top_k=3, retrieve_k=10, doc_ids=["D1"], decompose="rules")
    assert 3 in {h["chunk_index"] for h in hits}, [h["chunk_index"] for h in hits]
    # Sans décomposition, le classement global suffit à l'exclure : preuve que la garantie a un effet.
    plain = r.search("Limites par classe d'actifs (actions, devises) ?", top_k=3, retrieve_k=10, doc_ids=["D1"], decompose="off")
    assert 3 not in {h["chunk_index"] for h in plain}


def test_near_duplicates_are_collapsed():
    r = make_retriever()
    hits = r.search("obligations titres de taux 65 % 100 %", top_k=5, retrieve_k=12, doc_ids=["D1"], decompose="off")
    assert sum(1 for h in hits if h["chunk_index"] in (3, 7)) == 1


def test_hybrid_and_dense_only_both_work():
    for hybrid in (True, False):
        r = make_retriever(hybrid=hybrid)
        hits = r.search("frais de gestion annuels", top_k=2, retrieve_k=8, doc_ids=["D1"], decompose="off")
        assert hits[0]["chunk_index"] == 6 and hits[0]["chunk_type"] == "table"


def test_lexical_cache_invalidated_when_collection_changes():
    r = make_retriever()
    r.search("actions", top_k=2, retrieve_k=8, doc_ids=["D1"], decompose="off")
    first = next(iter(r._lexical_cache.values()))[2]
    r.search("actions", top_k=2, retrieve_k=8, doc_ids=["D1"], decompose="off")
    assert next(iter(r._lexical_cache.values()))[2] is first  # réutilisé
    r.client.points.append(
        SimpleNamespace(id="new", payload=make_payload("D1", 8, 12, "Nouveau chunk"), vector=FakeEncoder().encode("Nouveau chunk"))
    )
    r.search("actions", top_k=2, retrieve_k=8, doc_ids=["D1"], decompose="off")
    assert next(iter(r._lexical_cache.values()))[2] is not first  # reconstruit


def test_build_context_adds_ordered_neighbors_within_budget():
    r = make_retriever()
    hits = r.search("devises risque de change", top_k=1, retrieve_k=8, doc_ids=["D1"], decompose="off")
    assert hits[0]["chunk_index"] == 4
    ctx = r.build_context(hits, window=1, max_chars=5000)
    assert [c["chunk_index"] for c in ctx] == [3, 4, 5]  # ordre du document
    assert [c["is_neighbor"] for c in ctx] == [True, False, True]

    tight = r.build_context(hits, window=1, max_chars=len(hits[0]["text_content"]) + 5)
    assert [c["chunk_index"] for c in tight] == [4]  # budget épuisé : pas de voisin
    assert r.build_context(hits, window=0) == hits


# ---------------------------------------------------------------- générateur
def _png(path, size=(2400, 1500)):
    from PIL import Image

    Image.new("RGB", size, "white").save(path)
    return str(path)


def _extract(idx, page, text, score=None, **kw):
    return {
        "doc_id": "D1", "source_document": "D1.pdf", "chunk_index": idx, "page_number": page,
        "page_end": None, "chunk_type": kw.pop("chunk_type", "text"), "section": kw.pop("section", None),
        "text_content": text, "image_path": kw.pop("image_path", None), "needs_image": kw.pop("needs_image", False),
        "rerank_score": score, "is_neighbor": score is None, **kw,
    }


def test_prepare_orders_context_labels_and_resizes_images():
    with tempfile.TemporaryDirectory() as tmp:
        img = _png(Path(tmp) / "tab.png")
        extracts = [
            _extract(6, 9, "| Frais | 1,5 % |", 0.8, chunk_type="table", section="Frais", image_path=img, needs_image=True),
            _extract(1, 4, "Actions limitées à 35 %.", 0.9, section="Limites"),
        ]
        gen = MultimodalFinancialGenerator(settings=Settings(max_images=2, max_image_side=1280))
        prepared = gen.prepare("Question ?", extracts, mode="constraints", image_policy="auto")

        user = prepared.messages[1]
        text = user["content"]
        assert text.index("Extrait #1 | Page 4") < text.index("Extrait #2 | Page 9")  # ordre du document
        assert "Section : Limites" in text
        assert "image 1 = capture du tableau de l'Extrait #2 (page 9)" in text
        assert prepared.image_extract_numbers == [2] and len(user["images"]) == 1

        from PIL import Image
        import io

        w, h = Image.open(io.BytesIO(user["images"][0])).size
        assert max(w, h) == 1280  # redimensionné, ratio conservé
        assert (w, h) == (1280, 800)
        assert prepared.options["num_ctx"] == gen.settings.num_ctx  # constant : pas de rechargement Ollama

        # Politiques d'images
        assert "images" not in gen.prepare("Q", extracts, image_policy="never").messages[1]
        unreliable_only = [dict(e, needs_image=False) for e in extracts]
        assert "images" not in gen.prepare("Q", unreliable_only, image_policy="auto").messages[1]
        assert len(gen.prepare("Q", unreliable_only, image_policy="always").messages[1]["images"]) == 1


def test_prepare_shares_one_image_across_split_table_chunks():
    with tempfile.TemporaryDirectory() as tmp:
        img = _png(Path(tmp) / "tab.png", (800, 600))
        extracts = [
            _extract(6, 9, "| a | b |", 0.9, chunk_type="table", image_path=img, needs_image=True),
            _extract(7, 9, "| c | d |", 0.8, chunk_type="table", image_path=img, needs_image=True),
        ]
        prepared = MultimodalFinancialGenerator(settings=Settings(max_images=2)).prepare("Q", extracts)
        assert len(prepared.messages[1]["images"]) == 1


def test_prepare_trims_neighbors_first_when_over_budget():
    settings = Settings(num_ctx=2048, num_predict=256)
    big = "mot " * 400  # ~1600 caractères ≈ 530 tokens estimés
    extracts = [
        _extract(1, 4, big, -2.0),  # logits de reranker négatifs : le hit reste prioritaire sur un voisin
        _extract(2, 4, big, None),  # voisins
        _extract(3, 5, big, None),
        _extract(4, 5, big, None),
    ]
    prepared = MultimodalFinancialGenerator(settings=settings).prepare("Question", extracts)
    assert prepared.dropped_extracts >= 1
    kept = [e for e in prepared.extracts]
    assert any(not e["is_neighbor"] for e in kept)  # le hit est conservé
    assert prepared.estimated_tokens <= settings.num_ctx - settings.num_predict - 200


def test_unsupported_figures_uses_prepared_sources():
    gen = MultimodalFinancialGenerator(settings=Settings())
    prepared = gen.prepare("Q", [_extract(1, 4, "Actions limitées à 35 %.", 0.9)])
    answer = "| Actions | 0 % | 35 % | ... |\n| Devises | 0 % | 20 % |"
    assert gen.unsupported_figures(answer, prepared) == ["20 %"]


if __name__ == "__main__":  # exécution sans pytest
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                import traceback

                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
    sys.exit(1 if failures else 0)
