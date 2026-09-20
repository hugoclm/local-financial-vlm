from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional, Sequence

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range

from src.config import Settings, get_settings
from src.models import get_client, get_encoder, get_reranker, index_version
from src.retrieval.lexical import BM25Index
from src.retrieval.query_planner import plan_queries
from src.utils import Timings, normalize_text

logger = logging.getLogger(__name__)


def _payload_text(payload: dict) -> str:
    # `retrieval_text` (section + contenu) ; repli sur `text_content` pour les anciens index.
    return payload.get("retrieval_text") or payload.get("text_content") or ""


class LocalRetriever:
    """Recherche hybride (dense + BM25, fusion RRF) suivie d'un reranking cross-encoder.

    Nouveautés :
    - filtre par document (`doc_ids`) : plus de mélange entre PDF d'une même collection ;
    - BM25 en complément du dense (termes exacts, chiffres, sigles) ;
    - questions multi-parties : sous-requêtes (règles ou LLM), chaque sous-requête garantit
      au moins un passage dans le résultat final ;
    - déduplication des passages quasi identiques ;
    - `build_context` : ajoute les chunks voisins des meilleurs résultats, dans l'ordre du document.
    """

    def __init__(
        self,
        collection_name: str = "financial_docs",
        storage_path: Optional[str] = None,
        embedding_model: Optional[str] = None,
        reranker_model: Optional[str] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.collection_name = collection_name
        self.encoder = get_encoder(embedding_model)
        self.reranker = get_reranker(reranker_model)
        self.client = get_client(storage_path)
        self._lexical_cache: dict[tuple, tuple[int, int, BM25Index, list[str], dict[str, dict]]] = {}
        self.last_timings = Timings()
        self.last_queries: list[str] = []

    # ---------------------------------------------------------------- filtres
    @staticmethod
    def _doc_filter(doc_ids: Optional[Sequence[str]]) -> Optional[Filter]:
        if not doc_ids:
            return None
        if len(doc_ids) == 1:
            cond = FieldCondition(key="doc_id", match=MatchValue(value=doc_ids[0]))
        else:
            cond = FieldCondition(key="doc_id", match=MatchAny(any=list(doc_ids)))
        return Filter(must=[cond])

    # ---------------------------------------------------------------- lexical
    def _lexical(self, flt: Optional[Filter], doc_key: tuple):
        """Index BM25 du (des) document(s) ciblé(s), reconstruit si la collection a changé."""
        count = self.client.count(self.collection_name, count_filter=flt, exact=True).count
        version = index_version(self.collection_name)
        cached = self._lexical_cache.get(doc_key)
        if cached and cached[0] == count and cached[1] == version:
            return cached[2], cached[3], cached[4]

        payloads: dict[str, dict] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection_name,
                scroll_filter=flt,
                limit=512,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                payloads[str(p.id)] = p.payload or {}
            if offset is None:
                break
        ids = list(payloads)
        index = BM25Index([_payload_text(payloads[i]) for i in ids])
        self._lexical_cache[doc_key] = (count, version, index, ids, payloads)
        return index, ids, payloads

    # ------------------------------------------------------------- candidats
    def _candidates(
        self, query: str, k: int, flt: Optional[Filter], lexical
    ) -> list[tuple[str, dict]]:
        vector = self.encoder.encode(query, normalize_embeddings=True, show_progress_bar=False)
        response = self.client.query_points(
            self.collection_name,
            query=vector.tolist(),
            limit=k,
            query_filter=flt,
            with_payload=True,
        )
        dense = [(str(p.id), p.payload or {}) for p in response.points]
        if lexical is None:
            return dense

        index, ids, payloads = lexical
        fused: dict[str, float] = defaultdict(float)
        known: dict[str, dict] = {}
        rrf_k = self.settings.rrf_k
        for rank, (pid, payload) in enumerate(dense):
            fused[pid] += 1.0 / (rrf_k + rank + 1)
            known[pid] = payload
        for rank, (doc_idx, _score) in enumerate(index.search(query, k)):
            pid = ids[doc_idx]
            fused[pid] += 1.0 / (rrf_k + rank + 1)
            known.setdefault(pid, payloads[pid])
        best = sorted(fused, key=fused.__getitem__, reverse=True)[:k]
        return [(pid, known[pid]) for pid in best]

    # ---------------------------------------------------------------- search
    def search(
        self,
        query: str,
        top_k: int = 3,
        retrieve_k: int = 10,
        doc_ids: Optional[Sequence[str]] = None,
        decompose: Optional[str] = None,
    ) -> list[dict]:
        """Retourne les `top_k` meilleurs passages, triés par score de reranking décroissant."""
        timings = Timings()
        self.last_timings = timings
        flt = self._doc_filter(doc_ids)

        with timings.measure("planification"):
            queries = plan_queries(query, decompose or self.settings.decompose_mode, self.settings.ollama_model)
        self.last_queries = queries

        lexical = None
        if self.settings.hybrid:
            with timings.measure("index lexical"):
                lexical = self._lexical(flt, ("|".join(sorted(doc_ids)) if doc_ids else "*",))

        # Chaque sous-requête ramène moins de candidats : le coût du reranking reste ~constant.
        per_query_k = retrieve_k if len(queries) == 1 else max(8, -(-retrieve_k // len(queries)) + 4)
        pool: dict[str, dict] = {}
        with timings.measure("recherche"):
            for qi, q in enumerate(queries):
                for pid, payload in self._candidates(q, per_query_k, flt, lexical):
                    entry = pool.setdefault(pid, {"payload": payload, "queries": []})
                    entry["queries"].append(qi)
        if not pool:
            return []

        pairs: list[list[str]] = []
        owners: list[tuple[str, int]] = []
        for pid, entry in pool.items():
            text = _payload_text(entry["payload"])
            for qi in entry["queries"]:
                pairs.append([queries[qi], text])
                owners.append((pid, qi))
        with timings.measure("rerank"):
            scores = self.reranker.predict(pairs, batch_size=self.settings.rerank_batch_size, show_progress_bar=False)

        best: dict[str, float] = {}
        per_query: dict[int, list[tuple[float, str]]] = defaultdict(list)
        for (pid, qi), score in zip(owners, scores):
            score = float(score)
            best[pid] = max(best.get(pid, float("-inf")), score)
            per_query[qi].append((score, pid))

        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        selected: list[str] = []
        seen_texts: set[str] = set()

        def try_add(pid: str) -> bool:
            key = normalize_text(_payload_text(pool[pid]["payload"]))[:400]
            if pid in selected or key in seen_texts:
                return False
            selected.append(pid)
            seen_texts.add(key)
            return True

        # 1) une place garantie par sous-requête (couverture), si le passage reste pertinent ;
        if len(queries) > 1:
            top_score = ranked[0][1]
            floor = top_score - 0.7 * abs(top_score)
            for qi in range(1, len(queries)):
                for score, pid in sorted(per_query[qi], reverse=True):
                    if score >= floor and try_add(pid):
                        break
                if len(selected) >= top_k:
                    break
        # 2) le reste par score global.
        for pid, _ in ranked:
            if len(selected) >= top_k:
                break
            try_add(pid)

        selected.sort(key=lambda pid: best[pid], reverse=True)
        return [
            self._to_result(
                pool[pid]["payload"],
                best[pid],
                matched=[queries[qi] for qi in pool[pid]["queries"]],
            )
            for pid in selected[:top_k]
        ]

    # ---------------------------------------------------------------- contexte
    def build_context(
        self,
        hits: list[dict],
        window: Optional[int] = None,
        max_chars: Optional[int] = None,
    ) -> list[dict]:
        """Hits + chunks voisins (rang ±window dans le même document), triés dans l'ordre du document.

        Les listes de contraintes s'étalent souvent sur plusieurs chunks : les voisins évitent de
        n'en voir que la moitié. Le budget `max_chars` borne la taille du prompt.
        """
        window = self.settings.neighbor_window if window is None else window
        max_chars = max_chars or self.settings.max_context_chars

        context: dict[tuple, dict] = {}
        used = 0
        for hit in hits:
            context[(hit.get("doc_id"), hit.get("chunk_index"), hit.get("page_number"))] = hit
            used += len(hit.get("text_content") or "")

        if window > 0:
            for hit in hits:  # dans l'ordre des scores : les meilleurs obtiennent le budget en premier
                idx, doc_id = hit.get("chunk_index"), hit.get("doc_id")
                if idx is None or not doc_id:
                    continue  # ancien index sans numérotation
                flt = Filter(
                    must=[
                        FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                        FieldCondition(key="chunk_index", range=Range(gte=idx - window, lte=idx + window)),
                    ]
                )
                points, _ = self.client.scroll(
                    self.collection_name,
                    scroll_filter=flt,
                    limit=2 * window + 3,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in sorted(points, key=lambda p: (p.payload or {}).get("chunk_index", 0)):
                    payload = point.payload or {}
                    key = (payload.get("doc_id"), payload.get("chunk_index"), payload.get("page_number"))
                    size = len(payload.get("text_content") or "")
                    if key in context or used + size > max_chars:
                        continue
                    context[key] = self._to_result(payload, None, is_neighbor=True)
                    used += size

        return sorted(
            context.values(),
            key=lambda r: (r.get("source_document") or "", r.get("doc_id") or "", r.get("chunk_index") or 0, r.get("page_number") or 0),
        )

    # ------------------------------------------------------------------ divers
    @staticmethod
    def _to_result(
        payload: dict,
        score: Optional[float],
        is_neighbor: bool = False,
        matched: Optional[list[str]] = None,
    ) -> dict:
        return {
            "doc_id": payload.get("doc_id"),
            "source_document": payload.get("source_document"),
            "page_number": payload.get("page_number"),
            "page_end": payload.get("page_end"),
            "chunk_index": payload.get("chunk_index"),
            "chunk_type": payload.get("chunk_type"),
            "section": payload.get("section"),
            "text_content": payload.get("text_content"),
            "image_path": payload.get("image_path"),
            "needs_image": bool(payload.get("needs_image")),
            "rerank_score": score,
            "is_neighbor": is_neighbor,
            "matched_queries": matched or [],
        }

    def close(self) -> None:
        """Conservé pour compatibilité : le client est partagé et fermé à la sortie du processus."""
