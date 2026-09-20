"""BM25 en Python pur, pour compléter la recherche dense (recherche « hybride »).

La recherche dense est faible sur les termes exacts (« UCITS », « 35 % », noms de classes de
parts) ; BM25 les rattrape. Le corpus d'un prospectus tient en quelques milliers de chunks :
un index en mémoire, reconstruit quand la collection change, est plus simple et plus rapide
qu'une dépendance supplémentaire.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Sequence

from src.utils import normalize_text

_WORD = re.compile(r"[a-z0-9]+")

# Mots vides FR/EN (déjà sans accents, car le texte est normalisé avant).
_STOP = frozenset(
    """
    le la les l un une des du de d et ou en au aux a ce cet cette ces son sa ses leur leurs
    qui que quoi dont ou par pour sur sous dans avec sans se ne pas plus comme est sont ete etre
    il elle ils elles on nous vous je tu y quel quels quelle quelles chaque tout toute tous toutes
    donne donner decrit decrites decrits
    the of and to in is are be for on with as by at or an
    """.split()
)


def _stem(token: str) -> str:
    """Normalisation minimale du pluriel (actions/action, derives/derive)."""
    if len(token) > 4 and token.endswith(("s", "x")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    tokens = _WORD.findall(normalize_text(text))
    return [_stem(t) for t in tokens if t not in _STOP and (len(t) > 1 or t.isdigit())]


class BM25Index:
    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.n = len(documents)
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._doc_len: list[int] = []
        for i, doc in enumerate(documents):
            tokens = tokenize(doc)
            self._doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self._postings[term].append((i, tf))
        self._avg_len = (sum(self._doc_len) / self.n) if self.n else 0.0

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Retourne [(index_du_document, score)] triés par score décroissant."""
        if not self.n or self._avg_len == 0:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for i, tf in postings:
                norm = tf + self.k1 * (1 - self.b + self.b * self._doc_len[i] / self._avg_len)
                scores[i] += idf * tf * (self.k1 + 1) / norm
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
