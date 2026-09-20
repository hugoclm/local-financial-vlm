"""Utilitaires purs (aucune dépendance lourde) : timers, hash, normalisation, vérification de chiffres."""
from __future__ import annotations

import hashlib
import re
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path


class Timings:
    """Petit chronomètre multi-étapes : `with t.measure("rerank"): ...`."""

    def __init__(self) -> None:
        self.durations: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.durations[name] = self.durations.get(name, 0.0) + (time.perf_counter() - start)

    def summary(self) -> str:
        return " · ".join(f"{k} {v:.1f}s" for k, v in self.durations.items())


def file_sha256(path: str | Path, length: int = 16) -> str:
    """Empreinte courte et stable d'un fichier (sert d'identifiant de document)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:length]


def bytes_sha256(data: bytes, length: int = 16) -> str:
    return hashlib.sha256(data).hexdigest()[:length]


def normalize_text(text: str | None) -> str:
    """Minuscules, sans accents, espaces compactés (pour dédoublonnage, BM25, comparaisons)."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text.lower()).strip()


# --------------------------------------------------------------------------------------
# Vérification des chiffres : un pourcentage cité dans la réponse doit exister dans les sources.
# --------------------------------------------------------------------------------------
_NUM = r"\d+(?:[.,]\d+)?"
_PCT_GROUP = re.compile(rf"((?:{_NUM}\s*(?:-|–|—|à|et|to)\s*)*{_NUM})\s*%", re.IGNORECASE)
_TRIVIAL = {"0", "100"}


def _canon(num: str) -> str:
    n = num.replace(",", ".")
    if "." in n:
        n = n.rstrip("0").rstrip(".")
    return n or "0"


def extract_percent_figures(text: str) -> list[str]:
    """Nombres suivis d'un « % » (bornes de type « 0 - 35 % » incluses), sous forme canonique."""
    found: list[str] = []
    for match in _PCT_GROUP.finditer(text or ""):
        for num in re.findall(_NUM, match.group(1)):
            found.append(_canon(num))
    return list(dict.fromkeys(found))


def find_unsupported_figures(answer: str, sources: str) -> list[str]:
    """Pourcentages de la réponse absents du texte des sources (à vérifier manuellement).

    Les bornes triviales (0 et 100) sont ignorées. Un chiffre lu uniquement sur une image
    sera signalé : c'est volontaire, l'analyste doit alors le contrôler sur la capture.
    """
    source_numbers = {_canon(n) for n in re.findall(_NUM, sources or "")}
    return [
        f"{fig} %"
        for fig in extract_percent_figures(answer)
        if fig not in _TRIVIAL and fig not in source_numbers
    ]
