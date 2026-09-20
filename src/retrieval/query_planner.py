"""Décomposition d'une question large en sous-questions ciblées.

« contraintes par classe d'actifs (actions, obligations, devises, dérivés) » produit un embedding
dilué : aucun chunk ne ressemble à toute la phrase. En interrogeant séparément chaque classe
d'actifs puis en fusionnant, on récupère les passages propres à chacune.

Modes :
- "off"   : question inchangée ;
- "rules" : énumération entre parenthèses éclatée par règle (gratuit, instantané) ;
- "llm"   : le VLM local propose les sous-questions (un appel de plus, quelques secondes).
"""
from __future__ import annotations

import json
import logging
import re

from src.config import get_settings

logger = logging.getLogger(__name__)

_PARENS = re.compile(r"\(([^()]{3,200})\)")
_SPLIT = re.compile(r"\s*[,;/]\s*|\s+(?:et|ou)\s+", re.IGNORECASE)


def _dedupe(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        q = re.sub(r"\s+", " ", q).strip()
        key = q.lower()
        if q and key not in seen:
            seen.add(key)
            out.append(q)
    return out


def rules_decompose(query: str, max_items: int = 6) -> list[str]:
    """[question, sous-question 1, ...] si la question contient une énumération entre parenthèses."""
    for match in _PARENS.finditer(query):
        items = [i.strip() for i in _SPLIT.split(match.group(1)) if i.strip()]
        if not (2 <= len(items) <= max_items):
            continue
        if any(len(i.split()) > 4 for i in items):  # « (voir page 3 du document) » n'est pas une énumération
            continue
        subs = [f"{query[:match.start()]}{item}{query[match.end():]}" for item in items]
        return _dedupe([query, *subs])
    return [query]


def llm_decompose(query: str, model: str | None = None, max_queries: int = 6) -> list[str]:
    """Demande au VLM local 2 à 6 sous-questions de recherche ; retombe sur les règles en cas d'échec."""
    settings = get_settings()
    try:
        import ollama

        schema = {
            "type": "object",
            "properties": {"queries": {"type": "array", "items": {"type": "string"}, "maxItems": max_queries}},
            "required": ["queries"],
        }
        response = ollama.chat(
            model=model or settings.ollama_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu prépares des requêtes de recherche pour un prospectus financier. "
                        f"Décompose la question en au plus {max_queries} requêtes courtes et autonomes, "
                        "chacune ciblant un seul sujet (une classe d'actifs, un type de frais, une limite...). "
                        "Réponds uniquement en JSON : {\"queries\": [...]}."
                    ),
                },
                {"role": "user", "content": query},
            ],
            format=schema,
            # num_ctx identique à la génération : sinon Ollama recharge le modèle entre les deux appels.
            options={"temperature": 0, "num_predict": 200, "num_ctx": settings.num_ctx},
            keep_alive=settings.keep_alive,
        )
        data = json.loads(response["message"]["content"])
        subs = [q for q in data.get("queries", []) if isinstance(q, str)]
        if subs:
            return _dedupe([query, *subs])[: max_queries + 1]
    except Exception as exc:
        logger.warning("Décomposition LLM impossible, repli sur les règles : %s", exc)
    return rules_decompose(query)


def plan_queries(query: str, mode: str = "rules", model: str | None = None) -> list[str]:
    """Liste de requêtes à exécuter ; la première est toujours la question d'origine."""
    mode = (mode or "off").lower()
    if mode == "llm":
        return llm_decompose(query, model)
    if mode == "rules":
        return rules_decompose(query)
    return [query]
