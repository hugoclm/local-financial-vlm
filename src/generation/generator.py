from __future__ import annotations

import io
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from src.config import Settings, get_settings
from src.utils import find_unsupported_figures

logger = logging.getLogger(__name__)

SYSTEM_PROMPTS = {
    "constraints": (
        "Tu es un analyste quantitatif spécialisé en conformité réglementaire. "
        "Tu extrais avec rigueur les contraintes d'investissement de documents financiers "
        "(prospectus, DIC/KID).\n\n"
        "Règles strictes :\n"
        "1. Utilise UNIQUEMENT les extraits fournis (texte et images). Si une information n'y figure pas, "
        "écris « non précisé » : n'invente rien.\n"
        "2. Restitue la réponse sous la forme d'un tableau Markdown :\n"
        "   | Classe d'actifs / Instrument | Borne minimale | Borne maximale | Conditions / Précisions | Page source |\n"
        "3. Ne mélange pas les niveaux : une borne portant sur une poche, un sous-ensemble ou l'ensemble du "
        "portefeuille doit être rattachée explicitement à son niveau dans la colonne « Conditions / Précisions ».\n"
        "4. Recopie les chiffres exactement comme dans le document (unités et pourcentages compris). "
        "Ne calcule aucune valeur dérivée sauf demande explicite.\n"
        "5. Chaque contrainte n'apparaît qu'UNE SEULE FOIS. Indique la page d'après l'en-tête de l'extrait.\n"
        "6. Termine par une synthèse de 3 lignes maximum, seulement si elle apporte quelque chose."
    ),
    "free": (
        "Tu es un analyste financier rigoureux. Réponds à la question en français, de façon concise et "
        "structurée, uniquement à partir des extraits fournis (texte et images). "
        "Cite la page source entre parenthèses, par exemple (p. 12). Recopie les chiffres exactement. "
        "Si l'information n'est pas dans les extraits, dis-le clairement au lieu de la deviner."
    ),
}


@dataclass
class PreparedPrompt:
    """Prompt prêt à envoyer + informations de diagnostic pour l'interface."""

    messages: list[dict]
    options: dict
    extracts: list[dict]
    image_extract_numbers: list[int] = field(default_factory=list)  # n° d'extrait de chaque image jointe
    dropped_extracts: int = 0
    dropped_images: int = 0
    estimated_tokens: int = 0
    source_text: str = ""


def _estimate_text_tokens(text: str) -> int:
    # ~3 caractères/token : volontairement prudent pour du français chiffré (tableaux).
    return int(len(text) / 3.0) + 1


class MultimodalFinancialGenerator:
    """Générateur exploitant Qwen2.5-VL via Ollama pour analyser texte et images."""

    def __init__(self, model_name: Optional[str] = None, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.ollama_model

    # ------------------------------------------------------------ préparation
    @staticmethod
    def _order_key(extract: dict) -> tuple:
        return (
            extract.get("source_document") or "",
            extract.get("doc_id") or "",
            extract.get("chunk_index") if extract.get("chunk_index") is not None else 0,
            extract.get("page_number") or 0,
        )

    def _load_image(self, path: str) -> Optional[tuple[bytes, int]]:
        """Image redimensionnée (côté max configurable) + estimation de ses tokens visuels."""
        try:
            from PIL import Image

            with Image.open(path) as img:
                img = img.convert("RGB")
                side = self.settings.max_image_side
                img.thumbnail((side, side))
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                width, height = img.size
            # Qwen2.5-VL : ~1 token visuel par bloc de 28x28 pixels.
            return buffer.getvalue(), math.ceil(width / 28) * math.ceil(height / 28)
        except ImportError:
            return Path(path).read_bytes(), 1500
        except Exception as exc:
            logger.warning("Image illisible %s : %s", path, exc)
            return None

    def _select_images(self, extracts: list[dict], policy: str) -> dict[int, tuple[bytes, int]]:
        """Choisit au plus `max_images` images : uniquement pour les tableaux douteux (auto), ou les mieux classés (always)."""
        if policy == "never" or self.settings.max_images <= 0:
            return {}
        candidates = []
        for i, ex in enumerate(extracts):
            path = ex.get("image_path")
            if not path or not Path(path).exists():
                continue
            wanted = ex.get("needs_image") if policy == "auto" else ex.get("chunk_type") == "table"
            if wanted:
                score = ex.get("rerank_score")
                candidates.append((score if score is not None else float("-inf"), i, str(Path(path).resolve())))
        candidates.sort(key=lambda c: c[0], reverse=True)

        chosen: dict[int, tuple[bytes, int]] = {}
        seen_paths: set[str] = set()
        for _, i, path in candidates:
            if len(chosen) >= self.settings.max_images:
                break
            if path in seen_paths:  # un tableau scindé en plusieurs chunks partage une seule image
                continue
            loaded = self._load_image(path)
            if loaded:
                chosen[i] = loaded
                seen_paths.add(path)
        return chosen

    def prepare(
        self,
        query: str,
        retrieved_context: list[dict],
        mode: str = "constraints",
        image_policy: str = "auto",
    ) -> PreparedPrompt:
        """Construit le prompt : contexte dans l'ordre du document, images légendées, budget de tokens respecté."""
        s = self.settings
        system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["constraints"])
        extracts = sorted(retrieved_context, key=self._order_key)
        images = self._select_images(extracts, image_policy)

        def cost(ex: dict) -> int:
            return _estimate_text_tokens((ex.get("text_content") or "") + (ex.get("section") or "")) + 25

        fixed = _estimate_text_tokens(system_prompt) + _estimate_text_tokens(query) + 250
        budget = s.num_ctx - s.num_predict - 200  # marge de sécurité

        def total() -> int:
            return fixed + sum(cost(e) for e in extracts) + sum(t for _, t in images.values())

        dropped_extracts = dropped_images = 0
        while total() > budget and extracts:
            # On sacrifie d'abord les voisins (les moins utiles), puis les hits les moins bien classés.
            neighbors = [i for i, e in enumerate(extracts) if e.get("is_neighbor")]
            if neighbors:
                victim = neighbors[-1]
            else:
                victim = min(range(len(extracts)), key=lambda i: extracts[i].get("rerank_score") or 0.0)
            extracts.pop(victim)
            dropped_extracts += 1
            # Les indices d'images suivent la liste : on recale.
            images = {(i if i < victim else i - 1): v for i, v in images.items() if i != victim}
        while total() > budget and images:
            images.pop(next(iter(images)))
            dropped_images += 1

        # ---- texte du prompt
        blocks: list[str] = []
        for number, ex in enumerate(extracts, start=1):
            page = ex.get("page_number")
            page_end = ex.get("page_end")
            pages = f"{page}-{page_end}" if page_end and page_end != page else f"{page}"
            header = f"[Extrait #{number} | Page {pages}"
            if ex.get("section"):
                header += f" | Section : {ex['section']}"
            blocks.append(f"{header}]\n{ex.get('text_content') or ''}")

        user_content = f"Question : {query}\n\n--- EXTRAITS DU DOCUMENT (dans l'ordre du document) ---\n\n"
        user_content += "\n\n".join(blocks)

        image_numbers: list[int] = []
        image_bytes: list[bytes] = []
        if images:
            legend = []
            for k, idx in enumerate(sorted(images), start=1):
                image_numbers.append(idx + 1)
                image_bytes.append(images[idx][0])
                legend.append(f"image {k} = capture du tableau de l'Extrait #{idx + 1} (page {extracts[idx].get('page_number')})")
            user_content += (
                "\n\n--- CAPTURES JOINTES ---\n"
                + " ; ".join(legend)
                + ".\nEn cas d'écart entre le texte extrait et une capture, fais confiance à la capture pour les chiffres "
                "et la structure du tableau."
            )

        message: dict = {"role": "user", "content": user_content}
        if image_bytes:
            message["images"] = image_bytes

        return PreparedPrompt(
            messages=[{"role": "system", "content": system_prompt}, message],
            options={
                "temperature": s.temperature,
                "num_ctx": s.num_ctx,  # constant : évite le rechargement du modèle par Ollama
                "num_predict": s.num_predict,
                "seed": 42,
            },
            extracts=extracts,
            image_extract_numbers=image_numbers,
            dropped_extracts=dropped_extracts,
            dropped_images=dropped_images,
            estimated_tokens=total(),
            source_text="\n".join(e.get("text_content") or "" for e in extracts),
        )

    # -------------------------------------------------------------- inférence
    def stream(self, prepared: PreparedPrompt) -> Iterator[str]:
        """Génère la réponse morceau par morceau (latence perçue bien plus faible)."""
        import ollama

        for chunk in ollama.chat(
            model=self.model_name,
            messages=prepared.messages,
            options=prepared.options,
            stream=True,
            keep_alive=self.settings.keep_alive,
        ):
            piece = chunk["message"]["content"]
            if piece:
                yield piece

    def generate_response(
        self,
        query: str,
        retrieved_context: list[dict],
        mode: str = "constraints",
        image_policy: str = "auto",
    ) -> str:
        """Interface simple (compatible avec l'ancien code) : retourne la réponse complète."""
        prepared = self.prepare(query, retrieved_context, mode=mode, image_policy=image_policy)
        return "".join(self.stream(prepared))

    def unsupported_figures(self, answer: str, prepared: PreparedPrompt) -> list[str]:
        """Pourcentages cités dans la réponse mais absents du texte des extraits."""
        return find_unsupported_figures(answer, prepared.source_text)

    def warmup(self) -> None:
        """Charge le modèle en mémoire avec la bonne taille de contexte (à lancer en tâche de fond)."""
        try:
            import ollama

            ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": "ok"}],
                options={"num_ctx": self.settings.num_ctx, "num_predict": 1},
                keep_alive=self.settings.keep_alive,
            )
        except Exception as exc:
            logger.info("Préchauffage du modèle ignoré : %s", exc)
