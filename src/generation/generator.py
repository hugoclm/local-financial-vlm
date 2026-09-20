from pathlib import Path
import ollama


class MultimodalFinancialGenerator:
    """Générateur financier exploitant Qwen2.5-VL via Ollama pour analyser texte et images."""

    def __init__(self, model_name: str = "qwen2.5vl:7b"):
        self.model_name = model_name

    def generate_response(
        self,
        query: str,
        retrieved_context: list[dict],
    ) -> str:
        """Construit le prompt multimodal avec les images associées et interroge le VLM."""
        images_to_send = []
        text_chunks = []

        for idx, doc in enumerate(retrieved_context, start=1):
            source_info = f"[Extrait #{idx} - Page {doc.get('page_number')}]"
            text_chunks.append(f"{source_info}\n{doc.get('text_content', '')}")

            # Si le chunk contient une capture visuelle valide, on l'ajoute
            img_path = doc.get("image_path")
            if img_path and Path(img_path).exists():
                images_to_send.append(str(Path(img_path).resolve()))

        # Assemblage du contexte textuel
        full_text_context = "\n\n".join(text_chunks)

        system_prompt = (
           "Tu es un analyste quantitatif et conformité réglementaire de premier ordre. "
            "Ton objectif est de restituer de façon synthétique et rigoureuse les contraintes d'investissement.\n\n"
            "Consignes strictes de restitution :\n"
            "1. Structure impérativement ta réponse sous la forme d'un tableau Markdown synthétique :\n"
            "   | Classe d'actifs / Instrument | Borne minimale | Borne maximale | Conditions / Précisions | Page source |\n"
            "2. Reste factuel : ne confonds pas l'allocation actions (0-35%) et le compartiment de taux (65-100%).\n"
            "3. Chaque contrainte ne doit apparaître QU'UNE SEULE FOIS dans le tableau. Zéro répétition.\n"
            "4. Ne conclus pas par des listes superflues ou des paragraphes redondants : limite-toi au tableau suivi d'une courte synthèse de 3 lignes maximum si nécessaire."
        )       

        user_content = (
            f"Question : {query}\n\n"
            f"--- CONTEXTE EXTRAIT DU PROSPECTUS ---\n{full_text_context}\n\n"
        )
        if images_to_send:
            user_content += "Analyse attentivement le texte et les captures de tableaux jointes pour extraire les chiffres exacts."
        else:
            user_content += "Extrait l'ensemble des règles et contraintes stipulées dans les extraits textuels ci-dessus."

        # Message structuré pour l'API Ollama
        message_payload = {
            "role": "user",
            "content": user_content,
        }

        if images_to_send:
            message_payload["images"] = images_to_send

        # Appel d'inférence locale
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                message_payload,
            ],
            options={
                "temperature": 0.1,  # Faible température pour éviter les hallucinations
            },
        )

        return response["message"]["content"]