# Changements — refonte performance & qualité

## Migration (à lire avant de lancer)

1. **Ré-indexe tes PDF.** Le format des points Qdrant a changé (`doc_id`, `chunk_index`, `section`, `retrieval_text`, ids uuid5).
   La collection par défaut est maintenant `financial_docs_v2` : l'ancienne (`financial_docs_ui`, `prospectus_sg_v3`) n'est pas modifiée mais n'est plus compatible avec le filtrage par document.
2. **Chemins supposés** : `src/indexing/indexer.py`, `src/retrieval/retriever.py`, `src/generation/generator.py`.
   Si ton arborescence diffère, dépose chaque fichier à côté de l'ancien et ajuste les imports en tête de `app.py`, `main.py` et `evaluate.py`.
3. **Dépendances** : aucune nouvelle. Pillow est déjà tiré par Streamlit (sinon `uv add pillow`). PyMuPDF ≥ 1.23 (pour `find_tables`).
4. `main.py` est désormais un CLI (`--pdf`, `--query`, `--mode`, `--retrieval-only`…). Voir `--help`.

## Ce qui change, fichier par fichier

| Fichier | Changements |
|---|---|
| `src/config.py` (nouveau) | Tous les réglages, surchargeables par variables d'environnement. |
| `src/models.py` (nouveau) | Encodeur, reranker et client Qdrant **créés une seule fois** et partagés (avant : rechargés à chaque clic, en double). |
| `src/parsing/prospectus_parser.py` | Tableaux retirés du flux de texte (fin des doublons) ; tableaux en **Markdown** ; texte en flux continu (chunks à cheval sur 2 pages) ; **titres → section** préfixée à chaque chunk ; en-têtes/pieds de page supprimés ; légende rattachée à son tableau ; dé-césure. |
| `src/parsing/chunking.py` (nouveau) | Logique de découpage pure et testée (un paragraphe > `chunk_size` est désormais réellement redécoupé). |
| `src/parsing/kid_parser.py` | Tri lecture, tableaux des sections en Markdown, texte hors sections regroupé (au lieu d'un chunk par bloc), noms d'images uniques par document. |
| `src/parsing/schemas.py` / `base.py` / `dispatcher.py` | `doc_id`, `section`, `chunk_index`, `page_end`, `needs_image` ; `doc_id` = hash du fichier. |
| `src/indexing/indexer.py` | Ids déterministes (plus d'écrasement entre documents), `is_indexed` / `delete_document` / `list_documents`, vecteurs normalisés, batches triés par longueur, progression. |
| `src/retrieval/retriever.py` | Filtre par document ; **recherche hybride dense + BM25 (fusion RRF)** ; **sous-requêtes** avec place garantie par sous-requête ; dédoublonnage ; `build_context` ajoute les **chunks voisins** dans l'ordre du document. |
| `src/retrieval/lexical.py`, `query_planner.py` (nouveaux) | BM25 sans dépendance ; décomposition par règles (énumérations entre parenthèses) ou par le VLM. |
| `src/generation/generator.py` | Prompt **générique** (la mention « actions 0-35 % / taux 65-100 % » qui biaisait les autres documents a disparu) ; contexte dans l'ordre du document ; images **légendées** (« image 1 = tableau de l'Extrait #2 »), dédupliquées, **redimensionnées** et limitées ; `num_ctx` fixe + budget de tokens ; `keep_alive` ; **streaming** ; vérification des pourcentages cités. |
| `app.py` | Modèles chargés une fois (`st.cache_resource`), préchauffage d'Ollama, indexation **persistante** par document (liste + hash, pas de ré-indexation inutile), streaming, chronométrage par étape, options (décomposition, voisins, images, format de réponse). |
| `evaluate.py` + `eval/` (nouveaux) | page_hit@k, MRR, rappel de termes, temps ; `--generate` pour la réponse finale. |
| `tests/` (nouveaux) | 31 tests sans dépendance lourde (faux Qdrant / encodeur / reranker). |

## Réglages utiles (variables d'environnement)

| Variable | Défaut | Quand la changer |
|---|---|---|
| `EMBEDDING_DEVICE` / `RERANKER_DEVICE` | `cpu` | `cuda` pour accélérer l'indexation si ta VRAM le permet (BGE-M3 ≈ 1,1 Go en fp16 ; le CPU était sans doute un choix pour laisser la VRAM à Ollama). |
| `OLLAMA_NUM_CTX` | `8192` | Garde-la **constante** : la changer entre deux appels fait recharger le modèle. |
| `MAX_IMAGES` / `MAX_IMAGE_SIDE` | `2` / `1280` | Baisse-les si le préremplissage est lent. `0` = texte seul. |
| `DECOMPOSE_MODE` | `rules` | `llm` pour des questions très ouvertes (un appel VLM de plus). |
| `QDRANT_URL` | vide | `http://localhost:6333` pour un serveur Docker : indispensable si plusieurs sessions Streamlit écrivent en même temps (le mode embarqué verrouille le dossier). |

## Ce qui a été vérifié, et ce qui ne l'a pas été

- **Exécuté et validé** : chunking, BM25, planification de requêtes, vérification des chiffres, logique du retriever (filtre, RRF, sous-requêtes, doublons, voisins, cache), préparation du prompt (ordre, images, budget). Des tests « de mutation » (casser volontairement le code) confirment que les tests détectent les régressions.
- **Simulé** (faux PyMuPDF/Pydantic) : `FullProspectusParser`, `KIDParser`, `DocumentDispatcher` sur des documents synthétiques.
- **Non exécuté ici** (pas de réseau ni de GPU) : les appels réels à PyMuPDF (`find_tables`, drapeaux de police), Qdrant, sentence-transformers et Ollama.
  Premiers réflexes : `uv run python tests/test_core.py`, puis `uv run python main.py --pdf <ton.pdf> --retrieval-only -v`
  pour contrôler chunks, sections et tableaux avant de lancer le VLM.
- **Seuils à calibrer sur tes PDF** : détection des titres (`_looks_like_heading`), bandes d'en-tête/pied (9 %), qualité des tableaux (`table_quality`). Le jeu d'évaluation (`evaluate.py`) sert à cela.
