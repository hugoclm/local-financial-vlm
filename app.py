import threading
import time
from pathlib import Path

import streamlit as st

from src.config import get_settings
from src.generation.generator import MultimodalFinancialGenerator
from src.indexing.indexer import LocalVectorIndexer
from src.parsing.dispatcher import DocumentDispatcher
from src.retrieval.retriever import LocalRetriever
from src.utils import Timings, bytes_sha256

settings = get_settings()

st.set_page_config(page_title="Local Financial VLM RAG", page_icon="📊", layout="wide")
st.title("📊 Local Financial VLM RAG")
st.caption("Analyse multimodale 100 % locale de prospectus et DIC (PRIIPs) via BGE-M3, Qdrant et Qwen2.5-VL")


@st.cache_resource(show_spinner="Chargement des modèles (une seule fois)...")
def load_components(collection_name: str):
    """Instancie une seule fois modèles, client Qdrant et générateur (avant : à chaque clic)."""
    indexer = LocalVectorIndexer(collection_name=collection_name)
    retriever = LocalRetriever(collection_name=collection_name)
    generator = MultimodalFinancialGenerator()
    dispatcher = DocumentDispatcher()
    # Charge le VLM dans Ollama en tâche de fond : la première question n'attend plus le chargement.
    threading.Thread(target=generator.warmup, daemon=True).start()
    return indexer, retriever, generator, dispatcher


def fmt_timings(durations: dict) -> str:
    return " · ".join(f"{k} {v:.1f}s" for k, v in durations.items())


def render_answer_meta(result: dict) -> None:
    prepared_info = result["prepared_info"]
    st.caption("⏱ " + fmt_timings(result["timings"]))
    st.caption(
        f"Contexte : {prepared_info['extracts']} extraits (~{prepared_info['tokens']} tokens), "
        f"{prepared_info['images']} image(s) jointe(s)"
        + (f", {prepared_info['dropped']} extrait(s) écarté(s) faute de place" if prepared_info["dropped"] else "")
    )
    if len(result["queries"]) > 1:
        with st.expander(f"🔎 {len(result['queries']) - 1} sous-requêtes utilisées"):
            for q in result["queries"][1:]:
                st.write(f"- {q}")
    if result["unsupported"]:
        st.warning(
            "Chiffres de la réponse non retrouvés dans le texte extrait : "
            + ", ".join(result["unsupported"])
            + ". Ils peuvent provenir d'une capture de tableau : à vérifier dans les sources."
        )


def render_sources(result: dict) -> None:
    st.subheader("📌 Preuves & Extraits sources")
    for idx, r in enumerate(result["hits"], 1):
        pages = f"{r['page_number']}-{r['page_end']}" if r.get("page_end") else f"{r['page_number']}"
        title = f"Extrait #{idx} — Page {pages} (Score: {r['rerank_score']:.4f})"
        with st.expander(title, expanded=(idx == 1)):
            if r.get("section"):
                st.caption(f"Section : {r['section']}")
            if r.get("image_path") and Path(r["image_path"]).exists():
                st.image(r["image_path"], caption=f"Capture source (Page {r['page_number']})", use_container_width=True)
            st.text_area("Texte extrait", value=r["text_content"] or "", height=160, key=f"src_text_{idx}", disabled=True)


# --------------------------------------------------------------------------- barre latérale
with st.sidebar:
    st.header("⚙️ Configuration & Document")
    collection_name = st.text_input("Nom de la collection Qdrant", value="financial_docs_v2")
    indexer, retriever, generator, dispatcher = load_components(collection_name)

    uploaded_file = st.file_uploader("Déposer un document financier (PDF)", type=["pdf"])
    force_reindex = st.checkbox("Forcer la ré-indexation", value=False)

    if uploaded_file is not None and st.button("🚀 Parser et Indexer le document", use_container_width=True):
        data = uploaded_file.getvalue()
        doc_id = bytes_sha256(data)
        if indexer.is_indexed(doc_id) and not force_reindex:
            st.info("Document déjà indexé : parsing et vectorisation ignorés.")
        else:
            save_dir = Path(settings.samples_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = save_dir / Path(uploaded_file.name).name
            pdf_path.write_bytes(data)

            index_timings = Timings()
            with st.status("Indexation en cours...", expanded=True) as status:
                st.write("Analyse du document (DocumentDispatcher)...")
                with index_timings.measure("parsing"):
                    chunks = dispatcher.process(str(pdf_path), doc_id=doc_id)
                n_tables = sum(1 for c in chunks if c.chunk_type.value == "table")

                st.write(f"Vectorisation de {len(chunks)} chunks (BGE-M3)...")
                bar = st.progress(0.0)
                with index_timings.measure("vectorisation"):
                    indexer.index_chunks(chunks, replace=True, progress=bar.progress)
                status.update(label="Indexation terminée", state="complete", expanded=False)
            st.success(f"{len(chunks)} chunks ({n_tables} tableaux) — ⏱ {index_timings.summary()}")
        st.session_state["doc_select"] = doc_id

    documents = indexer.list_documents()
    doc_options = [d["doc_id"] for d in documents]
    labels = {d["doc_id"]: f"{d['source_document']} ({d['chunks']} chunks)" for d in documents}
    active_doc_id = None
    if st.session_state.get("doc_select") not in doc_options:
        st.session_state.pop("doc_select", None)  # collection changée : évite une sélection périmée
    if doc_options:
        active_doc_id = st.selectbox("Document actif", doc_options, format_func=lambda d: labels[d], key="doc_select")
    else:
        st.caption("Aucun document indexé dans cette collection.")

    st.divider()
    top_k = st.slider("Nombre d'extraits finaux (Top-K)", min_value=1, max_value=8, value=4)
    retrieve_k = st.slider("Candidats bi-encoder (Retrieve-K)", min_value=10, max_value=50, value=30)
    decompose = st.selectbox(
        "Décomposition de la question",
        ["rules", "off", "llm"],
        format_func=lambda m: {"rules": "Automatique (énumérations)", "off": "Aucune", "llm": "Par le VLM (plus lent)"}[m],
    )
    neighbor_window = st.slider("Chunks voisins ajoutés (contexte)", 0, 2, settings.neighbor_window)
    mode = st.radio(
        "Format de réponse",
        ["constraints", "free"],
        format_func=lambda m: "Tableau de contraintes" if m == "constraints" else "Réponse libre",
    )
    image_policy = st.selectbox(
        "Images envoyées au VLM",
        ["auto", "always", "never"],
        format_func=lambda m: {
            "auto": "Auto (tableaux douteux seulement)",
            "always": "Meilleurs tableaux (plus lent)",
            "never": "Jamais (texte seul, le plus rapide)",
        }[m],
    )

# --------------------------------------------------------------------------- zone principale
if not active_doc_id:
    st.warning("👈 Veuillez d'abord charger et indexer un document PDF depuis le panneau latéral.")
    st.stop()

st.info(f"📄 Document actif : **{labels[active_doc_id]}**")

query = st.text_area(
    "Pose ta question sur les contraintes d'investissement, risques ou frais :",
    value="Quelles sont les limites d'exposition et contraintes d'investissement par classe d'actifs (actions, obligations, devises, dérivés) ?",
    height=100,
)

if st.button("🔍 Interroger le document", type="primary"):
    timings = Timings()
    try:
        with st.spinner("Recherche hybride & reranking en cours..."):
            hits = retriever.search(query=query, top_k=top_k, retrieve_k=retrieve_k, doc_ids=[active_doc_id], decompose=decompose)
            queries = list(retriever.last_queries)
            timings.durations.update(retriever.last_timings.durations)
            with timings.measure("contexte"):
                context = retriever.build_context(hits, window=neighbor_window)
            prepared = generator.prepare(query, context, mode=mode, image_policy=image_policy)

        if not hits:
            st.warning("Aucun passage trouvé pour cette question.")
            st.stop()

        def timed_stream():
            start = time.perf_counter()
            first = True
            for piece in generator.stream(prepared):
                if first:
                    timings.durations["1er token"] = time.perf_counter() - start
                    first = False
                yield piece
            timings.durations["génération"] = time.perf_counter() - start

        col_answer, col_sources = st.columns([3, 2])
        with col_answer:
            st.subheader("💡 Analyse & Réponse")
            answer = st.write_stream(timed_stream())

        result = {
            "doc_id": active_doc_id,
            "answer": answer,
            "hits": hits,
            "queries": queries,
            "timings": dict(timings.durations),
            "unsupported": generator.unsupported_figures(answer, prepared),
            "prepared_info": {
                "extracts": len(prepared.extracts),
                "images": len(prepared.image_extract_numbers),
                "tokens": prepared.estimated_tokens,
                "dropped": prepared.dropped_extracts,
            },
        }
        st.session_state["last_result"] = result
        with col_answer:
            render_answer_meta(result)
        with col_sources:
            render_sources(result)
    except Exception as exc:  # Ollama arrêté, modèle absent, etc.
        st.error(f"Erreur pendant l'analyse : {exc}")
else:
    previous = st.session_state.get("last_result")
    if previous and previous["doc_id"] == active_doc_id:
        col_answer, col_sources = st.columns([3, 2])
        with col_answer:
            st.subheader("💡 Analyse & Réponse")
            st.markdown(previous["answer"])
            render_answer_meta(previous)
        with col_sources:
            render_sources(previous)
