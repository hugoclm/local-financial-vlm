from pathlib import Path
import streamlit as st

from src.parsing.dispatcher import DocumentDispatcher
from src.indexing.indexer import LocalVectorIndexer
from src.retrieval.retriever import LocalRetriever
from src.generation.generator import MultimodalFinancialGenerator

st.set_page_config(
    page_title="Local Financial VLM RAG",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Local Financial VLM RAG")
st.caption("Analyse multimodale 100 % locale de prospectus et DIC (PRIIPs) via BGE-M3, Qdrant et Qwen2.5-VL")

# Sidebar pour la gestion du document
with st.sidebar:
    st.header("⚙️ Configuration & Document")
    uploaded_file = st.file_uploader("Déposer un document financier (PDF)", type=["pdf"])
    
    collection_name = st.text_input("Nom de la collection Qdrant", value="financial_docs_ui")
    top_k = st.slider("Nombre d'extraits finaux (Top-K)", min_value=1, max_value=8, value=4)
    retrieve_k = st.slider("Candidats bi-encoder (Retrieve-K)", min_value=10, max_value=50, value=30)

    if uploaded_file is not None:
        save_dir = Path("data/samples")
        save_dir.mkdir(parents=True, exist_ok=True)
        saved_pdf_path = save_dir / uploaded_file.name

        if st.button("🚀 Parser et Indexer le document", use_container_width=True):
            with open(saved_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner("Analyse du document via DocumentDispatcher..."):
                dispatcher = DocumentDispatcher()
                chunks = dispatcher.process(str(saved_pdf_path))
                st.session_state["chunks_count"] = len(chunks)
                st.session_state["tables_count"] = sum(1 for c in chunks if c.chunk_type.value == "table")

            with st.spinner("Indexation vectorielle dans Qdrant (BGE-M3)..."):
                indexer = LocalVectorIndexer(collection_name=collection_name)
                indexer.index_chunks(chunks)
                indexer.close()
                st.session_state["indexed"] = True
                st.session_state["pdf_name"] = uploaded_file.name

            st.success(f"Indexation réussie : {st.session_state['chunks_count']} chunks ({st.session_state['tables_count']} tableaux)")

# Zone principale : Requête & Résultats
if st.session_state.get("indexed", False):
    st.info(f"📄 Document actif : **{st.session_state.get('pdf_name')}**")
    
    query = st.text_area(
        "Pose ta question sur les contraintes d'investissement, risques ou frais :",
        value="Quelles sont les limites d'exposition et contraintes d'investissement par classe d'actifs (actions, obligations, devises, dérivés) ?",
        height=100,
    )

    if st.button("🔍 Interroger le document", type="primary"):
        with st.spinner("Recherche hybride & Reranking en cours..."):
            retriever = LocalRetriever(collection_name=collection_name)
            results = retriever.search(query=query, top_k=top_k, retrieve_k=retrieve_k)
            retriever.close()

        with st.spinner("Synthèse multimodale via Qwen2.5-VL..."):
            generator = MultimodalFinancialGenerator()
            answer = generator.generate_response(query=query, retrieved_context=results)

        # Affichage côte-à-côte
        col_answer, col_sources = st.columns([3, 2])

        with col_answer:
            st.subheader("💡 Analyse & Réponse")
            st.markdown(answer)

        with col_sources:
            st.subheader("📌 Preuves & Extraits sources")
            for idx, r in enumerate(results, 1):
                with st.expander(f"Extrait #{idx} — Page {r['page_number']} (Score: {r['rerank_score']:.4f})", expanded=(idx == 1)):
                    if r.get("image_path") and Path(r["image_path"]).exists():
                        st.image(r["image_path"], caption=f"Capture source (Page {r['page_number']})", use_container_width=True)
                    st.text_area("Texte extrait", value=r["text_content"], height=160, key=f"src_text_{idx}", disabled=True)
else:
    st.warning("👈 Veuillez d'abord charger et indexer un document PDF depuis le panneau latéral.")