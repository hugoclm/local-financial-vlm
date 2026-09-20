# Local Financial VLM RAG

A high-precision, 100% local Multimodal Retrieval-Augmented Generation (RAG) pipeline designed for regulatory financial documents (PRIIPs KIDs, UCITS prospectuses).

This pipeline addresses tabular data extraction challenges and dense financial constraints by pairing a two-stage hybrid retrieval architecture (dense vector search and cross-encoder reranking) with an on-premise Vision-Language Model (VLM) via Ollama.

---

## Technical Architecture

The ingestion and query workflows operate entirely offline:

1. **Document Routing:** A Document Dispatcher inspects document metadata and structural cues to route between dedicated parsers:
   - **KIDParser:** Targeted section and high-DPI visual cropping for 3-4 page PRIIPs KIDs.
   - **FullProspectusParser:** Agnostic sliding-window text chunking with overlap and table extraction for comprehensive 40+ page prospectuses.
2. **Dense Vector Indexing:** Chunks are vectorized using the BAAI/bge-m3 dense embedding model (1024 dimensions) and stored locally in an embedded Qdrant database.
3. **Two-Stage Retrieval:**
   - **Stage 1 (Bi-Encoder):** Retrieves an initial candidate pool (e.g. retrieve_k=30) via cosine similarity.
   - **Stage 2 (Cross-Encoder):** Re-ranks candidate passages using BAAI/bge-reranker-base, isolating target regulatory constraints.
4. **Multimodal Inference:** Top-ranked contextual text and corresponding high-resolution table images are passed to Qwen2.5-VL-7B via Ollama for structured extraction.

---

## Tech Stack

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Dependency Manager** | uv | High-speed, deterministic Python virtual environment and lockfile resolution. |
| **PDF Extraction** | PyMuPDF (fitz) | High-performance text parsing and targeted visual table extraction (150 DPI). |
| **Embeddings** | BAAI/bge-m3 | Multilingual dense representation optimized for structured and regulatory text. |
| **Vector Database** | Qdrant | Serverless, persistent on-disk embedded vector store. |
| **Reranker** | BAAI/bge-reranker-base | Cross-encoder eliminating semantic noise and lexical false positives. |
| **VLM Runtime** | Ollama | C++ engine managing local GPU/CPU inference and context windows. |
| **Vision-Language Model** | Qwen2.5-VL-7B | Multimodal reasoning directly inspecting tabular crops and numeric constraints. |
| **Web Interface** | Streamlit | Interactive side-by-side inspection of generated responses and source crops. |

---

## Project Structure

```text
local-financial-vlm/
|-- data/
|   |-- samples/                 # Source financial PDFs
|   +-- cache/extracted_images/  # High-res PNG crops generated during parsing
|-- qdrant_storage/              # Embedded Qdrant persistent storage
|-- src/
|   |-- parsing/
|   |   |-- schemas.py           # Pydantic data schemas
|   |   |-- base.py              # Abstract base parser interface
|   |   |-- kid_parser.py        # Dedicated PRIIPs KID parser
|   |   |-- prospectus_parser.py # Sliding-window multi-page prospectus parser
|   |   +-- dispatcher.py        # Document dispatcher and heuristic router
|   |-- indexing/
|   |   +-- indexer.py           # BGE-M3 vectorization and Qdrant ingestion
|   |-- retrieval/
|   |   +-- retriever.py         # Bi-encoder search and cross-encoder reranker
|   +-- generation/
|       +-- generator.py         # Multimodal generator calling Qwen2.5-VL via Ollama
|-- app.py                       # Interactive Streamlit web interface
|-- main.py                      # CLI entrypoint for batch evaluation
|-- pyproject.toml               # Project definition and dependencies
+-- README.md
Prerequisites
Python 3.11+

uv (Astral)

Ollama running locally with the vision-language model installed:

PowerShell
ollama pull qwen2.5vl:7b
Installation & Usage
1. Clone repository and install dependencies
PowerShell
git clone <repo-url>
cd local-financial-vlm
uv sync
2. Launch the Streamlit web application
PowerShell
uv run streamlit run app.py
Open http://localhost:8501 in your browser, upload a financial PDF, and run queries to inspect extracted constraints alongside source visual snippets.

3. Run via command-line interface
PowerShell
uv run python main.py
