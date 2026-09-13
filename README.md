# Local Financial VLM RAG

A high-precision, 100% local Multimodal Retrieval-Augmented Generation (RAG) pipeline designed for regulatory financial documents (PRIIPs KIDs, UCITS prospectuses).

This system resolves standard tabular OCR data loss and alignment issues on dense financial grids by pairing hybrid vector retrieval with a Vision-Language Model (VLM) inspecting high-resolution image crops of source tables.

---

## Architecture Overview

\\	ext
                +-----------------------------------------------+
                |             Financial PDF Document            |
                +-----------------------+-----------------------+
                                        |
                         [ FinancialPDFParser (PyMuPDF) ]
                                        |
                      +-----------------+-----------------+
                      |                                   |
              [ Text Chunks ]                     [ Visual Croppings ]
             (Narrative text)                   (Tables / Grids as PNG)
                      |                                   |
                      +-----------------+-----------------+
                                        |
                         [ BGE-M3 Dense Embeddings ]
                                        |
                                        v
                          [ Qdrant Local Vector DB ]
                                        |
----------------------------------------+----------------------------------------
                                        |
[ User Financial Query ] ---> [ Top-10 Vector Search ]
                                        |
                                        v
                          [ BGE-Reranker-base (Cross-Encoder) ]
                                        |
                                        v (Filtered Top-K)
                          [ Context (Text + Image Paths) ]
                                        |
                                        v
                        [ Qwen2.5-VL-7B (via Ollama API) ]
                                        |
                                        v
                             [ Audited Structured Answer ]
                                (Exact values + Citations)
\
---

## Tech Stack

| Component | Technology | Role / Rationale |
| :--- | :--- | :--- |
| **Dependency Manager** | uv | High-speed, deterministic Python virtual environment and lockfile management. |
| **PDF Parsing** | PyMuPDF (fitz) | Fast vector text extraction and targeted high-DPI (150 DPI) image cropping. |
| **Embeddings Model** | BAAI/bge-m3 | Dense multilingual embeddings (dim 1024), resilient on structured financial contexts. |
| **Vector Store** | Qdrant | Embedded local on-disk vector store (qdrant_storage), persistent and serverless. |
| **Reranker** | BAAI/bge-reranker-base | Cross-encoder evaluating query-document cross-attention to remove lexical false positives. |
| **VLM Inference Engine** | Ollama | Standalone C++ engine handling GPU VRAM offloading for qwen2.5vl:7b. |
| **Vision-Language Model** | Qwen2.5-VL-7B | Multimodal reasoning directly extracting table cells and rates without OCR drift. |

---

## Key Features

1. **Hybrid Text + Image Extraction:**
   Complex regulatory tables (e.g., performance scenarios, cost breakdowns) are extracted as clean PNG images while retaining underlying text tokens for semantic retrieval.
2. **Two-Stage Semantic Retrieval:**
   - **Stage 1 (Bi-Encoder):** Rapid Top-10 candidate extraction via cosine similarity in Qdrant.
   - **Stage 2 (Cross-Encoder):** Deep reranking using bge-reranker-base elevating the target financial section to rank #1 (rerank score > 0.95).
3. **Multimodal Grounded Generation:**
   The VLM receives both the cropped source table and surrounding textual metadata, preventing column misalignments and hallucinated rates.
4. **100% On-Premise & Air-Gapped:**
   No proprietary financial documents or user queries leave the local environment.

---

## Project Structure

\\	ext
local-financial-vlm/
|-- data/
|   |-- samples/                 # Source input PDFs (e.g. sample_prospectus.pdf)
|   +-- cache/extracted_images/  # High-res PNG crops generated at parsing (gitignored)
|-- qdrant_storage/              # Embedded Qdrant persistent database files
|-- src/
|   |-- parsing/
|   |   |-- schemas.py           # Pydantic schemas for chunks and metadata
|   |   +-- pdf_parser.py        # Hybrid PDF parser & visual crop extractor
|   |-- indexing/
|   |   +-- indexer.py           # BGE-M3 vectorization and Qdrant ingestion
|   |-- retrieval/
|   |   +-- retriever.py         # Vector retrieval + Cross-Encoder reranker
|   +-- generation/
|       +-- generator.py         # Multimodal generator calling Qwen2.5-VL via Ollama
|-- main.py                      # End-to-end pipeline entrypoint
|-- pyproject.toml               # Project specifications and uv dependencies
+-- README.md
\
---

## Prerequisites

- **Python** 3.11+
- **Package manager**: uv
- **Ollama** installed with the vision model pulled:
  ollama pull qwen2.5vl:7b

---

## Getting Started

### 1. Clone repository and install dependencies
\\powershell
git clone <repo-url>
cd local-financial-vlm
uv sync
\
### 2. Run the end-to-end pipeline
Place your target financial PDF at data/samples/sample_prospectus.pdf, then run:

\\powershell
python main.py
\
---

## Roadmap

- [x] Hybrid text / targeted image extraction with PyMuPDF
- [x] Local vector indexing using Qdrant and BGE-M3
- [x] Semantic cross-encoder reranking (bge-reranker-base)
- [x] Local multimodal generation with Qwen2.5-VL-7B via Ollama
- [ ] Document Dispatcher (adaptive parsing strategy for KIDs vs. 50+ page prospectuses)
- [ ] Automated quantitative evaluation (Hit-Rate@k, Zero-hallucination rate on financial metrics)
- [ ] Interactive local UI (Streamlit side-by-side inspection view)
