# Multimodal RAG System

A complete local multimodal Retrieval-Augmented Generation (RAG) system built for cross-platform execution (macOS Apple Silicon, Linux/Windows NVIDIA CUDA, and CPU) using Moondream 2, ChromaDB, and Streamlit.

## Features
- **Local Execution:** Everything runs locally on your machine for complete privacy.
- **Cross-Platform Optimized:** Dynamically detects and utilizes Apple MPS (Metal Performance Shaders), NVIDIA CUDA, or gracefully falls back to CPU.
- **Multimodal RAG:** Extracts both text and images from PDFs. Images are retrieved alongside relevant text chunks and passed to the Vision-Language Model.
- **Embeddings:** Uses `BAAI/bge-small-en-v1.5` for fast, high-quality text embeddings.
- **Citations:** Generates answers with accurate page number citations, and the UI displays the retrieved images from those pages.

## Project Structure
```
rag-assignment/
├── app.py                # Streamlit UI application
├── ingest.py             # PDF ingestion, extraction, chunking, and embedding
├── retrieve.py           # Logic to retrieve top-k chunks and associated images
├── generate.py           # Moondream 2 integration and prompt construction
├── utils.py              # Cross-platform device selection and image utilities
├── requirements.txt      # Strictly pinned Python dependencies
├── Dockerfile            # Container definition
├── docker-compose.yml    # Docker orchestration
├── data/                 # Auto-generated directory for uploaded PDFs and extracted images
└── chroma_db/            # Auto-generated directory for ChromaDB storage
```

## Requirements & Compatibility

**Supported Python Versions:**
- `Python 3.10`
- `Python 3.11`

**Unsupported Python Versions:**
- `Python 3.12+` (Including Python 3.14)
- `Python 3.9 and below`

**Why this matters:**
This system heavily relies on `PyMuPDF` (for robust PDF extraction) and `PyTorch` (for local machine learning inference). These core libraries do not immediately release pre-compiled binary "wheels" for bleeding-edge Python versions (like 3.14). If you attempt to install the dependencies on an unsupported version, `pip` will attempt to build the C/C++ source code from scratch, which will almost certainly fail due to compiler incompatibilities. **To ensure a perfect, reviewer-proof installation, you must use Python 3.10 or 3.11.**

## Setup Instructions

### Option A: Local Setup (Recommended for Development)

1. **Create a virtual environment (Must use Python 3.10 or 3.11):**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

2. **Install pinned dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

### Option B: Docker Setup (Production/Reproducible)

Ensure Docker and Docker Compose are installed.

1. **Build and start the container:**
   ```bash
   docker-compose up --build
   ```
2. **Access the UI:**
   Open your browser and navigate to `http://localhost:8501`.

## Troubleshooting & Crash Prevention

- **Dependency Issues:** We strictly pin `transformers==4.44.2` to avoid weight-tying bugs present in newer HuggingFace releases that break Moondream.
- **Corrupt PDFs / Black Images:** The system includes defensive PyMuPDF parsing and background flattening to automatically reject corrupted or alpha-channel images that would normally crash the UI.
- **Slow Startup:** ChromaDB and the Embedding models are lazy-loaded. The first query or document upload will trigger the download of model weights.
- **Hardware Acceleration:** Verify `torch.cuda.is_available()` or `torch.backends.mps.is_available()` returns `True` if you expect GPU speeds. Otherwise, the system safely falls back to the CPU.
