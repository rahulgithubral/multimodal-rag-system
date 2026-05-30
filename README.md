# Multimodal RAG System (Apple Silicon Optimized)

A complete local multimodal RAG system built for Apple Silicon (MacBook Air M4, 16GB RAM) using Moondream 2, ChromaDB, and Streamlit.

## Features
- **Local Execution:** Everything runs locally on your Mac.
- **Apple Silicon Optimized:** Uses MPS (Metal Performance Shaders) and `torch.float16` for accelerated, memory-efficient inference.
- **Multimodal RAG:** Extracts both text and images from PDFs. Images are retrieved alongside relevant text chunks and passed to the Vision-Language Model.
- **Embeddings:** Uses `BAAI/bge-small-en-v1.5` for fast, high-quality text embeddings.
- **Citations:** Model generates answers with page number citations, and the UI displays the retrieved images from those pages.

## Project Structure
```
rag-assignment/
├── app.py                # Streamlit UI application
├── ingest.py             # PDF ingestion, extraction, chunking, and embedding
├── retrieve.py           # Logic to retrieve top-k chunks and associated images
├── generate.py           # Moondream 2 integration and prompt construction
├── requirements.txt      # Python dependencies
├── data/                 # Auto-generated directory for uploaded PDFs and extracted images
└── chroma_db/            # Auto-generated directory for ChromaDB storage
```

## Setup Instructions

1. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## Note on Memory Usage (16GB RAM)
- The system is configured to use Moondream 2, which is significantly smaller (~2B parameters) than the previous Qwen2.5-VL-7B model, making it perfect for 16GB Apple Silicon machines.
- It leverages `torch.float16` and the `mps` device backend for smooth execution.
- The first time you run a query, it will automatically download the Moondream weights from Hugging Face.

## How it Works
1. **Upload:** Upload a PDF via the Streamlit sidebar.
2. **Ingest:** The system extracts text and images page-by-page. Text is chunked (with overlap), embedded, and stored in ChromaDB. Extracted images are saved locally and associated with their page numbers.
3. **Query:** Ask a question. The system embeds the query and retrieves the top-K text chunks.
4. **Retrieve Images:** It also fetches any images that were extracted from the pages corresponding to those retrieved text chunks.
5. **Generate:** The query, text chunks, and images (merged into a collage if there are multiple) are sent to Moondream 2 to generate a grounded response.
