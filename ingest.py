import os
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
import hashlib

# Configuration
DATA_DIR = "data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Initialize embedding model and ChromaDB client
print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# We'll use a single collection for simplicity
collection_name = "rag_collection"
collection = chroma_client.get_or_create_collection(name=collection_name)

def get_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def extract_text_and_images(pdf_path):
    doc_hash = get_file_hash(pdf_path)
    doc_images_dir = os.path.join(IMAGES_DIR, doc_hash)
    os.makedirs(doc_images_dir, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    pages_data = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Extract text
        text = page.get_text("text").strip()
        
        # Extract images
        image_list = page.get_images(full=True)
        saved_images = []
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            image_filename = f"page_{page_num + 1}_{img_index}.{image_ext}"
            image_filepath = os.path.join(doc_images_dir, image_filename)
            
            with open(image_filepath, "wb") as f:
                f.write(image_bytes)
            saved_images.append(image_filepath)
            
        pages_data.append({
            "page_num": page_num + 1,
            "text": text,
            "images": saved_images,
            "doc_hash": doc_hash
        })
        
    return pages_data, doc_hash

def chunk_text(text, chunk_size=150, overlap=30):
    """Simple text chunker with overlap (words based)."""
    if not text:
        return []
        
    words = text.split()
    chunks = []
    
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
        
    return chunks

def ingest_pdf(pdf_path):
    print(f"Processing PDF: {pdf_path}")
    pages_data, doc_hash = extract_text_and_images(pdf_path)
    
    documents = []
    metadatas = []
    ids = []
    
    chunk_counter = 0
    for page in pages_data:
        if not page["text"]:
            continue
            
        chunks = chunk_text(page["text"], chunk_size=150, overlap=30)
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({
                "page_num": page["page_num"],
                "doc_hash": doc_hash,
                "source": os.path.basename(pdf_path)
            })
            ids.append(f"{doc_hash}_p{page['page_num']}_c{i}")
            chunk_counter += 1
            
    if documents:
        print(f"Generating embeddings for {len(documents)} chunks...")
        embeddings = embedding_model.encode(documents).tolist()
        
        print("Storing in ChromaDB...")
        collection.upsert(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print("Ingestion complete.")
    else:
        print("No text found to ingest.")
        
    return doc_hash
