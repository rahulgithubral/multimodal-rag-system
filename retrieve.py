import os
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "chroma_db"
IMAGES_DIR = "data/images"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Load model and DB lazily to save memory if not needed immediately
_embedding_model = None
_collection = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model

def get_collection():
    global _collection
    if _collection is None:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = chroma_client.get_or_create_collection(name="rag_collection")
    return _collection

def retrieve(query, top_k=3):
    model = get_embedding_model()
    collection = get_collection()
    
    query_embedding = model.encode([query]).tolist()[0]
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    retrieved_chunks = []
    retrieved_images = []
    
    if results['documents'] and len(results['documents']) > 0:
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        
        for doc, meta in zip(docs, metas):
            doc_hash = meta['doc_hash']
            page_num = meta['page_num']
            
            chunk_info = {
                "text": doc,
                "page_num": page_num,
                "source": meta.get('source', 'Unknown')
            }
            retrieved_chunks.append(chunk_info)
            
            # Find associated images for this page
            doc_images_dir = os.path.join(IMAGES_DIR, doc_hash)
            if os.path.exists(doc_images_dir):
                for img_file in os.listdir(doc_images_dir):
                    if img_file.startswith(f"page_{page_num}_"):
                        img_path = os.path.join(doc_images_dir, img_file)
                        if img_path not in retrieved_images:
                            retrieved_images.append(img_path)
                            
    return retrieved_chunks, retrieved_images
