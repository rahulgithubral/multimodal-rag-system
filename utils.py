import os
import chromadb
from PIL import Image

CHROMA_PATH = "chroma_db"
IMAGES_DIR = "data/images"

def get_db_stats():
    """Retrieves statistics about the current indexed documents."""
    stats = {
        "total_documents": 0,
        "unique_sources": [],
        "total_chunks": 0,
        "total_images": 0
    }
    
    # Connect to ChromaDB
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(name="rag_collection")
        data = collection.get()
        
        if data and 'ids' in data:
            stats["total_chunks"] = len(data['ids'])
            
            # Extract unique sources
            sources = set()
            for meta in data['metadatas']:
                if meta and 'source' in meta:
                    sources.add(meta['source'])
            
            stats["unique_sources"] = sorted(list(sources))
            stats["total_documents"] = len(stats["unique_sources"])
            
    except Exception as e:
        # Collection might not exist yet
        pass
        
    # Count images
    try:
        if os.path.exists(IMAGES_DIR):
            img_count = 0
            for root, dirs, files in os.walk(IMAGES_DIR):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        img_count += 1
            stats["total_images"] = img_count
    except Exception:
        pass
        
    return stats

def get_white_bg_image(img_path):
    """Loads an image and flattens any transparency onto a solid white background. Returns None if invalid."""
    try:
        img = Image.open(img_path)
        
        # Check if corrupted or zero size
        if img.size[0] == 0 or img.size[1] == 0:
            return None
            
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            alpha = img.convert('RGBA').split()[-1]
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            bg.paste(img, mask=alpha)
            img = bg
            
        # Convert to RGB unconditionally for Moondream
        img = img.convert("RGB")
        
        # Optional: Verify image is not entirely black (min and max pixel values are 0)
        extrema = img.getextrema()
        if all(ex[1] == 0 for ex in extrema):
            return None
            
        return img
    except Exception:
        return None
