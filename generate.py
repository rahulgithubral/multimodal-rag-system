import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from pathlib import Path
from utils import get_white_bg_image

# --- Hotfix for HfMoondream all_tied_weights_keys bug ---
_orig_getattr = torch.nn.Module.__getattr__

def _patched_getattr(self, name):
    if name == "all_tied_weights_keys":
        return {}
    return _orig_getattr(self, name)

torch.nn.Module.__getattr__ = _patched_getattr
# --------------------------------------------------------

MODEL_NAME = "vikhyatk/moondream2"

_model = None
_tokenizer = None

def load_model():
    global _model, _tokenizer
    if _model is None:
        print("Loading Moondream2 model...")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Load Moondream 2
        # Use float16 or bfloat16 for efficiency on Apple Silicon
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision="2024-08-26")
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True,
            revision="2024-08-26",
            torch_dtype=torch.float16, 
        ).to(device)
        
    return _model, _tokenizer

def generate_answer(query, retrieved_chunks, retrieved_images):
    model, tokenizer = load_model()
    
    # Analyze query type
    query_lower = query.lower()
    is_summary = any(kw in query_lower for kw in ["what is", "summarize", "explain paper", "overview", "describe", "contribution", "key idea"])
    is_figure = any(kw in query_lower for kw in ["figure", "image", "diagram", "chart", "table"])
    
    # Unique sources for multi-doc synthesis
    unique_doc_names = set(chunk['source'] for chunk in retrieved_chunks) if retrieved_chunks else set()
    is_multi_doc = len(unique_doc_names) > 1
    
    # Construct context text
    context_text = ""
    for i, chunk in enumerate(retrieved_chunks):
        context_text += f"[Document {i+1}, Page {chunk['page_num']}, Source: {chunk['source']}]\n{chunk['text']}\n\n"
        
    # Dynamic Prompt Construction
    prompt = f"You are a senior AI assistant providing clear, structured, and accurate answers.\n"
    prompt += f"Please answer the user's question based strictly on the provided Context documents and images.\n"
    prompt += f"Do not hallucinate or make up information. If the answer is not in the context, explicitly state \"I don't know based on the provided context.\"\n\n"
    
    if is_summary:
        prompt += f"Your response MUST follow this exact structure:\n"
        prompt += f"1. Overview\n2. Main Contribution\n3. Key Technical Ideas\n4. Results/Impact\n\n"
        prompt += f"Your response MUST be a minimum of 5 to 8 sentences long. Do not provide one-line or single-word answers.\n"
    else:
        prompt += f"Your response MUST include a brief summary followed by a detailed explanation.\n"
        prompt += f"Ensure your answer is a minimum of 3 sentences long and easy to read.\n"
        
    if is_figure:
        prompt += f"You are examining a figure, chart, or table. Identify objects, read labels, and identify table columns if present. Explain relationships clearly. Avoid hallucinating unseen visual content.\n"
        
    if is_multi_doc:
        prompt += f"CRITICAL: Use information from all retrieved documents to compare and synthesize a comprehensive answer.\n"
        
    prompt += f"\nContext:\n{context_text}\nUser Question: {query}\n"

    # Handle images dynamically
    max_images = 1 if is_figure else 2
    
    valid_images = []
    if len(retrieved_images) > 0:
        for img_path in retrieved_images:
            img = get_white_bg_image(Path(img_path).resolve())
            if img is not None:
                valid_images.append(img)
            if len(valid_images) == max_images:
                break
                
    if len(valid_images) > 0:
        if len(valid_images) > 1:
            widths, heights = zip(*(i.size for i in valid_images))
            total_height = sum(heights)
            max_width = max(widths)
            
            collage = Image.new('RGB', (max_width, total_height))
            y_offset = 0
            for im in valid_images:
                collage.paste(im, (0, y_offset))
                y_offset += im.size[1]
            final_image = collage
        else:
            final_image = valid_images[0]
            
        # Generate answer with the combined image
        image_embeds = model.encode_image(final_image)
        output_text = model.answer_question(image_embeds, prompt, tokenizer)
    else:
        # If no valid images are retrieved, pass a blank 1x1 dummy image.
        dummy_image = Image.new('RGB', (1, 1), color='white')
        image_embeds = model.encode_image(dummy_image)
        output_text = model.answer_question(image_embeds, prompt, tokenizer)
        
    # Programmatic citations deduplicated by source and page
    output_text += "\n\nSources:\n"
    if retrieved_chunks:
        source_pages = {}
        for chunk in retrieved_chunks:
            src = chunk['source']
            page = chunk['page_num']
            if src not in source_pages:
                source_pages[src] = set()
            source_pages[src].add(page)
            
        for source, pages in sorted(source_pages.items()):
            pages_list = sorted(list(pages))
            pages_str = ",".join(str(p) for p in pages_list)
            output_text += f"• {source} (Pages {pages_str})\n"
    else:
        output_text += "• No sources retrieved.\n"
        
    return output_text
