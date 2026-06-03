import streamlit as st
import sys

# Ensure supported Python version
if sys.version_info[:2] not in [(3, 10), (3, 11)]:
    st.set_page_config(page_title="Error", layout="centered")
    st.error(f"Unsupported Python Version: {sys.version_info[0]}.{sys.version_info[1]}")
    st.markdown("This application specifically requires **Python 3.10** or **Python 3.11** due to strictly compiled dependency wheels for PyMuPDF and PyTorch. Please recreate your virtual environment using a supported version.")
    st.stop()

import os
# Must be set BEFORE any torch-related imports occur
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from ingest import ingest_pdf
from retrieve import retrieve
from generate import generate_answer
from utils import get_db_stats, get_white_bg_image

st.set_page_config(page_title="Multimodal RAG with Moondream 2", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for Dark Theme & Layout ---
st.markdown("""
    <style>
    /* Ensure a dark theme base */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Better card spacing for chat messages */
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        background-color: #1A1C23;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Footer styling */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #12141a;
        padding: 10px 0;
        text-align: center;
        border-top: 1px solid #2e303d;
        z-index: 1000;
        font-size: 0.85rem;
        color: #a0a0a5;
    }
    
    .footer span {
        margin: 0 15px;
        font-weight: 500;
    }
    
    /* Padding to prevent chat overlap with footer */
    .block-container {
        padding-bottom: 80px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Multimodal RAG System")
st.markdown("Interact with your documents using advanced Vision-Language capabilities.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Ensure essential directories exist
os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/images", exist_ok=True)

# --- Sidebar ---
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        if st.button("Ingest Document", use_container_width=True):
            with st.spinner("Processing document... (Extracting text, images, and generating embeddings)"):
                os.makedirs("data/uploads", exist_ok=True)
                tmp_path = os.path.join("data/uploads", uploaded_file.name)
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                try:
                    ingest_pdf(tmp_path)
                    st.toast("✅ Document ingested successfully!")
                except Exception as e:
                    st.error(f"Error during ingestion: {e}")
                    
    st.divider()
    
    st.header("📚 Knowledge Base")
    stats = get_db_stats()
    
    st.metric("Total Documents Indexed", stats["total_documents"])
    col1, col2 = st.columns(2)
    col1.metric("Text Chunks", stats["total_chunks"])
    col2.metric("Extracted Images", stats["total_images"])
    
    if stats["total_documents"] > 0:
        st.subheader("Indexed Documents:")
        for source in stats["unique_sources"]:
            st.markdown(f"- `{source}`")
    else:
        st.info("No documents indexed yet.")

# --- Main Chat Interface ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display retrieval stats if they exist for this message
        if "retrieval_stats" in message and message["retrieval_stats"]:
            with st.expander("🔍 Retrieval Statistics"):
                for stat in message["retrieval_stats"]:
                    st.markdown(stat)
                    
        # Display images
        if "images" in message and message["images"]:
            # Filter valid images
            valid_images = []
            for img_path in message["images"]:
                processed = get_white_bg_image(img_path)
                if processed is not None:
                    valid_images.append(processed)
            
            if valid_images:
                st.markdown("**Referenced Images:**")
                num_cols = min(len(valid_images), 3)
                cols = st.columns(num_cols)
                for i, img_obj in enumerate(valid_images):
                    with cols[i % num_cols]:
                        st.image(img_obj, caption="Retrieved Image", use_column_width=True)

if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process assistant response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant context..."):
            retrieved_chunks, retrieved_images = retrieve(prompt, top_k=3)
            
            # Format retrieval stats
            retrieval_stats = []
            if retrieved_chunks:
                for idx, chunk in enumerate(retrieved_chunks):
                    retrieval_stats.append(f"**Chunk {idx+1}:** {chunk['source']} (Page {chunk['page_num']})")
            else:
                retrieval_stats.append("No context chunks retrieved.")
        
        with st.spinner("Generating answer... (This may take a minute)"):
            try:
                answer = generate_answer(prompt, retrieved_chunks, retrieved_images)
                st.markdown(answer)
                
                with st.expander("🔍 Retrieval Statistics"):
                    for stat in retrieval_stats:
                        st.markdown(stat)
                
                if retrieved_images:
                    # Filter valid images
                    valid_images = []
                    for img_path in retrieved_images:
                        processed = get_white_bg_image(img_path)
                        if processed is not None:
                            valid_images.append(processed)
                            
                    if valid_images:
                        st.markdown("**Referenced Images:**")
                        num_cols = min(len(valid_images), 3)
                        cols = st.columns(num_cols)
                        for i, img_obj in enumerate(valid_images):
                            with cols[i % num_cols]:
                                st.image(img_obj, caption="Retrieved Image", use_column_width=True)
                            
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "images": retrieved_images,
                    "retrieval_stats": retrieval_stats
                })
            except Exception as e:
                st.error(f"Error during generation: {e}")

# --- Footer ---
st.markdown("""
    <div class="footer">
        <span>🤖 Local VLM: Moondream2</span> | 
        <span>🧠 Embedding: BGE-small</span> | 
        <span>🗄️ DB: ChromaDB</span>
    </div>
""", unsafe_allow_html=True)
