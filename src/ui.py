"""
ui.py
=====
Streamlit chat interface with automated knowledge base initialization.
"""
import os
import sys
import json
import subprocess
import streamlit as st

# Make src/ importable when running `streamlit run src/ui.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    LLM_MODEL,
    CHROMA_DB_PATH,
    INDEX_META_PATH,
    TOP_K,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SIMILARITY_THRESHOLD,
    DATASET_NAME,
    DATASET_LICENSE,
)
from generator import generate, check_ollama_connection

# ── 1. Page Configuration MUST BE FIRST ────────────────────────────────────────
st.set_page_config(
    page_title="AeroRAG — Aerospace Knowledge Base",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 2. Database Fallback Initialization ────────────────────────────────────────
# Check if the database directory does not exist or has no files inside
if not os.path.exists(CHROMA_DB_PATH) or not os.listdir(CHROMA_DB_PATH):
    st.info("📦 First-time setup: Initializing Knowledge Base on the server... Please wait.")
    
    try:
        # Runs the ingestion script as a separate process
        subprocess.run([sys.executable, "ingest.py"], check=True)
        st.success("✅ Knowledge Base generated successfully!")
        st.rerun()  # Reloads the app now that data exists
    except Exception as e:
        st.error(f"❌ Failed to initialize index: {e}")
        st.stop()   # Stop execution if database is missing


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_index_meta() -> dict:
    if os.path.exists(INDEX_META_PATH):
        with open(INDEX_META_PATH, "r") as f:
            return json.load(f)
    return {}


def index_is_ready() -> bool:
    """Return True only if ChromaDB collection exists and has data."""
    try:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        col = client.get_collection(COLLECTION_NAME)
        return col.count() > 0
    except Exception:
        return False


def render_source_expander(sources: list, key_prefix: str = "") -> None:
    with st.expander("🔍 Retrieved Sources (Verified Metadata)", expanded=False):
        for i, s in enumerate(sources, 1):
            st.markdown(f"**Source {i} | Relevance Score: {s['score']:.3f}**")
            st.json({
                "source_file": s["source_file"],
                "chunk_index": s["chunk_index"],
                "char_range":  f"{s['char_start']}–{s['char_end']}",
                "section":     s.get("section", "N/A"),
                "text_preview": s["chunk_text"][:200] + "...",
            })
            if i < len(sources):
                st.markdown("---")


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🛠️ System Status")
    st.divider()

    ready = index_is_ready()
    if ready:
        st.success("🟢 System Ready")
        meta = load_index_meta()
        c1, c2 = st.columns(2)
        c1.metric("Chunks", f"{meta.get('total_chunks', '—'):,}")
        c2.metric("Avg Length", f"{int(meta.get('avg_chunk_length_chars', 0))} ch")
    else:
        st.error("🔴 Index Not Found")

    if check_ollama_connection():
        st.success("🟢 Ollama Connected")
    else:
        st.warning("🟡 Ollama Offline")

    st.divider()
    st.subheader("⚙️ Configuration")
    st.write(f"**Embedding model:** `{EMBEDDING_MODEL_NAME}`")
    st.write(f"**LLM:** `{LLM_MODEL}`")


# ── Main Header ────────────────────────────────────────────────────────────────
st.title("✈️ AeroRAG — Aerospace Knowledge Base Chatbot")

# ── Session State Init ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Chat Logic ─────────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_source_expander(msg["sources"])

chat_input = st.chat_input("Ask a question about aerospace history...")

if chat_input:
    st.session_state.messages.append({"role": "user", "content": chat_input})
    with st.chat_message("user"):
        st.markdown(chat_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base…"):
            answer, sources = generate(chat_input)
        st.markdown(answer)
        if sources:
            render_source_expander(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
