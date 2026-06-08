"""
ingest.py
=========
One-command setup script. Run this ONCE before launching the UI.

Usage
-----
    python ingest.py

What it does
------------
    Step 1 — Downloads ~100 Wikipedia aerospace articles and saves to data/raw/articles.json
    Step 2 — Splits all articles into 500-char chunks (10% overlap)
    Step 3 — Embeds every chunk with BAAI/bge-large-en-v1.5 and stores in ChromaDB

After this completes, start the app with:
    streamlit run src/ui.py

Re-running ingest.py is safe — it detects an existing index and skips.
"""
import sys
import os
import time

# Make src/ importable from root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def main():
    start = time.time()

    print("\n" + "=" * 60)
    print("  AeroRAG — Knowledge Base Ingestion Pipeline")
    print("=" * 60)

    # ── Step 1: Load Dataset ───────────────────────────────────────────────
    print("\n[1/3] Downloading and cleaning Wikipedia articles...")
    from data_loader import load_dataset
    docs = load_dataset()
    print(f"      ✅ {len(docs)} documents loaded")

    # ── Step 2: Chunk Documents ────────────────────────────────────────────
    print("\n[2/3] Splitting into 500-char chunks (10% overlap)...")
    from chunker import chunk_documents
    chunks = chunk_documents(docs)
    print(f"      ✅ {len(chunks):,} chunks created")

    # ── Step 3: Embed and Store ────────────────────────────────────────────
    print("\n[3/3] Embedding with BAAI/bge-large-en-v1.5 → ChromaDB...")
    print("      (This step takes 5–15 min on CPU. Please wait.)")
    from embedder import embed_and_store
    meta = embed_and_store(chunks)

    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print("  ✅ Ingestion complete!")
    print(f"  Total chunks indexed : {meta.get('total_chunks', 0):,}")
    print(f"  Embedding time       : {meta.get('embedding_time_seconds', 0):.1f}s")
    print(f"  Total pipeline time  : {elapsed:.1f}s")
    print("=" * 60)
    print("\nNext step — start the app:")
    print("  streamlit run src/ui.py\n")


if __name__ == "__main__":
    main()
