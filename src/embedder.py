"""
embedder.py
===========
Generates BAAI/bge-large-en-v1.5 embeddings and stores them in ChromaDB.

Key behaviours:
  - Checks if collection already exists before rebuilding (no redundant work).
  - Normalizes embeddings before storage (required for cosine similarity).
  - Stores ALL metadata fields required by the syllabus.
  - Saves index_metadata.json for the UI sidebar.
"""
import os
import sys
import json
import logging
import time
from typing import List, Dict

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    DISTANCE_METRIC,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    EMBEDDING_BATCH_SIZE,
    NORMALIZE_EMBEDDINGS,
    INDEX_META_PATH,
    PROCESSED_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


# ── ChromaDB Client ────────────────────────────────────────────────────────────

def get_chroma_client() -> chromadb.PersistentClient:
    """Create or reopen a persistent ChromaDB client."""
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    return chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )


def collection_is_populated(client: chromadb.PersistentClient) -> bool:
    """
    Return True if the collection exists AND contains at least one vector.
    Prevents rebuilding on every restart.
    """
    try:
        col   = client.get_collection(COLLECTION_NAME)
        count = col.count()
        if count > 0:
            logger.info(f"Existing index found — {count} chunks already stored. Skipping rebuild.")
            return True
        return False
    except Exception:
        return False


# ── Embedding Model ────────────────────────────────────────────────────────────

def load_embedding_model() -> SentenceTransformer:
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME} on [{EMBEDDING_DEVICE}]")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)
    logger.info("Embedding model loaded.")
    return model


# ── Main Indexing Function ─────────────────────────────────────────────────────

def embed_and_store(chunks: List[Dict]) -> Dict:
    """
    Embed all chunks and persist to ChromaDB.

    Parameters
    ----------
    chunks : list of dicts from chunker.chunk_documents()

    Returns
    -------
    dict with index metadata summary (also saved to index_metadata.json)
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    client = get_chroma_client()

    # ── Guard: skip if already indexed ────────────────────────────────────────
    if collection_is_populated(client):
        return load_index_metadata()

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_embedding_model()

    # ── Create collection (cosine distance) ───────────────────────────────────
    # Drop any partial/empty collection from a previous failed run
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info("Dropped previous partial collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": DISTANCE_METRIC},
    )

    texts = [c["chunk_text"] for c in chunks]
    logger.info(f"Embedding {len(texts)} chunks (batch size={EMBEDDING_BATCH_SIZE})...")

    start_time = time.time()
    all_embeddings: List[List[float]] = []

    # ── Embed in batches ──────────────────────────────────────────────────────
    for i in tqdm(range(0, len(texts), EMBEDDING_BATCH_SIZE), desc="Embedding"):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        # NOTE: for DOCUMENT embeddings, BGE-Large does NOT require a prefix.
        # The query prefix is applied only at retrieval time (retriever.py).
        vecs = model.encode(
            batch,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
            show_progress_bar=False,
            device=EMBEDDING_DEVICE,
        )
        all_embeddings.extend(vecs.tolist())

    elapsed = time.time() - start_time
    logger.info(f"Embedding complete in {elapsed:.1f}s")

    # ── Store in ChromaDB in batches ──────────────────────────────────────────
    STORE_BATCH = 500
    logger.info("Persisting vectors to ChromaDB...")

    for i in tqdm(range(0, len(chunks), STORE_BATCH), desc="Storing"):
        batch_chunks     = chunks[i : i + STORE_BATCH]
        batch_embeddings = all_embeddings[i : i + STORE_BATCH]
        batch_ids        = [f"chunk_{i + j}" for j in range(len(batch_chunks))]

        # ChromaDB requires flat metadata — all values must be str/int/float/bool.
        # We store chunk_text inside metadata so retrieval returns it directly.
        batch_metas = []
        for c in batch_chunks:
            m = c["metadata"]
            batch_metas.append({
                "chunk_text":   c["chunk_text"],
                "chunk_index":  int(m["chunk_index"]),
                "total_chunks": int(m["total_chunks"]),
                "char_start":   int(m["char_start"]),
                "char_end":     int(m["char_end"]),
                "source_file":  str(m["source_file"]),
                "title":        str(m["title"]),
                "section":      str(m["section"]),
                "doc_id":       str(m["doc_id"]),
            })

        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=[c["chunk_text"] for c in batch_chunks],
            metadatas=batch_metas,
        )

    # ── Save index metadata ───────────────────────────────────────────────────
    avg_len = sum(len(t) for t in texts) / len(texts)

    index_meta = {
        "total_chunks":             len(chunks),
        "embedding_model":          EMBEDDING_MODEL_NAME,
        "embedding_device":         EMBEDDING_DEVICE,
        "embedding_time_seconds":   round(elapsed, 2),
        "avg_chunk_length_chars":   round(avg_len, 1),
        "chunk_size":               CHUNK_SIZE,
        "chunk_overlap":            CHUNK_OVERLAP,
        "collection_name":          COLLECTION_NAME,
        "distance_metric":          DISTANCE_METRIC,
        "normalize_embeddings":     NORMALIZE_EMBEDDINGS,
    }

    with open(INDEX_META_PATH, "w") as f:
        json.dump(index_meta, f, indent=2)

    logger.info(f"✅ Index built: {len(chunks)} chunks stored in {elapsed:.1f}s")
    logger.info(f"   Metadata saved to {INDEX_META_PATH}")

    return index_meta


def load_index_metadata() -> Dict:
    """Load saved index metadata if it exists, else return empty dict."""
    if os.path.exists(INDEX_META_PATH):
        with open(INDEX_META_PATH, "r") as f:
            return json.load(f)
    return {}


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import load_dataset
    from chunker import chunk_documents

    docs   = load_dataset()
    chunks = chunk_documents(docs)
    meta   = embed_and_store(chunks)

    print(f"\n{'='*55}")
    print("Index metadata:")
    for k, v in meta.items():
        print(f"  {k:<35} {v}")
    print(f"{'='*55}")
