"""
retriever.py
============
Query embedding + cosine similarity retrieval from ChromaDB.

Key behaviours:
  - Applies BGE-Large's required query prefix to the user's question.
  - Converts ChromaDB cosine distance → cosine similarity score.
  - Filters out chunks below SIMILARITY_THRESHOLD (no empty context to LLM).
  - Assembles context string in the EXACT format specified by the syllabus.
"""
import os
import sys
import logging
from typing import List, Dict, Optional

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    NORMALIZE_EMBEDDINGS,
    BGE_QUERY_PREFIX,
    TOP_K,
    SIMILARITY_THRESHOLD,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


# ── Module-level singletons (loaded once, reused across all queries) ───────────
_model:      Optional[SentenceTransformer] = None
_collection: Optional[object]              = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model for retrieval: {EMBEDDING_MODEL_NAME}")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        try:
            _collection = client.get_collection(COLLECTION_NAME)
            count = _collection.count()
            if count == 0:
                raise RuntimeError("Collection exists but is empty.")
            logger.info(f"Collection '{COLLECTION_NAME}' loaded — {count} chunks available.")
        except Exception as e:
            raise RuntimeError(
                f"ChromaDB collection '{COLLECTION_NAME}' not found or empty.\n"
                "Please run ingestion first:\n"
                "  python ingest.py\n"
                f"  (original error: {e})"
            ) from e
    return _collection


# ── Core Retrieval ─────────────────────────────────────────────────────────────

def retrieve(query: str) -> List[Dict]:
    """
    Embed the user query and retrieve the top-K most relevant chunks
    via cosine similarity.

    Returns
    -------
    list of result dicts (sorted by score DESC), each containing:
        score, chunk_text, chunk_index, source_file, title,
        section, char_start, char_end, doc_id

    Returns an empty list if:
      - query is blank
      - no chunk exceeds SIMILARITY_THRESHOLD
    """
    if not query.strip():
        return []

    model      = _get_model()
    collection = _get_collection()

    # ── Embed query with BGE instruction prefix ────────────────────────────
    query_input     = BGE_QUERY_PREFIX + query.strip()
    query_embedding = model.encode(
        [query_input],
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        show_progress_bar=False,
        device=EMBEDDING_DEVICE,
    ).tolist()

    # ── Query ChromaDB ─────────────────────────────────────────────────────
    n_results = min(TOP_K, collection.count())
    results   = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    if not results["ids"] or not results["ids"][0]:
        return []

    # ── Parse results + apply threshold ───────────────────────────────────
    retrieved: List[Dict] = []

    for i, chunk_id in enumerate(results["ids"][0]):
        # ChromaDB cosine distance: distance = 1 − cosine_similarity
        # (true for normalized vectors + hnsw:space=cosine)
        distance = results["distances"][0][i]
        score    = round(1.0 - distance, 4)     # cosine similarity

        if score < SIMILARITY_THRESHOLD:
            logger.debug(
                f"Chunk {chunk_id} below threshold "
                f"(score={score:.3f} < {SIMILARITY_THRESHOLD}). Discarded."
            )
            continue

        meta = results["metadatas"][0][i]
        text = results["documents"][0][i]

        retrieved.append({
            "score":       score,
            "chunk_text":  text,
            "chunk_index": meta.get("chunk_index", 0),
            "source_file": meta.get("source_file", "unknown"),
            "title":       meta.get("title", "unknown"),
            "section":     meta.get("section", "N/A"),
            "char_start":  meta.get("char_start", 0),
            "char_end":    meta.get("char_end", 0),
            "doc_id":      meta.get("doc_id", ""),
        })

    # Sort by score descending (ChromaDB usually returns sorted, but enforce it)
    retrieved.sort(key=lambda x: x["score"], reverse=True)

    logger.info(
        f"Query: '{query[:60]}' → "
        f"{len(retrieved)} chunks above threshold {SIMILARITY_THRESHOLD}"
    )

    return retrieved


# ── Context Assembly ───────────────────────────────────────────────────────────

def build_context_string(retrieved: List[Dict]) -> str:
    """
    Assemble retrieved chunks into the EXACT context format
    specified in the syllabus (Section 6).

    Format per chunk:
      [CONTEXT N - Source: <file>, Chunk: <idx>, Score: <x.xxx>]
      <chunk text>
    """
    if not retrieved:
        return ""

    parts = []
    for i, r in enumerate(retrieved, 1):
        header = (
            f"[CONTEXT {i} - Source: {r['source_file']}, "
            f"Chunk: {r['chunk_index']}, Score: {r['score']:.3f}]"
        )
        parts.append(f"{header}\n{r['chunk_text']}")

    return "\n\n".join(parts)


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    query   = "Who were the Wright Brothers and when did they first fly?"
    results = retrieve(query)
    context = build_context_string(results)

    print(f"\nQuery   : {query}")
    print(f"Results : {len(results)} chunks\n")
    print(context[:1000])
