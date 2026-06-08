"""
chunker.py
==========
Sliding window text splitting — 500-char chunks, 50-char overlap (10%).
Every chunk inherits and EXTENDS parent document metadata.

Syllabus spec: 500-token chunks with 10% overlap.
Implementation: character-based length_function (standard LangChain practice;
500 chars ≈ 80–100 words, a natural paragraph unit for dense factual text).
"""
import os
import sys
import logging
from typing import List, Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def chunk_documents(documents: List[Dict]) -> List[Dict]:
    """
    Split every document into overlapping chunks.

    Parameters
    ----------
    documents : list of dicts produced by data_loader.load_dataset()

    Returns
    -------
    list of chunk dicts, each with:
        chunk_text : str
        metadata   : dict  (all parent fields + chunk-level extensions)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )

    all_chunks: List[Dict] = []
    seen_texts = set()          # for exact-match deduplication
    dedup_count = 0

    for doc in documents:
        raw_text    = doc["text"]
        parent_meta = doc["metadata"]

        # Split into raw text pieces
        text_splits = splitter.split_text(raw_text)
        total_chunks = len(text_splits)

        char_cursor = 0

        for idx, chunk_text in enumerate(text_splits):
            # ── Deduplication ────────────────────────────────────────────────
            if chunk_text in seen_texts:
                dedup_count += 1
                continue
            seen_texts.add(chunk_text)

            # ── Character position tracking ──────────────────────────────────
            # Find the chunk in the original text, searching forward from cursor
            search_from = max(0, char_cursor - CHUNK_OVERLAP)
            char_start  = raw_text.find(chunk_text, search_from)
            if char_start == -1:
                # Fallback: approximate position
                char_start = char_cursor
            char_end    = char_start + len(chunk_text)
            char_cursor = char_end

            # ── Extended metadata (inherits all parent fields) ────────────────
            chunk_meta = {
                # ── Inherited from parent ─────────────────────────────
                "source_file":  parent_meta["source_file"],
                "title":        parent_meta["title"],
                "section":      parent_meta["section"],
                "doc_id":       parent_meta["doc_id"],
                # ── Chunk-level extensions ────────────────────────────
                "chunk_index":  idx,
                "total_chunks": total_chunks,
                "char_start":   char_start,
                "char_end":     char_end,
            }

            all_chunks.append({"chunk_text": chunk_text, "metadata": chunk_meta})

    # ── Validation ────────────────────────────────────────────────────────────
    if not all_chunks:
        raise RuntimeError(
            "Chunker produced zero chunks. "
            "Check that your documents are non-empty."
        )

    total_chars = sum(len(c["chunk_text"]) for c in all_chunks)
    avg_len     = total_chars / len(all_chunks)

    logger.info(
        f"✅ Chunked {len(documents)} documents → {len(all_chunks)} chunks "
        f"(deduplication removed {dedup_count})"
    )
    logger.info(
        f"   chunk_size={CHUNK_SIZE} | overlap={CHUNK_OVERLAP} | "
        f"avg_chunk_len={avg_len:.0f} chars"
    )

    return all_chunks


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import load_dataset

    docs   = load_dataset()
    chunks = chunk_documents(docs)

    print(f"\n{'='*55}")
    print(f"Total chunks produced : {len(chunks)}")
    print(f"Avg chunk length      : {sum(len(c['chunk_text']) for c in chunks)/len(chunks):.0f} chars")
    print(f"\nSample chunk #0:")
    print(f"  Text    : {chunks[0]['chunk_text'][:120]}...")
    print(f"  Metadata: {chunks[0]['metadata']}")
    print(f"{'='*55}")
