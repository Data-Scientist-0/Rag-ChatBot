"""
data_loader.py
==============
Downloads, cleans, and validates Wikipedia articles.
Returns a list of structured Document dicts ready for chunking.

Dataset  : Wikipedia Aerospace & Aviation History
License  : CC BY-SA 3.0  —  https://en.wikipedia.org/wiki/Wikipedia:Copyrights
Source   : https://en.wikipedia.org
"""
import os
import json
import re
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional

import wikipedia
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    RAW_DATA_DIR,
    RAW_ARTICLES_PATH,
    WIKI_ARTICLE_TITLES,
    DATASET_NAME,
    DATASET_LICENSE,
    DATASET_SOURCE_URL,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def clean_text(raw: str) -> str:
    text = re.sub(r"<[^>]+>", "", raw)
    text = re.sub(r"\[[^\]]{0,40}\]", "", text)
    text = re.sub(r"={2,}.*?={2,}", "", text)
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    text = re.sub(r"[^\x0A\x20-\x7E]", " ", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_section(text: str) -> str:
    for line in text.split("\n")[:20]:
        stripped = line.strip()
        if stripped and 5 < len(stripped) < 80:
            if not stripped.startswith("http") and not stripped[0].isdigit():
                return stripped[:60]
    return "Introduction"


def _build_doc(page, title_fallback: str) -> Optional[Dict]:
    """Build a document dict from a wikipedia page object. Returns None if too short."""
    raw_text = clean_text(page.content)
    if len(raw_text) < 300:
        return None
    return {
        "text": raw_text,
        "metadata": {
            "source_file": title_fallback.lower().replace(" ", "_").replace("/", "_")[:80] + ".txt",
            "title":       page.title,
            "section":     extract_section(raw_text),
            "date":        datetime.today().strftime("%Y-%m-%d"),
            "doc_id":      f"wiki_{page.pageid}",
            "url":         page.url,
            "char_count":  len(raw_text),
        },
    }


def fetch_article(title: str) -> Optional[Dict]:
    """
    Fetch a single Wikipedia article with full error handling.
    page.content is now INSIDE the try block — fixes the JSONDecodeError crash.
    Retries once on transient network errors.
    """
    for attempt in range(2):
        try:
            page = wikipedia.page(title, auto_suggest=False)
            doc  = _build_doc(page, title)
            if doc is None:
                logger.warning(f"Article too short, skipping: '{title}'")
            return doc

        except wikipedia.exceptions.DisambiguationError as e:
            for option in e.options[:3]:
                try:
                    page = wikipedia.page(option, auto_suggest=False)
                    doc  = _build_doc(page, option)
                    if doc:
                        return doc
                except Exception:
                    continue
            logger.warning(f"Disambiguation unresolvable: '{title}'")
            return None

        except wikipedia.exceptions.PageError:
            logger.warning(f"Page not found: '{title}'")
            return None

        except Exception as e:
            if attempt == 0:
                logger.warning(f"Attempt {attempt+1} failed for '{title}': {type(e).__name__}. Retrying...")
                time.sleep(3)
                continue
            else:
                logger.warning(f"Gave up on '{title}': {type(e).__name__}: {e}")
                return None

    return None


def load_dataset() -> List[Dict]:
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    if os.path.exists(RAW_ARTICLES_PATH):
        logger.info("Raw dataset cache found. Loading from disk...")
        with open(RAW_ARTICLES_PATH, "r", encoding="utf-8") as f:
            documents = json.load(f)
        logger.info(f"✅ Loaded {len(documents)} documents from cache.")
        return documents

    wikipedia.set_lang("en")
    logger.info(f"Downloading {len(WIKI_ARTICLE_TITLES)} Wikipedia articles...")
    logger.info(f"Dataset : {DATASET_NAME}")
    logger.info(f"License : {DATASET_LICENSE} — {DATASET_SOURCE_URL}")

    documents = []
    failed    = []

    for title in tqdm(WIKI_ARTICLE_TITLES, desc="Fetching Wikipedia articles"):
        doc = fetch_article(title)
        if doc:
            documents.append(doc)
        else:
            failed.append(title)
        time.sleep(0.4)

    if not documents:
        raise RuntimeError("No documents loaded. Check your internet connection.")

    if failed:
        logger.warning(f"Skipped {len(failed)} articles: {failed}")

    with open(RAW_ARTICLES_PATH, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    total_chars = sum(d["metadata"]["char_count"] for d in documents)
    logger.info(f"✅ Saved {len(documents)} documents | {total_chars:,} total chars")

    return documents


if __name__ == "__main__":
    docs = load_dataset()
    print(f"\nTotal documents : {len(docs)}")
    print(f"Total chars     : {sum(d['metadata']['char_count'] for d in docs):,}")
    print(f"Sample title    : {docs[0]['metadata']['title']}")