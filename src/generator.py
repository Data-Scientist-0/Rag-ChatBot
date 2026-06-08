"""
generator.py
============
LLM response generation via Groq API.
Uses the VERBATIM strict system prompt from the syllabus.
"""
import os
import sys
import logging
from typing import List, Dict, Tuple

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    LLM_MODEL,
    GROQ_API_KEY,
    GROQ_API_URL,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_MAX_TOKENS,
    OLLAMA_TIMEOUT,
    SYSTEM_PROMPT_TEMPLATE,
    MISSING_INFO_RESPONSE,
)
from retriever import retrieve, build_context_string

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def check_ollama_connection() -> bool:
    try:
        resp = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            timeout=5
        )
        return resp.status_code == 200
    except Exception:
        return False


def call_llm(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model":       LLM_MODEL,
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": LLM_TEMPERATURE,
        "max_tokens":  LLM_MAX_TOKENS,
        "top_p":       LLM_TOP_P,
    }
    try:
        resp = requests.post(
            GROQ_API_URL,
            headers=headers,
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"].strip()
        return answer
    except requests.exceptions.ConnectionError:
        return "⚠️ Cannot connect to Groq API. Check your internet connection."
    except requests.exceptions.Timeout:
        return "⚠️ The model is taking too long to respond. Please try again."
    except requests.exceptions.HTTPError as e:
        return f"⚠️ Groq API error: {e}"
    except Exception as e:
        return f"⚠️ Unexpected error: {type(e).__name__}: {e}"


def generate(user_query: str) -> Tuple[str, List[Dict]]:
    if not user_query.strip():
        return "Please enter a question.", []

    try:
        retrieved = retrieve(user_query)
    except RuntimeError as e:
        return f"⚠️ Retrieval error: {e}", []

    if not retrieved:
        return MISSING_INFO_RESPONSE, []

    context_string = build_context_string(retrieved)

    full_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        retrieved_chunks=context_string,
        user_query=user_query.strip(),
    )

    logger.info(f"Generating response | query='{user_query[:60]}' | context_chunks={len(retrieved)}")

    answer = call_llm(full_prompt)
    return answer, retrieved


if __name__ == "__main__":
    queries = [
        "Who were the Wright Brothers?",
    ]
    for q in queries:
        print(f"\nQ: {q}")
        answer, sources = generate(q)
        print(f"A: {answer[:300]}")
        print(f"Sources: {len(sources)}")