# Engineering Report — AeroRAG Knowledge Base Chatbot

**Course:** Implementation of Intelligent Systems
**Project:** Engineering a Knowledge-Base RAG Chatbot

---

## 1. Dataset Selection & Provenance

**Dataset:** Wikipedia — Aerospace & Aviation History (~100 articles)
**License:** Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0)
**Source URL:** https://en.wikipedia.org
**License URL:** https://en.wikipedia.org/wiki/Wikipedia:Copyrights

Wikipedia was selected because it satisfies all syllabus constraints simultaneously:
it is open-source with a verifiable license, publicly accessible without authentication,
dense with structured factual content, and reproducibly downloadable via the
official `wikipedia` Python library. The aerospace domain provides rich factual
content (dates, names, technical specifications) that enables clear evaluation
of retrieval accuracy versus hallucination.

The dataset was downloaded programmatically from a fixed list of ~100 article
titles defined in `config.py`, ensuring identical results across machines and
runs. Articles shorter than 300 characters were discarded. Final corpus:
approximately 500,000+ characters across 90–100 articles.

---

## 2. Architecture Design & Component Rationale

```
Wikipedia Articles → data_loader → chunker → embedder → ChromaDB
                                                              ↓
User Query → retriever (cosine similarity) → generator (Ollama mistral) → UI
```

**LangChain** was chosen as the orchestration framework for its mature and
stable `RecursiveCharacterTextSplitter` API, which natively supports the
sliding-window strategy with configurable separators.

**ChromaDB** was selected over FAISS and Pinecone for three reasons: it is
fully local (no API key), persistence is built-in (survives restarts), and it
exposes the full metadata schema needed to display verified source strings in the UI.

**Ollama + mistral** was chosen over cloud LLM APIs to make the project fully
reproducible on any machine without API credentials or network access to external
services. Temperature was set to 0.0 to eliminate randomness in responses.

---

## 3. Chunking & Embedding Strategy

**Chunking:** `RecursiveCharacterTextSplitter` with `chunk_size=500` and
`chunk_overlap=50` (exactly 10%). The recursive separator hierarchy
(`\n\n`, `\n`, `. `, etc.) ensures splits happen at natural linguistic
boundaries rather than arbitrary character positions. A 10% overlap preserves
context that would otherwise be lost at chunk boundaries.

**Embedding:** `BAAI/bge-large-en-v1.5` was selected because it consistently
ranks among the top open-source models on the MTEB (Massive Text Embedding
Benchmark) retrieval tasks. Embeddings are L2-normalized before storage,
which makes the ChromaDB cosine distance computation equivalent to
dot-product similarity — the most numerically stable similarity measure for
high-dimensional vectors.

**BGE query prefix:** BGE-Large requires a specific instruction prefix on
query embeddings: `"Represent this sentence for searching relevant passages: "`.
Document embeddings do not use a prefix. This asymmetric encoding is the main
quality improvement BGE-Large offers over general-purpose models.

---

## 4. Retrieval & Prompt Engineering

**Retrieval:** ChromaDB is configured with `hnsw:space=cosine`. The distance
returned by ChromaDB is `1 − cosine_similarity`, so we convert:
`score = 1.0 − distance`. Top-4 chunks are retrieved (within the syllabus
range of 3–5). Any chunk scoring below 0.25 is discarded; if zero chunks
pass this threshold, the "missing info" response is returned immediately
without calling the LLM, preventing hallucination on empty context.

**Strict prompt design:** The system prompt is used verbatim as specified
in the syllabus:

> *"You are an objective AI assistant. Synthesize a response to the user query
> using ONLY the verified source context fragments provided below. If the answer
> cannot be confidently deduced directly from the text, reply explicitly with:
> 'The requested target info is missing from the provided dataset.'
> Do not extrapolate."*

This prompt design constrains the LLM to grounded answers, makes refusal
explicit, and provides a testable out-of-scope behaviour.

---

## 5. Proof Samples

See `notebooks/evaluation.ipynb` for the full test run with 10 questions.
Sample results below (human-judged):

| # | Question | Behaviour | Pass |
|---|---|---|---|
| 1 | Who were the Wright Brothers? | Correct answer with source | ✅ |
| 2 | When did Yuri Gagarin go to space? | Correct: 1961 | ✅ |
| 3 | What caused the Challenger disaster? | Correct: O-ring failure | ✅ |
| 4 | How does a jet engine work? | Correct with context | ✅ |
| 5 | What is the ISS? | Correct summary | ✅ |
| 6 | Compare Apollo 11 and Apollo 13 | Synthesised from multiple chunks | ✅ |
| 7 | What is the role of SpaceX in reusable rockets? | Correct with Falcon 9 details | ✅ |
| 8 | Who won the 2024 Super Bowl? | Refused: "missing from dataset" | ✅ |
| 9 | What is the population of Pakistan? | Refused: "missing from dataset" | ✅ |
| 10 | Who is the current Prime Minister of the UK? | Refused: "missing from dataset" | ✅ |

---

## 6. Known Limitations & Future Work

**Limitations:**
- Character-based chunk size (500 chars) is a practical approximation of
  the 500-token specification; actual token count varies by text density.
- Wikipedia text quality varies — some articles are stubs with little information.
- Ollama inference on CPU is slow (~15–30 seconds per query).

**Future improvements:**
- Replace Ollama mistral with a Groq API call (llama-3.3-70b-versatile) for
  10× faster inference at no cost.
- Add a re-ranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) between
  retrieval and generation to improve precision.
- Expand the dataset to include government aviation legislation and
  technical documentation for broader factual coverage.
