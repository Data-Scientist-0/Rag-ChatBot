# ✈️ AeroRAG — Aerospace Knowledge Base Chatbot

An end-to-end **Retrieval-Augmented Generation (RAG)** system built on
Wikipedia's Aerospace and Aviation History corpus.

---

## Dataset

| Property | Value |
|---|---|
| **Name** | Wikipedia — Aerospace & Aviation History |
| **License** | [CC BY-SA 3.0](https://en.wikipedia.org/wiki/Wikipedia:Copyrights) |
| **Source URL** | https://en.wikipedia.org |
| **Size** | ~100 articles, ~500 000+ characters |
| **Download** | Automated via `wikipedia` Python library on first run |

---

## Architecture

```
Raw Wikipedia Articles (~100 articles)
           │
           ▼
  [ data_loader.py ]
  Strip HTML, citations, invalid chars
  Normalize whitespace
  Structured metadata per document
           │
           ▼
  [ chunker.py ]
  RecursiveCharacterTextSplitter
  chunk_size=500 chars │ overlap=50 chars (10%)
  Metadata inherited + extended per chunk
           │
           ▼
  [ embedder.py ]
  BAAI/bge-large-en-v1.5
  Normalized embeddings → ChromaDB (cosine, persistent)
           │
           ▼
  [ retriever.py ]
  BGE query prefix + embed user query
  Cosine similarity → top-4 chunks
  Threshold filter (score ≥ 0.25)
           │
           ▼
  [ generator.py ]
  Strict verbatim system prompt (syllabus spec)
  → Ollama mistral (temp=0.0)
           │
           ▼
  [ ui.py ]
  Streamlit chat interface
  ✅ Chat history
  ✅ Source metadata expander (verified from vector DB)
  ✅ Sidebar stats + status indicators
```

---

## Tech Stack

| Component | Choice | Reason |
|---|---|---|
| Orchestration | LangChain 0.2.14 | Stable text-splitting APIs; widely adopted |
| Embedding | BAAI/bge-large-en-v1.5 | Top-ranked open-source embedding model on MTEB |
| Vector DB | ChromaDB 0.5.3 | Local persistent storage, no API key needed |
| LLM | mistral via Ollama | Fully local inference, zero API cost, reproducible |
| UI | Streamlit 1.35.0 | Fast chat interface, native JSON display |

---

## Prerequisites

1. **Python 3.10+**
2. **[Ollama](https://ollama.ai)** installed and running

---

## Setup & Reproduction (zero manual configuration)

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd rag-chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Pull the LLM (one-time, ~4 GB)
ollama pull mistral

# 4. Start Ollama server (keep this running in a separate terminal)
ollama serve

# 5. Download dataset + build vector index (one-time, ~10–15 min on CPU)
python ingest.py

# 6. Launch the app
streamlit run src/ui.py
```

Open **http://localhost:8501** in your browser.

---

## Running Individual Components

```bash
# Download only
python src/data_loader.py

# Chunk only
python src/chunker.py

# Embed + index only
python src/embedder.py

# Test retrieval
python src/retriever.py

# Test full pipeline (no UI)
python src/generator.py
```

---

## Project Structure

```
rag-chatbot/
├── README.md
├── requirements.txt
├── .gitignore
├── ingest.py                  ← Run this once to build the index
├── src/
│   ├── __init__.py
│   ├── config.py              ← All constants (no magic numbers elsewhere)
│   ├── data_loader.py         ← Download + clean Wikipedia articles
│   ├── chunker.py             ← 500-char sliding window, 10% overlap
│   ├── embedder.py            ← BGE-Large embeddings → ChromaDB
│   ├── retriever.py           ← Cosine similarity retrieval
│   ├── generator.py           ← Strict prompt → Ollama mistral
│   └── ui.py                  ← Streamlit chat + source metadata display
├── data/
│   ├── raw/                   ← articles.json (auto-downloaded)
│   └── processed/             ← ChromaDB index (gitignored)
├── docs/
│   └── engineering_report.md
└── notebooks/
    └── evaluation.ipynb
```

---

## Error Reference

| Error | Fix |
|---|---|
| `🔴 Index Not Found` | Run `python ingest.py` |
| `🟡 Ollama Offline` | Run `ollama serve` in a separate terminal |
| `Model not found` | Run `ollama pull mistral` |
| Slow first query | Embedding model loading on first query (~30s) |
