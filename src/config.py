"""
config.py
=========
Single source of truth for every tunable constant.
No magic numbers anywhere else in the codebase.
"""
import os
from dotenv import load_dotenv

load_dotenv()

try:
    import torch
    _cuda_available = torch.cuda.is_available()
except ImportError:
    _cuda_available = False

# ── Directory Paths ────────────────────────────────────────────────────────────
BASE_DIR           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR       = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR      = os.path.join(BASE_DIR, "data", "processed")
CHROMA_DB_PATH     = os.path.join(PROCESSED_DIR, "chroma_db")
INDEX_META_PATH    = os.path.join(PROCESSED_DIR, "index_metadata.json")
RAW_ARTICLES_PATH  = os.path.join(RAW_DATA_DIR, "articles.json")

# ── Dataset Metadata ────────────────────────────────────────────────────────────
DATASET_NAME        = "Wikipedia Aerospace & Aviation History"
DATASET_LICENSE     = "CC BY-SA 3.0"
DATASET_SOURCE_URL  = "https://en.wikipedia.org"
DATASET_LICENSE_URL = "https://en.wikipedia.org/wiki/Wikipedia:Copyrights"

# ── Text Chunking (Syllabus Hard Requirements) ─────────────────────────────────
CHUNK_SIZE         = 500
CHUNK_OVERLAP      = 50
CHUNK_SEPARATORS   = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

# ── Embedding Model ────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME  = "all-MiniLM-L6-v2"
EMBEDDING_DEVICE      = "cuda" if _cuda_available else "cpu"
EMBEDDING_BATCH_SIZE  = 32
NORMALIZE_EMBEDDINGS  = True
BGE_QUERY_PREFIX      = ""

# ── ChromaDB ───────────────────────────────────────────────────────────────────
COLLECTION_NAME    = "rag_knowledge_base"
DISTANCE_METRIC    = "cosine"

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K              = 15
SIMILARITY_THRESHOLD = 0.15

# ── LLM — Groq API ─────────────────────────────────────────────────────────────
LLM_MODEL          = "llama-3.3-70b-versatile"
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL       = "https://api.groq.com/openai/v1/chat/completions"
LLM_TEMPERATURE    = 0.0
LLM_TOP_P          = 1.0
LLM_MAX_TOKENS     = 512
OLLAMA_TIMEOUT     = 30

# ── System Prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = (
    "You are an objective AI assistant. Synthesize a response to the user query "
    "using ONLY the verified source context fragments provided below. If the answer "
    "cannot be reasonably inferred from the provided context, reply explicitly with: "
    "'The requested target info is missing from the provided dataset.' "
    "Do not extrapolate.\n\n"
    "[CONTEXT]: {retrieved_chunks}\n"
    "[USER QUERY]: {user_query}"
)

MISSING_INFO_RESPONSE = (
    "The requested target info is missing from the provided dataset."
)

# ── Wikipedia Article List ─────────────────────────────────────────────────────
WIKI_ARTICLE_TITLES = [
    "History of aviation", "Wright brothers", "Charles Lindbergh",
    "Amelia Earhart", "Jet engine", "Space exploration",
    "Apollo program", "International Space Station", "Neil Armstrong",
    "Yuri Gagarin", "NASA", "SpaceX", "Boeing", "Airbus",
    "Concorde", "Supersonic aircraft", "Helicopter", "Hot air balloon",
    "Rocket", "Satellite", "Hubble Space Telescope", "Mars rover",
    "Voyager program", "Space Shuttle", "Sputnik 1", "Mercury program",
    "Gemini program", "Cassini–Huygens", "New Horizons",
    "Kepler space telescope", "James Webb Space Telescope",
    "Falcon 9", "Saturn V", "V-2 rocket", "Robert Goddard",
    "Wernher von Braun", "Alan Shepard", "Buzz Aldrin",
    "John Glenn", "Valentina Tereshkova", "Space Race",
    "Apollo 11", "Apollo 13", "Challenger disaster",
    "Columbia disaster", "X-15", "Lockheed SR-71 Blackbird",
    "Boeing 747", "Airbus A380", "Zeppelin", "Hindenburg disaster",
    "Montgolfier brothers", "Otto Lilienthal", "Alberto Santos-Dumont",
    "First World War aviation", "Second World War aviation",
    "Turbojet", "Turbofan", "Aerodynamics", "Sound barrier",
    "Chuck Yeager", "Commercial aviation", "Air traffic control",
    "Aviation safety", "Unmanned aerial vehicle",
    "Blue Origin", "Virgin Galactic", "Orbital mechanics",
    "Low Earth orbit", "Moon landing", "Space suit", "Spacewalk",
    "Mir", "Skylab", "Tiangong space station",
    "Roscosmos", "European Space Agency", "ISRO", "JAXA",
    "Chandrayaan", "Starship (spacecraft)", "Dragon (spacecraft)",
    "Soyuz (spacecraft)", "Space debris", "Astronaut", "Cosmonaut",
    "Ion thruster", "Reusable launch vehicle", "Scramjet",
    "Douglas DC-3", "Boeing 707", "Northrop Grumman",
    "Aircraft engine", "Glider (aircraft)", "Seaplane",
    "Propeller (aircraft)", "Wind tunnel", "Flight simulator",
    "Air force", "Aircraft carrier", "Stealth aircraft",
    "Reconnaissance aircraft", "Bomber aircraft",
    "Fighter aircraft", "General aviation",
]