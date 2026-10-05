import os
from dotenv import load_dotenv

load_dotenv()
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

COLLECTION = "code"
DENSE_MODEL = "BAAI/bge-small-en-v1.5"   # 384 dims
SPARSE_MODEL = "Qdrant/bm25"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"   # check Groq's console for current model names

MAX_FILE_BYTES = 100_000
REPOS_DIR = "data/repos"