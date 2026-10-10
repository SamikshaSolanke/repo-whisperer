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
GROQ_MODEL = "openai/gpt-oss-120b"   # check Groq's console for current model names

MAX_FILE_BYTES = 100_000
REPOS_DIR = "data/repos"

DENSE_DIM = 384                     # bge-small-en-v1.5 output size
MAX_CHUNK_CHARS = 2000              # ~450 tokens; bge-small truncates at 512 tokens
OVERLAP_LINES = 5
INCLUDE_TESTS = False               # flip to True later for an ablation

TOP_K = 6                  # chunks sent to the LLM
GROQ_TEMPERATURE = 0.1     # low = more faithful to the code
GROQ_MAX_TOKENS = 1024

EXTENSIONS = {".py": "python", ".md": "markdown", ".rst": "rst"}
SKIP_DIRS = {".git", ".github", ".devcontainer", "node_modules", "venv", ".venv",
             "__pycache__", "_build", "_static", "dist", "build"}
TEST_DIRS = {"tests", "test"}
SKIP_FILES = {"CHANGES.rst", "LICENSE.txt"}