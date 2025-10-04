import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", "chroma_db"))

# Models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
