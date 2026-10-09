"""Central configuration. Secrets come from environment variables / .env (never from the frontend)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
try:  # python-dotenv is optional
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

WEIGHTS_DIR = Path(os.getenv("PG_WEIGHTS_DIR", ROOT / "model_weights"))
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "plantguard.db"
MODEL_FILE = "plantguard_model.keras"

# Predictions below this calibrated confidence are NOT shown as a diagnosis.
LOW_CONF = float(os.getenv("PG_LOW_CONF", "0.60"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# Groq retires/renames models from time to time - check console.groq.com/docs/models if chat fails.
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

DATA_DIR.mkdir(exist_ok=True)
