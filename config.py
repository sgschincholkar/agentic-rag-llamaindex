"""Central configuration: env vars, paths, and global LlamaIndex Settings."""

import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / os.getenv("PDF_PATH", "data/state.pdf")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-ada-002")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))


def configure_settings() -> None:
    """Set the global LLM and embedding model. Call once at startup."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    Settings.llm = OpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY)
    Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
