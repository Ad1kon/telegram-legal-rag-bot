from dataclasses import dataclass
from os import getenv

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    groq_api_key: str
    groq_model: str
    chroma_path: str
    raw_docs_path: str
    collection_name: str
    top_k: int
    ingest_batch_size: int
    anonymized_telemetry: bool


def get_settings() -> Settings:
    return Settings(
        telegram_bot_token=getenv("TELEGRAM_BOT_TOKEN", ""),
        groq_api_key=getenv("GROQ_API_KEY", ""),
        groq_model=getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        chroma_path=getenv("CHROMA_PATH", "data/chroma"),
        raw_docs_path=getenv("RAW_DOCS_PATH", "data/raw"),
        collection_name=getenv("COLLECTION_NAME", "legal_documents"),
        top_k=int(getenv("TOP_K", "6")),
        ingest_batch_size=int(getenv("INGEST_BATCH_SIZE", "64")),
        anonymized_telemetry=getenv("ANONYMIZED_TELEMETRY", "False").lower() == "true",
    )
