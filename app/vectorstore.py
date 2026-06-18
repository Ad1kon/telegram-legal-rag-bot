from os import getenv

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings as ChromaSettings


EMBEDDING_MODEL_NAME = getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
EMBEDDING_ENCODE_BATCH_SIZE = int(getenv("EMBEDDING_ENCODE_BATCH_SIZE", "8"))
_embeddings: "Embeddings | None" = None


class Embeddings:
    def __init__(self) -> None:
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts,
            batch_size=EMBEDDING_ENCODE_BATCH_SIZE,
            normalize_embeddings=True,
        ).tolist()


def get_embeddings() -> Embeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = Embeddings()
    return _embeddings


def get_client(chroma_path: str, anonymized_telemetry: bool = False):
    return chromadb.PersistentClient(
        path=chroma_path,
        settings=ChromaSettings(anonymized_telemetry=anonymized_telemetry),
    )


def get_collection(
    chroma_path: str,
    collection_name: str,
    anonymized_telemetry: bool = False,
):
    client = get_client(chroma_path, anonymized_telemetry)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
