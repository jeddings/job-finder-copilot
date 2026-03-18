"""
ChromaDB ingestion: write chunks to the persistent vector store.
Uses local sentence-transformers embeddings (all-MiniLM-L6-v2).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.api import ClientAPI
from chromadb.utils import embedding_functions

from ingest.chunker import Chunk

CHROMA_PATH = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "jeff_eddings_career"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_client() -> ClientAPI:
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


def get_embedding_function():
    # Use ChromaDB's default ONNX-backed embedding function (all-MiniLM-L6-v2).
    # This avoids torch/numpy version conflicts while using the same model.
    return embedding_functions.DefaultEmbeddingFunction()


def get_collection(client: Optional[ClientAPI] = None):
    if client is None:
        client = get_client()
    ef = get_embedding_function()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection(client: Optional[ClientAPI] = None) -> None:
    """Delete and recreate the collection (used with --reset flag)."""
    if client is None:
        client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # Collection didn't exist yet
    get_collection(client)


def ingest_chunks(chunks: list[Chunk], reset: bool = False) -> int:
    """
    Ingest chunks into ChromaDB.
    If reset=True, clears the existing collection first.
    Returns number of chunks ingested.
    """
    client = get_client()

    if reset:
        reset_collection(client)

    collection = get_collection(client)

    # Batch ingest to avoid memory issues with large corpora
    batch_size = 50
    total = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.add(
            documents=[c.text for c in batch],
            metadatas=[c.metadata for c in batch],
            ids=[f"chunk_{i + j:04d}" for j in range(len(batch))],
        )
        total += len(batch)

    return total


def collection_count() -> int:
    """Return number of chunks currently in the collection."""
    return get_collection().count()
