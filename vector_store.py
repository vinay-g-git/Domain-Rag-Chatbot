"""
vector_store.py
----------------
Module 3 (Chunking) + Module 4 (Embeddings and Vector Store) + Module 5
(Retrieval).

- Splits extracted pages into overlapping chunks with LangChain's text
  splitter, keeping (document, page) metadata on every chunk.
- Embeds chunks with a local Sentence-Transformers model
  (all-MiniLM-L6-v2 by default -- small, free, good for beginners).
- Stores vectors + metadata in a FAISS index that can be saved/loaded
  from disk so the app doesn't have to re-embed on every run.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from document_loader import PageRecord

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 4


@dataclass
class Chunk:
    """A chunk of text with its originating document and page number."""
    text: str
    source_document: str
    page_number: int
    chunk_id: int


def split_pages_into_chunks(
    pages: list[PageRecord],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split each page's text into overlapping chunks, preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    chunk_id = 0
    for page in pages:
        for piece in splitter.split_text(page.text):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    text=piece,
                    source_document=page.source_document,
                    page_number=page.page_number,
                    chunk_id=chunk_id,
                )
            )
            chunk_id += 1
    return chunks


class VectorStore:
    """A thin FAISS wrapper that keeps chunk metadata alongside the index."""

    def __init__(self, embedding_model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.embedding_model_name = embedding_model_name
        self._model: SentenceTransformer | None = None
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def build(self, chunks: list[Chunk]) -> None:
        """Embed all chunks and build a fresh FAISS index (cosine similarity)."""
        if not chunks:
            raise ValueError("No chunks to index -- did text extraction find any content?")

        self.chunks = chunks
        texts = [c.text for c in chunks]
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")

        dim = embeddings.shape[1]
        # Inner product on normalized vectors == cosine similarity.
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[tuple[Chunk, float]]:
        """Return the top_k most similar chunks to the query with their scores."""
        if self.index is None or not self.chunks:
            return []

        query_vec = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, indices = self.index.search(query_vec, min(top_k, len(self.chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, directory: str) -> None:
        """Persist the FAISS index + chunk metadata to disk."""
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(out_dir / "index.faiss"))
        with open(out_dir / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        with open(out_dir / "config.json", "w") as f:
            json.dump({"embedding_model_name": self.embedding_model_name}, f)

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        """Load a previously saved index + metadata from disk."""
        in_dir = Path(directory)
        with open(in_dir / "config.json") as f:
            config = json.load(f)

        store = cls(embedding_model_name=config["embedding_model_name"])
        store.index = faiss.read_index(str(in_dir / "index.faiss"))
        with open(in_dir / "chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)
        return store

    def to_debug_dicts(self) -> list[dict]:
        """Helper for tests/inspection -- dump chunks as plain dicts."""
        return [asdict(c) for c in self.chunks]
