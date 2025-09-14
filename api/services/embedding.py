"""
Embedding service using sentence-transformers with Gemma model.

This module provides text embedding functionality for dense vector search,
with fallback options for offline usage.
"""

import asyncio
import logging
import os
from typing import List, Optional

import numpy as np
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings using sentence-transformers.

    Uses a lightweight model suitable for job-candidate matching with
    fallback options for offline usage.
    """

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None
    ):
        """
        Initialize the embedding service.

        Args:
            model_name: Name of the sentence-transformer model to use
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[SentenceTransformer] = None
        self.embedding_dim: int = 384

    async def initialize(self) -> None:
        """
        Initialize the embedding model asynchronously.

        Raises:
            RuntimeError: If model initialization fails
        """
        try:
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")

            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, lambda: SentenceTransformer(self.model_name, device=self.device)
            )

            test_embedding = await self.encode_single("test")
            self.embedding_dim = len(test_embedding)

        except Exception as e:
            raise RuntimeError(f"Embedding model initialization failed: {e}")

    async def encode_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Numpy array containing the text embedding

        Raises:
            RuntimeError: If model is not initialized or encoding fails
        """
        if self.model is None:
            raise RuntimeError(
                "Embedding model not initialized. Call initialize() first."
            )

        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, lambda: self.model.encode(text, convert_to_numpy=True)
            )

            return embedding.astype(np.float32)

        except Exception as e:
            raise RuntimeError(f"Text encoding failed: {e}")

    async def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed

        Returns:
            List of numpy arrays containing the text embeddings

        Raises:
            RuntimeError: If model is not initialized or encoding fails
        """
        if self.model is None:
            raise RuntimeError(
                "Embedding model not initialized. Call initialize() first."
            )

        if not texts:
            return []

        try:
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(texts, convert_to_numpy=True, batch_size=32),
            )

            return [emb.astype(np.float32) for emb in embeddings]

        except Exception as e:
            raise RuntimeError(f"Batch text encoding failed: {e}")

    def get_embedding_dim(self) -> int:
        """
        Get the embedding dimension.

        Returns:
            Embedding vector dimension
        """
        return self.embedding_dim

    def is_initialized(self) -> bool:
        """
        Check if the embedding model is initialized.

        Returns:
            True if model is ready, False otherwise
        """
        return self.model is not None


def create_searchable_text(job_or_candidate: dict) -> str:
    """
    Create searchable text representation for jobs or candidates.

    Args:
        job_or_candidate: Job or candidate data as dictionary

    Returns:
        Concatenated text suitable for embedding and search
    """
    if "title" in job_or_candidate and "description" in job_or_candidate:
        skills = " ".join(job_or_candidate.get("skills", []))
        location = (
            f"{job_or_candidate['location']['city']}, "
            f"{job_or_candidate['location']['country']}"
        )

        return (
            f"{job_or_candidate['title']} {job_or_candidate['description']} "
            f"{skills} {location} {job_or_candidate.get('employment_type', '')} "
            f"{job_or_candidate.get('seniority', '')}"
        )

    else:
        skills = " ".join(job_or_candidate.get("skills", []))
        languages = " ".join(job_or_candidate.get("languages", []))
        location = (
            f"{job_or_candidate['location']['city']}, "
            f"{job_or_candidate['location']['country']}"
        )

        return (
            f"{job_or_candidate['title']} {job_or_candidate.get('summary', '')} "
            f"{skills} {languages} {location}"
        )


embedding_service = EmbeddingService(
    model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
)
