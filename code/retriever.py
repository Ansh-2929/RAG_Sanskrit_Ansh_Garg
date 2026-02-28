"""
Retrieval module for the Sanskrit RAG pipeline.

Provides two retrieval methods:
- a TF-IDF retriever, and
- a hybrid retriever that adds a simple character n-gram similarity bonus.

Both approaches work with Devanagari text as well as IAST transliteration.
"""

from __future__ import annotations

import math
import re
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from document_loader import DocumentChunk


# Tokenizer that handles Devanagari Unicode characters


def unicode_tokenizer(text: str) -> List[str]:
    
    # Keep Devanagari ranges and ASCII word chars together
    tokens = re.findall(r"[\u0900-\u097F]+|[a-zA-Z0-9]+", text)
    return [t.lower() for t in tokens if len(t) >= 1]


def unicode_tokenizer_str(text: str) -> str:
    
    return " ".join(unicode_tokenizer(text))


# TF-IDF Retriever


class TFIDFRetriever:
    

    def __init__(
        self,
        max_features: int = 20_000,
        ngram_range: Tuple[int, int] = (1, 2),
    ):
        self.vectorizer = TfidfVectorizer(
            tokenizer=unicode_tokenizer,
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,       # apply log(1 + tf)
            min_df=1,
            token_pattern=None,      # disabled: using custom tokenizer
        )
        self.chunks: List[DocumentChunk] = []
        self.index: Optional[np.ndarray] = None   # shape (n_chunks, n_features)


    # Index management
   

    def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Build the TF-IDF matrix for the given chunks."""
        self.chunks = chunks
        corpus = [c.text for c in chunks]
        self.index = self.vectorizer.fit_transform(corpus)
        print(f"[TFIDFRetriever] Indexed {len(chunks)} chunks, "
              f"vocab size = {len(self.vectorizer.vocabulary_)}")

    def save(self, path: str) -> None:
        """Persist the index and vectorizer to disk."""
        data = {
            "chunks": self.chunks,
            "index": self.index,
            "vectorizer": self.vectorizer,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[TFIDFRetriever] Index saved to {path}")

    def load(self, path: str) -> None:
        """Load a previously saved index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.index = data["index"]
        self.vectorizer = data["vectorizer"]
        print(f"[TFIDFRetriever] Loaded index with {len(self.chunks)} chunks from {path}")

   
    # Retrieval


    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Tuple[DocumentChunk, float]]:
       
        if self.index is None:
            raise RuntimeError("Index is empty. Call index_chunks() or load() first.")

        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.index).flatten()

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= score_threshold:
                results.append((self.chunks[idx], score))
        return results



# Hybrid Retriever (TF-IDF + character n-gram overlap bonus)


class HybridRetriever(TFIDFRetriever):
   

    def __init__(self, alpha: float = 0.8, ngram_size: int = 3, **kwargs):
       
        super().__init__(**kwargs)
        self.alpha = alpha
        self.ngram_size = ngram_size

    def _char_ngrams(self, text: str) -> set:
        text = text.lower()
        return {text[i:i+self.ngram_size] for i in range(len(text) - self.ngram_size + 1)}

    def _ngram_overlap(self, query: str, chunk_text: str) -> float:
       
        qn = self._char_ngrams(query)
        cn = self._char_ngrams(chunk_text)
        if not qn or not cn:
            return 0.0
        return len(qn & cn) / len(qn | cn)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Tuple[DocumentChunk, float]]:
        if self.index is None:
            raise RuntimeError("Index is empty. Call index_chunks() or load() first.")

        q_vec = self.vectorizer.transform([query])
        tfidf_scores = cosine_similarity(q_vec, self.index).flatten()

        # Combine TF-IDF with n-gram overlap
        hybrid_scores = np.array([
            self.alpha * tfidf_scores[i] +
            (1 - self.alpha) * self._ngram_overlap(query, self.chunks[i].text)
            for i in range(len(self.chunks))
        ])

        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(hybrid_scores[idx])
            if score >= score_threshold:
                results.append((self.chunks[idx], score))
        return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from document_loader import SanskritDocumentLoader

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "../data"
    loader = SanskritDocumentLoader(chunk_size=300, chunk_overlap=50)
    chunks = loader.load_directory(data_dir)

    retriever = HybridRetriever()
    retriever.index_chunks(chunks)

    test_query = "कालीदासः"
    print(f"\nQuery: {test_query}")
    results = retriever.retrieve(test_query, top_k=3)
    for chunk, score in results:
        print(f"\n  Score: {score:.4f} | {chunk.chunk_id}")
        print(f"  {chunk.text[:150]}...")
