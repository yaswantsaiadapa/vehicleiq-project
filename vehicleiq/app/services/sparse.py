"""
VehicleIQ — SparseRetrieverService (BM25)
===========================================
High-performance, pure-Python BM25Okapi sparse lexical retriever.
Indexes vehicle textual descriptions, visual attribute tags, and
movement keywords to provide exact-keyword matching for hybrid search.

Mathematical model:
    Score(D, Q) = sum_{q in Q} IDF(q) * (f(q, D) * (k1 + 1)) / (f(q, D) + k1 * (1 - b + b * (|D| / avgdl)))
    IDF(q) = ln((N - n(q) + 0.5) / (n(q) + 0.5) + 1.0)
    Default parameters: k1 = 1.5, b = 0.75
"""

import json
import logging
import math
import re
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Standard English stopwords to filter out non-discriminative noise
_STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "of", "for", "with",
    "by", "from", "up", "about", "into", "over", "after", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "it", "its", "they", "them", "their", "this", "that", "these", "those",
    "which", "who", "whom", "whose", "what", "there", "here",
}


def tokenize(text: str) -> list[str]:
    """Tokenize, lowercase, and filter stopwords from text."""
    if not text:
        return []
    words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


class BM25Index:
    """
    Inverted-index BM25Okapi lexical retriever.
    Thread-safe, lightweight, and supports serialization to JSON.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._lock = threading.Lock()

        # Inverted index: term -> {doc_id: term_frequency}
        self.inverted_index: dict[str, dict[str, int]] = defaultdict(dict)
        # Document lengths: doc_id -> token_count
        self.doc_lengths: dict[str, int] = {}
        # Document frequencies: term -> count of docs containing term
        self.doc_frequencies: dict[str, int] = defaultdict(int)
        # Raw text representation per document (for debugging / inspection)
        self.doc_texts: dict[str, str] = {}
        # Total documents count
        self.n_docs: int = 0
        # Average document length
        self.avgdl: float = 0.0

    def fit(self, documents: dict[str, str]) -> None:
        """
        Build the BM25 index from a dictionary of {doc_id: text}.
        """
        with self._lock:
            self.inverted_index.clear()
            self.doc_lengths.clear()
            self.doc_frequencies.clear()
            self.doc_texts.clear()

            total_length = 0
            for doc_id, text in documents.items():
                tokens = tokenize(text)
                doc_len = len(tokens)
                self.doc_lengths[doc_id] = doc_len
                self.doc_texts[doc_id] = text
                total_length += doc_len

                term_counts = Counter(tokens)
                for term, count in term_counts.items():
                    self.inverted_index[term][doc_id] = count
                    self.doc_frequencies[term] += 1

            self.n_docs = len(documents)
            self.avgdl = (total_length / self.n_docs) if self.n_docs > 0 else 0.0

        log.info(
            "BM25 index built: %d documents, %d unique terms, avgdl=%.2f",
            self.n_docs,
            len(self.inverted_index),
            self.avgdl,
        )

    def add_document(self, doc_id: str, text: str) -> None:
        """
        Incrementally add or update a single document in the index.
        Thread-safe.
        """
        with self._lock:
            tokens = tokenize(text)
            doc_len = len(tokens)

            # If document previously existed, decrement old frequencies
            if doc_id in self.doc_lengths:
                old_text = self.doc_texts.get(doc_id, "")
                for old_term in set(tokenize(old_text)):
                    if doc_id in self.inverted_index.get(old_term, {}):
                        del self.inverted_index[old_term][doc_id]
                        self.doc_frequencies[old_term] = max(0, self.doc_frequencies[old_term] - 1)

            self.doc_lengths[doc_id] = doc_len
            self.doc_texts[doc_id] = text

            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                self.inverted_index[term][doc_id] = count
                self.doc_frequencies[term] += 1

            self.n_docs = len(self.doc_lengths)
            total_len = sum(self.doc_lengths.values())
            self.avgdl = (total_len / self.n_docs) if self.n_docs > 0 else 0.0

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        """
        Score documents for the query using BM25Okapi.

        Returns
        -------
        list of dicts: [{"vehicle_id": id, "score": bm25_score, "rank": rank}]
        """
        query_terms = tokenize(query)
        if not query_terms or self.n_docs == 0:
            return []

        scores: dict[str, float] = defaultdict(float)

        for term in query_terms:
            if term not in self.inverted_index:
                continue

            # IDF computation with Robertson-Spärck Jones formula
            df = self.doc_frequencies[term]
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
            if idf <= 0:
                continue

            postings = self.inverted_index[term]
            for doc_id, tf in postings.items():
                doc_len = self.doc_lengths.get(doc_id, self.avgdl)
                # BM25 term frequency saturation
                tf_component = (tf * (self.k1 + 1.0)) / (
                    tf + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avgdl or 1.0)))
                )
                scores[doc_id] += idf * tf_component

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        results = []
        for rank, (doc_id, score) in enumerate(ranked[:top_k], start=1):
            results.append(
                {
                    "vehicle_id": doc_id,
                    "score": round(float(score), 4),
                    "rank": rank,
                }
            )
        return results

    def save(self, filepath: Path) -> None:
        """Persist index state to disk as JSON."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {
                "k1": self.k1,
                "b": self.b,
                "n_docs": self.n_docs,
                "avgdl": self.avgdl,
                "doc_lengths": self.doc_lengths,
                "doc_frequencies": dict(self.doc_frequencies),
                "doc_texts": self.doc_texts,
                "inverted_index": {
                    term: dict(postings) for term, postings in self.inverted_index.items()
                },
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f)
        log.info("BM25 index saved to %s", filepath)

    def load(self, filepath: Path) -> None:
        """Load index state from JSON file."""
        if not filepath.exists():
            log.warning("BM25 index file %s does not exist", filepath)
            return
        with self._lock:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.k1 = data.get("k1", 1.5)
            self.b = data.get("b", 0.75)
            self.n_docs = data.get("n_docs", 0)
            self.avgdl = data.get("avgdl", 0.0)
            self.doc_lengths = data.get("doc_lengths", {})
            self.doc_frequencies = defaultdict(int, data.get("doc_frequencies", {}))
            self.doc_texts = data.get("doc_texts", {})
            self.inverted_index = defaultdict(
                dict,
                {term: dict(postings) for term, postings in data.get("inverted_index", {}).items()},
            )
        log.info("BM25 index loaded from %s (%d documents)", filepath, self.n_docs)
