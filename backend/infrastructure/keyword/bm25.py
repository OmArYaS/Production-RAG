"""Keyword index adapters (sparse lexical search via BM25).

Prefers ``rank_bm25``; falls back to a pure-python BM25Okapi implementation so
the app starts and tests run fully offline. Fully offline either way.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Set, Tuple

from domain.ports import ChunkRecord, KeywordIndex

from .base import BaseKeywordIndex


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= 2]


class BM25OkapiScorer:
    """Self-contained BM25Okapi implementation (fallback when rank_bm25 missing)."""

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        n = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / n if n else 0.0
        df: Dict[str, int] = {}
        for doc in corpus:
            for token in set(doc):
                df[token] = df.get(token, 0) + 1
        self.idf: Dict[str, float] = {}
        for token, doc_freq in df.items():
            self.idf[token] = math.log((n - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

    def scores(self, query_tokens: List[str]) -> Dict[int, float]:
        n = len(self.corpus)
        out: Dict[int, float] = {}
        for idx, doc in enumerate(self.corpus):
            dl = len(doc)
            if dl == 0:
                continue
            freq: Dict[str, int] = {}
            for t in doc:
                freq[t] = freq.get(t, 0) + 1
            score = 0.0
            for q in set(query_tokens):
                if q not in self.idf:
                    continue
                tf = freq.get(q, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                score += self.idf[q] * (tf * (self.k1 + 1)) / denom
            out[idx] = score
        return out


def _rank_bm25_available() -> bool:
    try:
        import rank_bm25  # type: ignore
        return True
    except Exception:
        return False


class BM25KeywordAdapter(BaseKeywordIndex):
    """Sparse keyword index using BM25 over indexed chunk texts."""

    def __init__(self) -> None:
        self._use_rank_bm25 = _rank_bm25_available()
        self._model = None
        self._fallback: BM25OkapiScorer | None = None
        super().__init__()

    def _rebuild(self) -> None:
        if self._use_rank_bm25:
            try:
                from rank_bm25 import BM25Okapi  # type: ignore

                corpus = [_tokenize(self._chunks[cid].text) for cid in sorted(self._chunks)]
                if not corpus:
                    self._model = None
                else:
                    self._model = BM25Okapi(corpus)
                self._fallback = None
                return
            except Exception:
                self._use_rank_bm25 = False
        corpus = [_tokenize(self._chunks[cid].text) for cid in sorted(self._chunks)]
        self._fallback = BM25OkapiScorer(corpus)

    def _score_all(self, query: str) -> Dict[str, float]:
        tokens = _tokenize(query)
        if not tokens or not self._chunks:
            return {}
        if self._use_rank_bm25 and self._model is not None:
            sorted_ids = sorted(self._chunks)
            scores = self._model.get_scores(tokens)
            return {cid: float(s) for cid, s in zip(sorted_ids, scores)}
        assert self._fallback is not None
        idx_scores = self._fallback.scores(tokens)
        sorted_ids = sorted(self._chunks)
        return {sorted_ids[i]: float(s) for i, s in idx_scores.items()}
