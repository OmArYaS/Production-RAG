"""Keyword index adapters (sparse lexical search)."""

from domain.ports import KeywordIndex

from .bm25 import BM25KeywordAdapter


def build_keyword_index() -> KeywordIndex:
    """Factory: build the sparse keyword index (BM25).

    Uses ``rank_bm25`` when installed, otherwise a pure-python BM25Okapi
    fallback — fully offline.
    """
    return BM25KeywordAdapter()
