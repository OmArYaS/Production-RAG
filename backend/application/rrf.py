"""Reciprocal Rank Fusion (RRF) for hybrid retrieval.

Depends only on ``domain/``. Combines dense semantic ranks and sparse BM25
ranks into a single ranked list.
"""

from typing import Dict, List, Tuple

# RRF constant: higher = less influence from top ranks. 60 is a common default.
RRF_K = 60


def rrf_fuse(
    dense: List[Tuple[str, float]],
    sparse: List[Tuple[str, float]],
    top_k: int,
) -> List[Tuple[str, float]]:
    """Fuse two ranked lists via Reciprocal Rank Fusion.

    Args:
        dense: [(chunk_id, score)] ranked by dense relevance (best first).
        sparse: [(chunk_id, score)] ranked by BM25 relevance (best first).
        top_k: maximum number of fused results to return.

    Returns:
        [(chunk_id, rrf_score)] sorted by rrf_score desc, truncated to top_k.

    Handles chunks present in only one of the lists (the missing side
    contributes 0 to the RRF sum) and empty lists gracefully.
    """
    # Rank maps: chunk_id -> 1-based rank position.
    dense_rank: Dict[str, int] = {cid: i + 1 for i, (cid, _score) in enumerate(dense)}
    sparse_rank: Dict[str, int] = {cid: i + 1 for i, (cid, _score) in enumerate(sparse)}

    all_ids = set(dense_rank) | set(sparse_rank)

    fused: List[Tuple[str, float]] = []
    for cid in all_ids:
        score = 0.0
        if cid in dense_rank:
            score += 1.0 / (RRF_K + dense_rank[cid])
        if cid in sparse_rank:
            score += 1.0 / (RRF_K + sparse_rank[cid])
        fused.append((cid, score))

    fused.sort(key=lambda item: item[1], reverse=True)
    return fused[:top_k]
