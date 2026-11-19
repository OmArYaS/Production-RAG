"""Tests for Reciprocal Rank Fusion."""

import math

from application.rrf import RRF_K, rrf_fuse


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-9)


def test_union_of_dense_and_sparse():
    dense = [("a", 10.0), ("b", 5.0)]
    sparse = [("b", 9.0), ("c", 1.0)]
    res = rrf_fuse(dense, sparse, top_k=10)
    ids = [r[0] for r in res]
    assert set(ids) == {"a", "b", "c"}
    scores = dict(res)
    # dense rank: a=1, b=2 ; sparse rank: b=1, c=2
    assert close(scores["a"], 1 / (RRF_K + 1))
    assert close(scores["b"], 1 / (RRF_K + 2) + 1 / (RRF_K + 1))
    assert close(scores["c"], 1 / (RRF_K + 2))
    # b is best (rank 1 on both lists)
    assert res[0][0] == "b"


def test_dense_only_and_sparse_only():
    dense = [("x", 1.0), ("y", 0.5)]
    assert [r[0] for r in rrf_fuse(dense, [], 10)] == ["x", "y"]
    assert [r[0] for r in rrf_fuse([], dense, 10)] == ["x", "y"]


def test_zero_hits():
    assert rrf_fuse([], [], 5) == []


def test_top_k_truncates():
    dense = [(f"c{i}", float(i)) for i in range(20)]
    assert len(rrf_fuse(dense, [], 3)) == 3


def test_score_is_monotonic():
    dense = [("a", 1.0), ("b", 0.9)]
    res = rrf_fuse(dense, [], 10)
    assert res[0][1] >= res[1][1]
