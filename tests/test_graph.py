"""Tests on tiny toy graphs where the answer can be computed by hand."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from ctds.graph.build import cooccurrence_graph, graph_from_edge_list
from ctds.graph.cluster import louvain, modularity, networkx_modularity, spectral_clustering
from ctds.graph.data import iterative_kcore
from ctds.graph.sparsify import disparity_filter, disparity_pvalues, top_k_neighbours
from ctds.graph.weights import jaccard, ppmi


def sym(edges, n):
    """Symmetric CSR from (i, j, w) triplets."""
    r, c, w = zip(*edges)
    M = sp.csr_matrix((w, (r, c)), shape=(n, n), dtype=float)
    return (M + M.T).tocsr()


# ------------------------------------------------------------------ PPMI / Jaccard

def test_ppmi_matches_hand_computation():
    # Triangle 0-1 (w=4), 0-2 (w=1), 1-2 (w=1).
    W = sym([(0, 1, 4), (0, 2, 1), (1, 2, 1)], 3)
    P = ppmi(W).toarray()
    total = 2 * (4 + 1 + 1)              # sum over ordered pairs
    r = np.array([5, 5, 2])              # row sums
    expected_01 = np.log(4 * total / (r[0] * r[1]))   # log(48/25) > 0
    expected_02 = np.log(1 * total / (r[0] * r[2]))   # log(12/10) > 0
    assert P[0, 1] == pytest.approx(expected_01, rel=1e-6)
    assert P[0, 2] == pytest.approx(expected_02, rel=1e-6)
    assert np.allclose(P, P.T)


def test_ppmi_drops_negative_pmi():
    # Star around a hub plus one strong pair: hub edges co-occur less than expected.
    W = sym([(0, 1, 1), (0, 2, 1), (0, 3, 1), (1, 2, 10)], 4)
    P = ppmi(W).toarray()
    total = 2 * 13
    r = W.sum(axis=1).A1
    assert P[0, 3] == pytest.approx(max(0, np.log(1 * total / (r[0] * r[3]))), rel=1e-6)
    assert P[0, 1] == 0  # log(26 / (3 * 11)) < 0, so the edge is removed
    assert P[1, 2] > 0


def test_jaccard_on_reader_sets():
    # Users x books: book 0 read by users {0,1,2}, book 1 by {1,2}, book 2 by {2,3}.
    B = sp.csr_matrix(np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=np.float32))
    g = cooccurrence_graph(B, user_weighting="none", min_shared=1)
    J = jaccard(g.weights, g.strength).toarray()
    assert J[0, 1] == pytest.approx(2 / 3)
    assert J[0, 2] == pytest.approx(1 / 4)
    assert J[1, 2] == pytest.approx(1 / 3)


def test_cooccurrence_matches_dense_with_user_weights_and_threshold():
    rng = np.random.default_rng(0)
    B = (rng.random((40, 15)) < 0.3).astype(np.float32)
    g = cooccurrence_graph(sp.csr_matrix(B), user_weighting="log", min_shared=2, block_size=4)
    w = 1 / np.log1p(np.maximum(B.sum(1), 1))
    C, S = (B * w[:, None]).T @ B, B.T @ B
    np.fill_diagonal(C, 0)
    C[S < 2] = 0
    assert np.allclose(g.weights.toarray(), C, atol=1e-5)


# ------------------------------------------------------------------ disparity filter

def test_disparity_pvalue_formula():
    # Node 0 has edges of weight 8, 1, 1 -> p = 0.8, 0.1, 0.1 and k = 3.
    W = sym([(0, 1, 8), (0, 2, 1), (0, 3, 1)], 4)
    A = disparity_pvalues(W).toarray()
    assert A[0, 1] == pytest.approx((1 - 0.8) ** 2)
    assert A[0, 2] == pytest.approx((1 - 0.1) ** 2)
    assert A[1, 0] == pytest.approx(1.0)  # degree-1 node: (1-1)^0 = 1


def test_disparity_filter_keeps_dominant_edge_only():
    # Node 0 has one dominant edge among ten. Leaves have degree 1, so only node 0's view counts.
    edges = [(0, 1, 100)] + [(0, j, 1) for j in range(2, 11)]
    W = sym(edges, 11)
    kept = disparity_filter(W, alpha=0.05).toarray()
    # alpha_01 = (1 - 100/109)^9 ~ 1e-10 -> kept; alpha_0j = (1 - 1/109)^9 ~ 0.92 -> dropped.
    assert kept[0, 1] == 100 and kept[1, 0] == 100
    assert (kept[0, 2:] == 0).all()
    assert np.allclose(kept, kept.T)


def test_disparity_filter_or_vs_and():
    # Edge 0-1 is significant for node 0 (heavy share) but not for node 1 (one of many equal edges).
    edges = [(0, 1, 50), (0, 2, 1), (0, 3, 1), (0, 4, 1)] + [(1, j, 50) for j in range(5, 15)]
    W = sym(edges, 15)
    assert disparity_filter(W, 0.05, rule="or")[0, 1] > 0
    assert disparity_filter(W, 0.05, rule="and")[0, 1] == 0


def test_top_k_neighbours():
    W = sym([(0, 1, 5), (0, 2, 4), (0, 3, 1), (2, 3, 2)], 4)
    K = top_k_neighbours(W, k=1).toarray()
    # Top-1 of 0 is 1, of 1 is 0, of 2 is 0, of 3 is 2 -> edges {0-1, 0-2, 2-3}; 0-3 dropped.
    assert K[0, 1] == 5 and K[0, 2] == 4 and K[2, 3] == 2 and K[0, 3] == 0


# ------------------------------------------------------------------ modularity / clustering

def test_modularity_two_triangles_by_hand():
    # Two triangles joined by one edge: m = 7, each community has 3 internal edges and degree 7.
    W = sym([(0, 1, 1), (1, 2, 1), (0, 2, 1), (3, 4, 1), (4, 5, 1), (3, 5, 1), (2, 3, 1)], 6)
    labels = np.array([0, 0, 0, 1, 1, 1])
    expected = 2 * (3 / 7 - (7 / 14) ** 2)  # = 5/14
    assert modularity(W, labels) == pytest.approx(expected)
    assert modularity(W, labels) == pytest.approx(networkx_modularity(W, labels))
    assert modularity(W, np.zeros(6, dtype=int)) == pytest.approx(0.0)


def test_modularity_weighted_matches_networkx():
    G = nx.karate_club_graph()
    W = nx.to_scipy_sparse_array(G, weight="weight", format="csr")
    labels = np.array([0 if G.nodes[v]["club"] == "Mr. Hi" else 1 for v in G])
    for gamma in [0.5, 1.0, 2.0]:
        assert modularity(W, labels, gamma) == pytest.approx(
            nx.community.modularity(G, [set(np.flatnonzero(labels == c)) for c in (0, 1)], weight="weight", resolution=gamma))


def test_louvain_and_spectral_find_planted_communities():
    G = nx.planted_partition_graph(4, 25, 0.5, 0.01, seed=1)
    W = nx.to_scipy_sparse_array(G, format="csr").astype(float)
    truth = np.repeat(np.arange(4), 25)
    from sklearn.metrics import normalized_mutual_info_score as nmi
    assert nmi(truth, louvain(W, seed=0)) > 0.9
    assert nmi(truth, louvain(W, seed=0, backend="networkx")) > 0.9
    assert nmi(truth, spectral_clustering(W, k=4, seed=0)) > 0.9


# ------------------------------------------------------------------ data helpers

def test_iterative_kcore_cascades():
    # c has 1 book -> dropped; then z has 1 reader -> dropped; then b has 1 book -> dropped.
    # a and d (3 shared books each) survive.
    df = pd.DataFrame({"user_id": list("aaadddbbc"), "item_id": list("xywxywxzz")})
    core, log = iterative_kcore(df, min_user=2, min_book=2)
    assert set(core.user_id) == {"a", "d"}
    assert set(core.item_id) == {"x", "y", "w"}
    assert len(core) == 6
    assert log.iloc[-1]["reviews"] == 6
    assert log["reviews_removed"].sum() == 3


def test_edge_list_adapter_maps_and_merges():
    edges = pd.DataFrame({"book_a": ["A", "B", "A", "Z", "A2"], "book_b": ["B", "A", "C", "A", "C"], "weight": [0.5, 0.7, 0.2, 1.0, 0.9]})
    G, rep = graph_from_edge_list(edges, np.array(["A", "B", "C"]), canonical_map=pd.Series({"A2": "A"}))
    D = G.toarray()
    assert D[0, 1] == pytest.approx(0.7)      # A-B listed twice -> max
    assert D[0, 2] == pytest.approx(0.9)      # A2 mapped to A -> max(0.2, 0.9)
    assert rep["unknown_book_edges"] == 1     # Z is not in our index
    assert np.allclose(D, D.T)


# ------------------------------------------------------------------ recommender metrics

def test_topk_metrics_by_hand():
    from ctds.graph.recommend import _metrics_for
    top = np.array([[3, 1, 2], [0, 1, 2], [4, 5, 6]])
    # user 0: relevant {1, 9} -> hit at rank 2; user 1: relevant {0, 1, 2, 3} -> 3 hits; user 2: none relevant.
    rel = sp.csr_matrix((np.ones(6), ([0, 0, 1, 1, 1, 1], [1, 9, 0, 1, 2, 3])), shape=(3, 10))
    m = _metrics_for(top, rel, k=3)
    d = 1 / np.log2(np.arange(2, 5))
    assert m["hit"][0] == 1 and m["hit"][1] == 1
    assert np.isnan(m["hit"][2])
    assert m["recall"][0] == pytest.approx(0.5) and m["recall"][1] == pytest.approx(0.75)
    assert m["ndcg"][0] == pytest.approx(d[1] / (d[0] + d[1]))
    assert m["ndcg"][1] == pytest.approx(1.0)


def test_community_recommender_prefers_users_community():
    from ctds.graph.recommend import CommunityRecommender
    # Two communities {0,1,2} and {3,4,5}; book 2 is the most central of community 0.
    W = sym([(0, 2, 5), (1, 2, 5), (0, 1, 1), (3, 4, 1), (4, 5, 1), (3, 5, 1)], 6)
    rec = CommunityRecommender(np.array([0, 0, 0, 1, 1, 1]), W, popularity=np.ones(6))
    s = rec.scores(sp.csr_matrix(np.array([[1, 0, 0, 0, 0, 0]], dtype=np.float32)))[0]
    assert s[2] > s[1] > s[3]
