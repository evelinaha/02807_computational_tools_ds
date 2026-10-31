"""Graph sparsification (implemented by us): the disparity-filter backbone and a top-k baseline.

Co-occurrence graphs are very dense (millions of edges, most of them weak), which slows down
clustering and blurs communities. Both methods keep a symmetric subgraph on the same nodes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def disparity_pvalues(W: sp.csr_matrix) -> sp.csr_matrix:
    """Disparity-filter p-value of every edge, seen from its row node (Serrano, Boguna &
    Vespignani 2009, PNAS 106:6483).

    Null model: node i's strength s_i is split over its k_i edges uniformly at random. Under
    that null the normalised weight p_ij = w_ij / s_i of one edge has density
    (k_i - 1)(1 - x)^(k_i - 2), so the probability of a weight at least as large is
        alpha_ij = (1 - p_ij)^(k_i - 1).
    A small alpha_ij means the edge carries a significantly large share of i's strength.

    Returns a matrix with the same pattern as ``W`` where entry (i, j) is alpha_ij for node i
    (not symmetric: alpha_ij != alpha_ji in general). Nodes with a single edge get alpha = 1
    (the formula's limit k=1); their edge survives only if it is significant for the other end.
    """
    W = W.tocsr()
    k = np.diff(W.indptr)
    s = np.asarray(W.sum(axis=1)).ravel()
    row = np.repeat(np.arange(W.shape[0]), k)
    p = W.data / s[row]
    # Clip so that extremely significant edges (alpha underflows to 0) are not mistaken for
    # absent entries by scipy's sparse operations.
    alpha = np.maximum(np.power(1.0 - p, k[row] - 1), 1e-300)
    return sp.csr_matrix((alpha, W.indices.copy(), W.indptr.copy()), shape=W.shape)


def disparity_filter(W: sp.csr_matrix, alpha: float = 0.05, rule: str = "or") -> sp.csr_matrix:
    """Keep the edges that are significant at level ``alpha`` for at least one endpoint
    (``rule="or"``, as in Serrano et al.) or for both (``rule="and"``, a stricter backbone).

    Unlike a global weight threshold, this keeps the locally important edges of weak nodes
    too, so long-tail books keep their strongest links instead of disappearing.
    """
    A = disparity_pvalues(W)
    sig = A.copy()
    sig.data = (A.data < alpha).astype(np.int8)
    sig.eliminate_zeros()
    sigT = sig.T.tocsr()
    keep = (sig + sigT) if rule == "or" else sig.multiply(sigT)
    keep.data[:] = 1
    out = W.multiply(keep).tocsr()
    out.eliminate_zeros()
    return out


def disparity_alpha_for_edges(W: sp.csr_matrix, n_edges: int) -> float:
    """Significance level at which the ("or"-rule) disparity backbone keeps ~``n_edges`` edges.

    The disparity p-values depend strongly on the weighting: PPMI weights are nearly uniform
    around each node, so at alpha = 0.05 almost no PPMI edge is significant while raw counts
    keep many. To compare weightings and sparsifiers fairly we therefore also match the edge
    budget (e.g. to the top-k baseline) instead of fixing alpha.
    """
    A = disparity_pvalues(W)
    edge_p = sp.triu(A.minimum(A.T), k=1).tocoo().data  # an edge survives if min(alpha_ij, alpha_ji) < alpha
    if n_edges >= len(edge_p):
        return 1.0 + 1e-9
    return float(np.partition(edge_p, n_edges)[n_edges])


def top_k_neighbours(W: sp.csr_matrix, k: int = 10) -> sp.csr_matrix:
    """Baseline: keep each node's ``k`` heaviest edges; an edge stays if either endpoint keeps it."""
    W = W.tocsr()
    n = W.shape[0]
    deg = np.diff(W.indptr)
    row = np.repeat(np.arange(n), deg)
    # Sort entries by (row, -weight) and keep the first k of every row.
    order = np.lexsort((-W.data, row))
    rank = np.arange(len(order)) - np.repeat(W.indptr[:-1], deg)
    chosen = order[rank < k]
    mask = sp.csr_matrix((np.ones(len(chosen), dtype=np.int8), (row[chosen], W.indices[chosen])), shape=W.shape)
    keep = mask + mask.T
    keep.data[:] = 1
    out = W.multiply(keep).tocsr()
    out.eliminate_zeros()
    return out


def giant_component(W: sp.csr_matrix) -> np.ndarray:
    """Indices of the nodes in the largest connected component."""
    _, labels = connected_components(W, directed=False)
    return np.flatnonzero(labels == np.bincount(labels).argmax())


def backbone_stats(W_full: sp.csr_matrix, W_kept: sp.csr_matrix, name: str = "") -> dict:
    """Edges/nodes/weight kept by a sparsifier and the size of the giant component."""
    n = W_full.shape[0]
    nodes_full = int((np.diff(W_full.tocsr().indptr) > 0).sum())
    nodes_kept = int((np.diff(W_kept.tocsr().indptr) > 0).sum())
    gc = len(giant_component(W_kept))
    return {
        "graph": name,
        "nodes_total": n,
        "nodes_with_edges": nodes_kept,
        "nodes_kept_pct": round(100 * nodes_kept / max(nodes_full, 1), 1),
        "edges": W_kept.nnz // 2,
        "edges_kept_pct": round(100 * W_kept.nnz / max(W_full.nnz, 1), 2),
        "weight_kept_pct": round(100 * W_kept.sum() / max(W_full.sum(), 1e-12), 1),
        "giant_component": gc,
        "giant_component_pct": round(100 * gc / n, 1),
        "mean_degree": round(W_kept.nnz / max(nodes_kept, 1), 1),
    }


def alpha_sweep(W: sp.csr_matrix, alphas: list[float], name: str = "") -> pd.DataFrame:
    """Backbone statistics for several significance levels (to choose alpha)."""
    rows = []
    for a in alphas:
        stats = backbone_stats(W, disparity_filter(W, a), name)
        rows.append({"alpha": a, **stats})
    return pd.DataFrame(rows)
