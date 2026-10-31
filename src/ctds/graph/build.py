"""Book-book graphs from sparse products of the user x book matrix, plus an adapter for
external edge lists (e.g. the LSH/SBERT review-text graph from a teammate).

All graphs are returned as a symmetric ``scipy.sparse.csr_matrix`` over the same book
index (the columns of :class:`~ctds.graph.matrix.Bipartite`) with an empty diagonal, so
every graph goes through the same weighting / sparsification / clustering code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class CoGraph:
    """Co-occurrence graph of books.

    Attributes:
        weights: symmetric matrix of (user-weighted) co-occurrence counts, zero diagonal.
        shared: symmetric matrix of the raw number of shared users for the same edges.
        strength: per book, the (user-weighted) number of users that read it - the
            "degree" in the bipartite graph, needed by Jaccard.
        name: label used in tables.
    """

    weights: sp.csr_matrix
    shared: sp.csr_matrix
    strength: np.ndarray
    name: str = ""

    @property
    def n_edges(self) -> int:
        return self.weights.nnz // 2


def user_weights(user_degree: np.ndarray, scheme: str = "log") -> np.ndarray:
    """Weight of each user's contribution to co-occurrence counts.

    ``"log"``: w_u = 1 / log(1 + d_u). One user who read d books adds a pair to d(d-1)/2
    book pairs, so a handful of users with hundreds or thousands of books would otherwise
    dominate the edge weights (here the 34 training users with >500 books create 17.5%
    of all co-occurrence pairs). Their co-reads are also less informative: someone who
    reviews everything says little about two books being similar. The log keeps them in
    the graph but makes a user with 1,000 books count ~2.6x less than one with 14 books -
    the same idea as inverse document frequency / the Adamic-Adar index. ``"none"``: w_u = 1.
    """
    user_degree = np.asarray(user_degree, dtype=np.float64)
    if scheme == "none":
        return np.ones_like(user_degree, dtype=np.float32)
    if scheme == "log":
        return (1.0 / np.log1p(np.maximum(user_degree, 1))).astype(np.float32)
    raise ValueError(f"unknown user weighting scheme {scheme!r}")


def cap_user_degree(B: sp.csr_matrix, max_books: int | None, seed: int = 42) -> sp.csr_matrix:
    """Randomly keep at most ``max_books`` books per user (fixed seed). ``None`` = no cap.

    This bounds the cost of B^T B (a user with d books creates d^2 products). With the agreed
    k-core the heaviest training user has ~2k books and the full product fits easily in
    memory, so the default pipeline only down-weights (see :func:`user_weights`); the cap is
    there for larger thresholds or other data.
    """
    if max_books is None:
        return B
    B = B.tocsr(copy=True)
    rng = np.random.default_rng(seed)
    deg = np.diff(B.indptr)
    for u in np.flatnonzero(deg > max_books):
        start, end = B.indptr[u], B.indptr[u + 1]
        drop = rng.choice(np.arange(start, end), size=end - start - max_books, replace=False)
        B.data[drop] = 0
    B.eliminate_zeros()
    return B


def cooccurrence_graph(
    B: sp.csr_matrix,
    user_weighting: str = "log",
    min_shared: int = 2,
    max_user_books: int | None = None,
    block_size: int = 4096,
    seed: int = 42,
    name: str = "",
) -> CoGraph:
    """Book-book co-occurrence graph C = B^T W B from a user x book 0/1 matrix ``B``.

    The product is computed in blocks of ``block_size`` books (rows of B^T), and each block
    is thresholded before it is kept, so memory stays bounded and no dense all-pairs matrix
    is ever formed.

    Args:
        B: user x book 0/1 matrix (``read`` for the co-reader graph, ``liked`` for co-liked).
        user_weighting: see :func:`user_weights`.
        min_shared: keep an edge only if at least this many users are shared (raw count).
            Pairs with one shared reader are most of the non-zeros and mostly noise.
        max_user_books: optional cap, see :func:`cap_user_degree`.
    """
    B = cap_user_degree(B.tocsr(), max_user_books, seed=seed)
    B.data[:] = 1.0
    w = user_weights(np.diff(B.indptr), user_weighting)
    Bw = sp.diags(w) @ B
    Bt = B.T.tocsr()
    n = B.shape[1]

    rows, cols, wvals, svals = [], [], [], []
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        shared = (Bt[start:stop] @ B).tocsr()
        weighted = (Bt[start:stop] @ Bw).tocsr()
        # Bw has the same pattern as B, so both products have the same pattern: after sorting
        # the indices their data arrays line up entry by entry.
        shared.sort_indices()
        weighted.sort_indices()
        assert np.array_equal(shared.indices, weighted.indices)
        row = np.repeat(np.arange(stop - start), np.diff(shared.indptr)) + start
        # Upper triangle only (j > i) and enough shared users.
        keep = (shared.indices > row) & (shared.data >= min_shared)
        rows.append(row[keep])
        cols.append(shared.indices[keep])
        svals.append(shared.data[keep].astype(np.float32))
        wvals.append(weighted.data[keep].astype(np.float32))
    r, c = np.concatenate(rows), np.concatenate(cols)
    W = _symmetric(r, c, np.concatenate(wvals), n)
    S = _symmetric(r, c, np.concatenate(svals), n)
    strength = np.asarray(Bw.sum(axis=0)).ravel()
    return CoGraph(weights=W, shared=S, strength=strength, name=name)


def _symmetric(r: np.ndarray, c: np.ndarray, v: np.ndarray, n: int) -> sp.csr_matrix:
    """Symmetric CSR matrix from upper-triangle triplets."""
    M = sp.csr_matrix((np.concatenate([v, v]), (np.concatenate([r, c]), np.concatenate([c, r]))), shape=(n, n))
    M.sum_duplicates()
    return M


def graph_from_edge_list(
    edges: pd.DataFrame,
    book_ids: np.ndarray,
    a_col: str = "book_a",
    b_col: str = "book_b",
    weight_col: str = "weight",
    canonical_map: pd.Series | None = None,
    agg: str = "max",
) -> tuple[sp.csr_matrix, dict]:
    """Turn an external weighted edge list into a graph on our book index.

    This is the entry point for the teammate's LSH/SBERT review-text similarity graph: any
    ``(book_a, book_b, weight)`` table can then be weighted, sparsified, clustered and used for
    recommendations exactly like the co-reader graphs.

    Args:
        edges: one row per edge; direction is ignored.
        book_ids: our column order (``Bipartite.book_ids``).
        canonical_map: optional ``asin -> canonical_id`` Series applied to both endpoints first.
        agg: how duplicate edges (e.g. after canonical merging, or listed both ways) are combined:
            ``"max"``, ``"sum"`` or ``"mean"``.

    Returns:
        The symmetric CSR graph and a small report (how many edges were dropped and why).
    """
    e = edges[[a_col, b_col, weight_col]].rename(columns={a_col: "a", b_col: "b", weight_col: "w"})
    if canonical_map is not None:
        e["a"] = e["a"].map(canonical_map).fillna(e["a"])
        e["b"] = e["b"].map(canonical_map).fillna(e["b"])
    index = pd.Series(np.arange(len(book_ids)), index=book_ids)
    ia, ib = e["a"].map(index), e["b"].map(index)
    known = ia.notna() & ib.notna()
    e = pd.DataFrame({"i": ia[known].astype(np.int64), "j": ib[known].astype(np.int64), "w": e.loc[known, "w"].astype(np.float32)})
    self_loops = int((e.i == e.j).sum())
    e = e[e.i != e.j]
    lo, hi = np.minimum(e.i, e.j), np.maximum(e.i, e.j)
    e = pd.DataFrame({"i": lo, "j": hi, "w": e.w}).groupby(["i", "j"], as_index=False)["w"].agg(agg)
    e = e[e.w > 0]
    G = _symmetric(e.i.to_numpy(), e.j.to_numpy(), e.w.to_numpy(np.float32), len(book_ids))
    report = {"input_edges": len(edges), "unknown_book_edges": int((~known).sum()),
              "self_loops": self_loops, "edges": int(G.nnz // 2)}
    return G, report
