"""Edge weighting schemes for co-occurrence graphs (implemented by us).

All functions take and return a symmetric CSR matrix with an empty diagonal and keep the
sparsity pattern (PPMI may drop edges whose PMI is <= 0).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _upper_coo(W: sp.csr_matrix) -> sp.coo_matrix:
    """Upper triangle (i < j) of a symmetric matrix as COO."""
    return sp.triu(W, k=1, format="coo")


def _from_upper(U: sp.coo_matrix, values: np.ndarray, n: int) -> sp.csr_matrix:
    keep = values > 0
    r, c, v = U.row[keep], U.col[keep], values[keep].astype(np.float32)
    return sp.csr_matrix((np.concatenate([v, v]), (np.concatenate([r, c]), np.concatenate([c, r]))), shape=(n, n))


def raw_counts(W: sp.csr_matrix) -> sp.csr_matrix:
    """Raw (user-weighted) co-occurrence count: the number of readers two books share."""
    return W.tocsr().copy()


def jaccard(W: sp.csr_matrix, strength: np.ndarray) -> sp.csr_matrix:
    """Jaccard similarity of the two books' reader sets.

    J(i, j) = |U_i & U_j| / |U_i | U_j| = C_ij / (s_i + s_j - C_ij), where ``strength`` s_i is
    the (user-weighted) number of readers of book i and C_ij the (user-weighted) co-count, so
    with user weights this is the weighted Jaccard index. Corrects for popularity: two
    bestsellers share many readers simply because both have many readers.
    """
    U = _upper_coo(W)
    s = np.asarray(strength, dtype=np.float64)
    denom = s[U.row] + s[U.col] - U.data
    vals = np.divide(U.data, denom, out=np.zeros_like(U.data, dtype=np.float64), where=denom > 0)
    return _from_upper(U, vals, W.shape[0])


def ppmi(W: sp.csr_matrix, smoothing: float = 1.0) -> sp.csr_matrix:
    """Positive pointwise mutual information of the co-occurrence matrix.

    With T = sum_ij C_ij and row sums r_i:
        PMI(i, j) = log( C_ij * T / (r_i * r_j) ),   PPMI = max(PMI, 0).
    PMI > 0 means the two books share readers more often than expected from how often each
    co-occurs with anything, so it down-weights hubs much more strongly than Jaccard.

    ``smoothing`` < 1 (e.g. 0.75, Levy, Goldberg & Dagan 2015) raises the marginal of rare
    books, which damps PMI's known bias towards very rare pairs. The ``min_shared`` threshold
    used when building the graph serves the same purpose.
    """
    W = W.tocsr()
    U = _upper_coo(W)
    r = np.asarray(W.sum(axis=1)).ravel().astype(np.float64)
    total = r.sum()
    if smoothing == 1.0:
        p_i = r / total
        p_j = p_i
    else:
        p_i = r / total
        rs = r ** smoothing
        p_j = rs / rs.sum()
    p_ij = U.data / total
    # Symmetrise the smoothed version by averaging both directions.
    pmi = 0.5 * (np.log(p_ij / (p_i[U.row] * p_j[U.col])) + np.log(p_ij / (p_i[U.col] * p_j[U.row])))
    return _from_upper(U, np.maximum(pmi, 0.0), W.shape[0])


def apply_weighting(W: sp.csr_matrix, scheme: str, strength: np.ndarray | None = None, **kwargs) -> sp.csr_matrix:
    """Dispatch by name: ``"raw"``, ``"jaccard"`` or ``"ppmi"``."""
    if scheme == "raw":
        return raw_counts(W)
    if scheme == "jaccard":
        if strength is None:
            raise ValueError("jaccard needs the per-book strength")
        return jaccard(W, strength)
    if scheme == "ppmi":
        return ppmi(W, **kwargs)
    raise ValueError(f"unknown weighting {scheme!r}")
