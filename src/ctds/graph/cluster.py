"""Community detection: Louvain (library) and spectral clustering + modularity (ours)."""

from __future__ import annotations

import random
from itertools import combinations

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# ---------------------------------------------------------------- modularity (ours)

def modularity(W: sp.csr_matrix, labels: np.ndarray, resolution: float = 1.0) -> float:
    """Newman-Girvan modularity of a partition of a weighted undirected graph.

    Q = sum_c [ in_c / (2m) - gamma * (d_c / (2m))^2 ]
    where 2m = sum_ij A_ij, in_c = sum of A_ij over ordered pairs inside community c (each
    undirected edge counted twice, self-loops once as stored) and d_c the total strength of c.
    Equivalent to (1/2m) sum_ij (A_ij - gamma k_i k_j / 2m) delta(c_i, c_j).
    Nodes labelled -1 are treated as singletons.
    """
    W = W.tocsr()
    labels = np.asarray(labels).copy()
    singles = labels < 0
    labels[singles] = labels.max() + 1 + np.arange(singles.sum())
    _, labels = np.unique(labels, return_inverse=True)
    two_m = W.sum()
    if two_m == 0:
        return 0.0
    coo = W.tocoo()
    inside = labels[coo.row] == labels[coo.col]
    in_c = np.bincount(labels[coo.row[inside]], weights=coo.data[inside], minlength=labels.max() + 1)
    d_c = np.bincount(labels, weights=np.asarray(W.sum(axis=1)).ravel(), minlength=labels.max() + 1)
    return float(np.sum(in_c / two_m - resolution * (d_c / two_m) ** 2))


def networkx_modularity(W: sp.csr_matrix, labels: np.ndarray, resolution: float = 1.0) -> float:
    """networkx's modularity, used only as a sanity check for :func:`modularity`."""
    G = nx.from_scipy_sparse_array(W)
    comms: dict[int, set] = {}
    for node, c in enumerate(labels):
        comms.setdefault(int(c), set()).add(node)
    return nx.community.modularity(G, comms.values(), weight="weight", resolution=resolution)


# ---------------------------------------------------------------- Louvain (library)

def to_igraph(W: sp.csr_matrix) -> ig.Graph:
    """Undirected weighted igraph graph from a symmetric matrix."""
    U = sp.triu(W, k=1).tocoo()
    g = ig.Graph(n=W.shape[0], edges=np.column_stack([U.row, U.col]).tolist(), directed=False)
    g.es["weight"] = U.data.astype(float).tolist()
    return g


def louvain(W: sp.csr_matrix | ig.Graph, resolution: float = 1.0, seed: int = 0, backend: str = "igraph") -> np.ndarray:
    """Louvain community labels (one int per node; isolated nodes become singletons).

    ``backend="igraph"`` (C implementation, ~100x faster, default for the full graph) or
    ``"networkx"`` (``nx.community.louvain_communities``, used on the sample to check that both
    give similar partitions). The seed fixes the random node order.
    """
    if backend == "igraph":
        g = W if isinstance(W, ig.Graph) else to_igraph(W)
        random.seed(seed)
        ig.set_random_number_generator(random)
        part = g.community_multilevel(weights="weight", resolution=resolution)
        return np.asarray(part.membership)
    if backend == "networkx":
        G = nx.from_scipy_sparse_array(W)
        comms = nx.community.louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
        labels = np.empty(W.shape[0], dtype=np.int64)
        for c, nodes in enumerate(comms):
            labels[list(nodes)] = c
        return labels
    raise ValueError(backend)


def resolution_sweep(W: sp.csr_matrix, resolutions: list[float], seeds: list[int]) -> tuple[pd.DataFrame, dict]:
    """Run Louvain for every (resolution, seed); report modularity (ours, at gamma=1 and at the
    run's gamma), number and size of communities, and stability across seeds (mean pairwise NMI
    and ARI between the seeds' partitions).

    Returns the summary table and ``{(resolution, seed): labels}``.
    """
    g = to_igraph(W)
    has_edge = np.diff(W.tocsr().indptr) > 0
    partitions, rows = {}, []
    for res in resolutions:
        labs = []
        for seed in seeds:
            lab = louvain(g, res, seed)
            partitions[(res, seed)] = lab
            labs.append(lab)
            sizes = np.bincount(lab[has_edge])
            sizes = sizes[sizes > 0]
            rows.append({"resolution": res, "seed": seed,
                         "modularity": modularity(W, lab, 1.0),
                         "modularity_at_resolution": modularity(W, lab, res),
                         "n_communities": len(sizes),
                         "n_communities_ge10": int((sizes >= 10).sum()),
                         "largest_pct": round(100 * sizes.max() / has_edge.sum(), 1),
                         "median_size": float(np.median(sizes))})
        pairs = list(combinations(range(len(labs)), 2))
        nmi = np.mean([normalized_mutual_info_score(labs[a][has_edge], labs[b][has_edge]) for a, b in pairs]) if pairs else np.nan
        ari = np.mean([adjusted_rand_score(labs[a][has_edge], labs[b][has_edge]) for a, b in pairs]) if pairs else np.nan
        for r in rows[-len(seeds):]:
            r["seed_stability_nmi"], r["seed_stability_ari"] = nmi, ari
    return pd.DataFrame(rows), partitions


def summarise_sweep(sweep: pd.DataFrame) -> pd.DataFrame:
    """Mean over seeds per resolution."""
    return sweep.drop(columns="seed").groupby("resolution").mean().round(4).reset_index()


# ---------------------------------------------------------------- spectral clustering (ours)

def normalized_laplacian(W: sp.csr_matrix) -> tuple[sp.csr_matrix, np.ndarray]:
    """L_sym = I - D^-1/2 W D^-1/2, and the vector D^-1/2 (zero for isolated nodes)."""
    d = np.asarray(W.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(d)
    inv_sqrt[d > 0] = 1.0 / np.sqrt(d[d > 0])
    Dm = sp.diags(inv_sqrt)
    L = sp.identity(W.shape[0], format="csr") - Dm @ W @ Dm
    return L.tocsr(), inv_sqrt


def spectral_embedding(W: sp.csr_matrix, k: int, seed: int = 0, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """The k eigenvectors of L_sym with the smallest eigenvalues (and those eigenvalues).

    Uses that the smallest eigenvalues of L_sym = I - M are 1 minus the *largest* eigenvalues
    of M = D^-1/2 W D^-1/2, which ARPACK (scipy ``eigsh``) finds much faster than the
    smallest eigenvalues of L directly. Run on a connected graph (the giant component),
    otherwise every component adds an eigenvalue 0.
    """
    L, _ = normalized_laplacian(W)
    M = sp.identity(W.shape[0], format="csr") - L
    rng = np.random.default_rng(seed)
    vals, vecs = eigsh(M, k=k, which="LA", tol=tol, v0=rng.standard_normal(W.shape[0]))
    order = np.argsort(-vals)
    return 1.0 - vals[order], vecs[:, order]


def spectral_clustering(W: sp.csr_matrix, k: int, seed: int = 0, embedding: np.ndarray | None = None) -> np.ndarray:
    """Normalised spectral clustering (Ng, Jordan & Weiss 2002): embed every node with the first
    k eigenvectors of L_sym, normalise each row to unit length, then k-means (k clusters).

    Pass ``embedding`` (from :func:`spectral_embedding` with >= k vectors) to reuse it for
    several k.
    """
    if embedding is None:
        _, embedding = spectral_embedding(W, k, seed)
    X = embedding[:, :k].copy()
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = np.divide(X, norms, out=np.zeros_like(X), where=norms > 0)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
