"""Book-book content similarity from descriptions (Ana Rita's similarity method) as a graph.

The text preparation and the embedding model follow Ana Rita's notebook
(``CTFDS_project (5).ipynb``, section "Book Similarity") exactly:

- ``clean_text``: unescape HTML entities, strip HTML tags, collapse whitespace;
- SBERT ``all-MiniLM-L6-v2`` embeddings of the cleaned description, L2-normalised, so the dot
  product of two vectors is their cosine similarity;
- only books with a title and a non-empty description are compared (her ``recommend_books``
  filter); books without a description get no content edges.

Her notebook ranks the most similar books for one query book on a 5,000-book sample. Here the
same similarity is computed for every book in the graph-clustering catalogue and stored as a
sparse k-nearest-neighbour graph on our book index, so it goes through the same clustering,
recommendation and evaluation code as the co-reader graphs.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

MODEL_NAME = "all-MiniLM-L6-v2"


def clean_text(value) -> str:
    """Clean a description while preserving useful information (Ana Rita's function)."""
    if not isinstance(value, str):
        return ""
    value = html.unescape(value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def comparable_books(meta: pd.DataFrame) -> np.ndarray:
    """Boolean mask of books with a title and a non-empty cleaned description."""
    has_title = meta["title"].fillna("").astype(str).str.strip().ne("").to_numpy()
    has_desc = meta["description"].map(clean_text).str.len().gt(0).to_numpy()
    return has_title & has_desc


def description_embeddings(meta: pd.DataFrame, cache: Path | None = None, batch_size: int = 32,
                           device: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """SBERT embeddings of the cleaned descriptions of the comparable books.

    Args:
        meta: book metadata aligned with the graph's book index (row i = book i), with
            ``item_id``, ``title`` and ``description``.
        cache: optional ``.npz`` file; reused when it holds the same item ids.

    Returns:
        ``(nodes, E)``: graph indices of the embedded books and their float32 unit vectors.
    """
    ids = meta["item_id"].to_numpy().astype(str)
    nodes = np.flatnonzero(comparable_books(meta))
    if cache is not None and Path(cache).exists():
        z = np.load(cache, allow_pickle=False)
        if np.array_equal(z["item_id"], ids[nodes]):
            return nodes, z["E"]
    from sentence_transformers import SentenceTransformer  # optional dependency group "text"

    encoder = SentenceTransformer(MODEL_NAME, device=device)
    texts = meta["description"].iloc[nodes].map(clean_text).tolist()
    E = encoder.encode(texts, batch_size=batch_size, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    if cache is not None:
        np.savez(cache, item_id=ids[nodes], E=E)
    return nodes, E


def knn_similarity_graph(nodes: np.ndarray, E: np.ndarray, n_books: int, k: int = 10,
                         min_sim: float = 0.0, block_size: int = 2048) -> sp.csr_matrix:
    """Symmetric k-nearest-neighbour graph of cosine similarities.

    Every embedded book keeps its ``k`` most similar other books (exact search: one matrix
    product per block of rows, never the full n x n matrix); an edge stays if either endpoint
    chose it, as in :func:`~ctds.graph.sparsify.top_k_neighbours`. Edge weight = cosine
    similarity. The result lives on the full graph index (``n_books`` nodes); books without an
    embedding are isolated.
    """
    n = len(nodes)
    rows, cols, vals = [], [], []
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        S = E[start:stop] @ E.T
        S[np.arange(stop - start), np.arange(start, stop)] = -np.inf      # no self-similarity
        top = np.argpartition(-S, k, axis=1)[:, :k]
        sims = np.take_along_axis(S, top, axis=1)
        keep = sims > min_sim
        rows.append(np.repeat(np.arange(start, stop), k)[keep.ravel()])
        cols.append(top.ravel()[keep.ravel()])
        vals.append(sims.ravel()[keep.ravel()])
    r, c, v = nodes[np.concatenate(rows)], nodes[np.concatenate(cols)], np.concatenate(vals).astype(np.float32)
    A = sp.csr_matrix((v, (r, c)), shape=(n_books, n_books))
    return A.maximum(A.T).tocsr()


def to_edge_list(G: sp.csr_matrix, book_ids: np.ndarray) -> pd.DataFrame:
    """``book_a, book_b, weight`` table (each undirected edge once), the format that
    :func:`~ctds.graph.build.graph_from_edge_list` and the notebook's ``TEXT_GRAPH_PATH`` read."""
    U = sp.triu(G, k=1).tocoo()
    return pd.DataFrame({"book_a": book_ids[U.row], "book_b": book_ids[U.col], "weight": U.data})


def combine_graphs(A: sp.csr_matrix, B: sp.csr_matrix, beta: float) -> sp.csr_matrix:
    """Weighted union (1 - beta) * A / mean(A) + beta * B / mean(B).

    Each graph is first divided by its mean edge weight, because Jaccard weights (mostly
    below 0.1) and cosine similarities (0.3-0.9) live on different scales. ``beta = 0`` is
    ``A`` alone and ``beta = 1`` is ``B`` alone; an edge present in both graphs gets both
    contributions.
    """
    a = A / A.data.mean()
    b = B / B.data.mean()
    out = ((1 - beta) * a + beta * b).tocsr()
    out.eliminate_zeros()
    return out
