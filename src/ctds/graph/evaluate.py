"""Evaluation of communities: size distribution, NMI against genre labels, mixed-genre communities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def community_sizes(labels: np.ndarray, nodes: np.ndarray | None = None) -> pd.Series:
    """Size of every community (descending), optionally only counting the given nodes."""
    lab = labels if nodes is None else labels[nodes]
    return pd.Series(lab).value_counts()


def size_summary(labels: np.ndarray, nodes: np.ndarray | None = None) -> dict:
    """Headline numbers of the community size distribution."""
    s = community_sizes(labels, nodes)
    n = s.sum()
    return {"n_communities": len(s), "n_ge10": int((s >= 10).sum()), "n_ge100": int((s >= 100).sum()),
            "largest": int(s.iloc[0]), "largest_pct": round(100 * s.iloc[0] / n, 1),
            "median": float(s.median()), "pct_nodes_in_ge10": round(100 * s[s >= 10].sum() / n, 1),
            "singletons": int((s == 1).sum())}


def genre_nmi(labels: np.ndarray, meta: pd.DataFrame, nodes: np.ndarray | None = None, seed: int = 0) -> dict:
    """NMI and ARI between communities and the primary genre, on books that have a genre.

    ``meta`` must be aligned with the node index (row i = node i) and have ``primary_genre`` and
    ``n_genres``. Also reports NMI on single-genre books only (no tie-breaking needed) and a
    baseline where community labels are randomly permuted (expected NMI by chance for this
    size distribution), because NMI grows with the number of clusters.
    """
    mask = meta["primary_genre"].notna().to_numpy().copy()  # pandas copy-on-write arrays are read-only
    if nodes is not None:
        in_nodes = np.zeros(len(labels), dtype=bool)
        in_nodes[nodes] = True
        mask &= in_nodes
    lab, gen = labels[mask], meta["primary_genre"].to_numpy()[mask].astype(str)
    single = (meta["n_genres"].to_numpy()[mask] == 1)
    rng = np.random.default_rng(seed)
    return {
        "labelled_books": int(mask.sum()),
        "nmi": normalized_mutual_info_score(gen, lab),
        "ari": adjusted_rand_score(gen, lab),
        "nmi_single_genre": normalized_mutual_info_score(gen[single], lab[single]) if single.sum() > 1 else np.nan,
        "nmi_random_baseline": normalized_mutual_info_score(gen, rng.permutation(lab)),
    }


def community_genre_table(labels: np.ndarray, meta: pd.DataFrame, W: sp.csr_matrix | None = None,
                          min_labelled: int = 15, n_titles: int = 5) -> pd.DataFrame:
    """Genre make-up of every community with at least ``min_labelled`` genre-labelled books.

    Columns: size, labelled books, the two most common genres and their shares, normalised
    genre entropy (0 = one genre, 1 = uniform over all genres present in the data), and the
    most central titles (highest within-community weighted degree if ``W`` is given).
    A community is flagged ``mixed`` when its second genre still covers >= 25% of the
    labelled books - e.g. sci-fi + religion - these are the cross-genre reader communities.
    """
    df = pd.DataFrame({"community": labels, "genre": meta["primary_genre"].to_numpy(),
                       "title": meta["title"].to_numpy()})
    if W is not None:
        coo = W.tocoo()
        same = labels[coo.row] == labels[coo.col]
        df["centrality"] = np.bincount(coo.row[same], weights=coo.data[same], minlength=len(labels))
    else:
        df["centrality"] = 0.0
    n_genres_total = df["genre"].nunique()
    rows = []
    for c, grp in df.groupby("community"):
        g = grp["genre"].dropna().value_counts()
        if g.sum() < min_labelled:
            continue
        p = (g / g.sum()).to_numpy()
        entropy = float(-(p * np.log(p)).sum() / np.log(n_genres_total))
        top = grp.dropna(subset=["title"]).nlargest(n_titles, "centrality")["title"].str.slice(0, 60)
        rows.append({"community": c, "size": len(grp), "labelled": int(g.sum()),
                     "genre_1": g.index[0], "share_1": round(p[0], 2),
                     "genre_2": g.index[1] if len(g) > 1 else None, "share_2": round(p[1], 2) if len(g) > 1 else 0.0,
                     "entropy": round(entropy, 3), "mixed": len(g) > 1 and p[1] >= 0.25,
                     "top_titles": " | ".join(top)})
    return pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)
