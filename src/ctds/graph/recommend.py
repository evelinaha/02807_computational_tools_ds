"""Community-based recommender, baselines and top-k evaluation on the held-out period.

Every recommender exposes ``scores(users) -> dense (len(users) x n_books) array`` for a batch
of training-matrix rows; :func:`evaluate` masks books the user already read, takes the top k
and compares with the user's liked test books. The same test set (users and items) is used for
every graph variant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .matrix import Bipartite

GROUPS = ("head", "mid", "tail")


# ---------------------------------------------------------------- test set and popularity groups

@dataclass
class TestSet:
    """Held-out liked books of the evaluable users.

    Attributes:
        users: training-matrix row of each test user.
        relevant: CSR (n_test_users x n_books), 1 for every liked book in the test period that
            exists in the training graph and was not already read before the cut-off.
    """

    users: np.ndarray
    relevant: sp.csr_matrix


def make_test_set(test: pd.DataFrame, bp: Bipartite) -> TestSet:
    """Test users = users with >= 1 liked training book and >= 1 liked test book that is in the
    training graph. Books first reviewed after the cut-off cannot be reached by any graph and
    are left to the cold-start part."""
    uidx, bidx = bp.user_index(), bp.book_index()
    t = test[test["liked"]]
    u, b = t["user_id"].map(uidx), t["item_id"].map(bidx)
    ok = u.notna() & b.notna()
    u, b = u[ok].astype(np.int64).to_numpy(), b[ok].astype(np.int64).to_numpy()
    rel = sp.csr_matrix((np.ones(len(u), dtype=np.float32), (u, b)), shape=bp.read.shape)
    rel = rel - rel.multiply(bp.read)          # drop anything already read in training
    rel.eliminate_zeros()
    has_seed = np.diff(bp.liked.indptr) > 0
    users = np.flatnonzero((np.diff(rel.indptr) > 0) & has_seed)
    return TestSet(users=users, relevant=rel[users].tocsr())


def book_popularity(bp: Bipartite) -> np.ndarray:
    """Number of training readers per book."""
    return np.asarray(bp.read.sum(axis=0)).ravel()


def popularity_groups(popularity: np.ndarray) -> np.ndarray:
    """Head / mid / tail label per book: books sorted by popularity and cut so that each group
    holds one third of all training interactions. The head is therefore a few hundred bestsellers
    and the tail the large majority of books."""
    order = np.argsort(-popularity, kind="stable")
    cum = np.cumsum(popularity[order]) / popularity.sum()
    g = np.empty(len(popularity), dtype=object)
    g[order] = np.where(cum <= 1 / 3, "head", np.where(cum <= 2 / 3, "mid", "tail"))
    return g


# ---------------------------------------------------------------- recommenders

class PopularityRecommender:
    """Most popular books (training readers) that the user has not read."""

    name = "most popular"

    def __init__(self, popularity: np.ndarray):
        self.pop = popularity.astype(np.float32)

    def scores(self, seeds: sp.csr_matrix) -> np.ndarray:
        return np.broadcast_to(self.pop, (seeds.shape[0], len(self.pop))).copy()


class ItemKNNRecommender:
    """"Readers also bought": score(u, b) = sum over the user's liked books j of C[j, b], with C the
    co-reader graph. This is purchase overlap alone - the baseline the project wants to beat.
    Ties/empty rows are filled by popularity."""

    def __init__(self, C: sp.csr_matrix, popularity: np.ndarray, name: str = "item-kNN (co-reader)"):
        self.C = C.tocsr().astype(np.float32)
        self.fill = (1e-6 * popularity / popularity.max()).astype(np.float32)
        self.name = name

    def scores(self, seeds: sp.csr_matrix) -> np.ndarray:
        return (seeds @ self.C).toarray() + self.fill


class CommunityRecommender:
    """Recommend books from the communities of the books the user liked before the cut-off.

    score(u, b) = share_u(c(b)) * rank_c(b), where share_u(c) is the fraction of the user's liked
    books that fall in community c, and rank_c(b) in (0, 1] is the book's within-community score
    divided by the community maximum - either its weighted degree inside the community
    (``within="strength"``, central books first) or its number of training readers
    (``within="popularity"``). Books outside the user's communities get only a tiny
    popularity term, which fills the list when the communities are too small.

    ``within="personal"``: rank books by their backbone edges to the user's own liked books,
    counting only edges *inside* the same community, i.e. score(u, b) = sum_{j liked by u,
    c(j) = c(b)} W[j, b]. The community restricts the candidate set and the graph personalises
    the order; the share x popularity score above is kept as a small tie-breaker.
    """

    def __init__(self, labels: np.ndarray, W: sp.csr_matrix, popularity: np.ndarray,
                 within: str = "strength", name: str = "community"):
        labels = np.asarray(labels)
        _, self.labels = np.unique(labels, return_inverse=True)
        n, k = len(self.labels), self.labels.max() + 1
        coo = W.tocoo()
        same = self.labels[coo.row] == self.labels[coo.col]
        self.within = within
        self.W_intra = None
        if within == "strength":
            s = np.bincount(coo.row[same], weights=coo.data[same], minlength=n)
        elif within in ("popularity", "personal"):
            s = popularity.astype(np.float64)
            if within == "personal":
                self.W_intra = sp.csr_matrix((coo.data[same].astype(np.float32), (coo.row[same], coo.col[same])), shape=(n, n))
        else:
            raise ValueError(within)
        cmax = np.zeros(k)
        np.maximum.at(cmax, self.labels, s)
        rank = np.divide(s, cmax[self.labels], out=np.zeros(n), where=cmax[self.labels] > 0)
        # Book -> community membership, and the same weighted by the within-community rank.
        self.M = sp.csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n), self.labels)), shape=(n, k))
        self.MrT = sp.csr_matrix((rank.astype(np.float32), (self.labels, np.arange(n))), shape=(k, n))
        self.fill = (1e-6 * popularity / popularity.max()).astype(np.float32)
        self.name = name

    def scores(self, seeds: sp.csr_matrix) -> np.ndarray:
        share = seeds @ self.M                                   # users x communities
        totals = np.asarray(seeds.sum(axis=1)).ravel()
        share = sp.diags(1.0 / np.maximum(totals, 1)) @ share
        base = (share @ self.MrT).toarray()
        if self.W_intra is None:
            return base + self.fill
        personal = HybridRecommender._row_normalise((seeds @ self.W_intra).toarray())
        return personal + 1e-3 * base + self.fill


class HybridRecommender:
    """Purchase overlap plus a community boost:
        score(u, b) = (1 - lam) * knn(u, b) / max_b knn(u, b) + lam * comm(u, b) / max_b comm(u, b).
    ``lam = 0`` is the item-kNN baseline and ``lam = 1`` the pure community recommender, so the
    curve in between shows whether communities add information to co-purchase overlap.
    """

    def __init__(self, knn: ItemKNNRecommender, community: CommunityRecommender, lam: float, name: str | None = None):
        self.knn, self.community, self.lam = knn, community, lam
        self.name = name or f"hybrid lam={lam}"

    @staticmethod
    def _row_normalise(x: np.ndarray) -> np.ndarray:
        m = x.max(axis=1, keepdims=True)
        return np.divide(x, m, out=np.zeros_like(x), where=m > 0)

    def scores(self, seeds: sp.csr_matrix) -> np.ndarray:
        a = self._row_normalise(self.knn.scores(seeds))
        b = self._row_normalise(self.community.scores(seeds))
        return (1 - self.lam) * a + self.lam * b


def split_test_users(test: TestSet, val_frac: float = 0.2, seed: int = 42) -> tuple[TestSet, TestSet]:
    """Random user split of the test set into validation (for choosing resolution, ranking and
    lam) and the final test users that every reported number is computed on."""
    rng = np.random.default_rng(seed)
    is_val = rng.random(len(test.users)) < val_frac
    pick = lambda m: TestSet(users=test.users[m], relevant=test.relevant[np.flatnonzero(m)].tocsr())
    return pick(is_val), pick(~is_val)


# ---------------------------------------------------------------- evaluation

def _series_mask(read_rows: sp.csr_matrix, series_onehot: sp.csr_matrix) -> sp.csr_matrix:
    """users x books: 1 where the book belongs to a series the user read in training."""
    user_series = read_rows @ series_onehot
    user_series.data[:] = 1
    m = user_series @ series_onehot.T
    m.data[:] = 1
    return m.tocsr()


def series_onehot(series: pd.Series) -> sp.csr_matrix:
    """books x series one-hot matrix (books without a series key have an empty row)."""
    s = series.astype("string").to_numpy()
    has = pd.notna(s)
    codes, _ = pd.factorize(s[has])
    rows = np.flatnonzero(has)
    return sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, codes)), shape=(len(s), codes.max() + 1 if len(codes) else 0))


def _metrics_for(top: np.ndarray, rel: sp.csr_matrix, k: int) -> dict[str, np.ndarray]:
    """Per-user hit, recall and NDCG at k for one batch (users with no relevant item get NaN)."""
    n = top.shape[0]
    hit_matrix = rel.astype(bool).toarray()[np.arange(n)[:, None], top]
    n_rel = np.diff(rel.indptr).astype(np.float64)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hit_matrix * discounts).sum(axis=1)
    idcg = np.array([discounts[: int(min(m, k))].sum() for m in n_rel])
    valid = n_rel > 0
    nan = np.full(n, np.nan)
    hits = hit_matrix.sum(axis=1)
    return {"hit": np.where(valid, hits > 0, nan),
            "recall": np.where(valid, hits / np.maximum(n_rel, 1), nan),
            "ndcg": np.where(valid, dcg / np.where(idcg > 0, idcg, 1), nan)}


def evaluate(recommender, bp: Bipartite, test: TestSet, groups: np.ndarray,
             series: sp.csr_matrix | None = None, k: int = 10, batch_size: int = 1024,
             return_per_user: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Hit rate, Recall and NDCG at ``k`` overall and per popularity group, with and without
    same-series books.

    - Per group g, only the user's relevant items in group g count, and only users who have at
      least one such item are averaged (the top-k list is the same).
    - ``series`` given: a second evaluation where books from a series the user already read in
      training are removed both from the candidates and from the relevant items (series
      continuations are easy hits that inflate scores). Users left without relevant items drop
      out - identically for every recommender.

    Returns one row per (setting, group) with the metrics, the number of users and the share of
    all recommendations that fall in each popularity group. With ``return_per_user`` also a dict
    ``{(setting, group): {metric: per-user array}}`` (NaN for users without relevant items), in the
    same user order for every recommender, for paired significance tests.
    """
    group_masks = {g: (groups == g) for g in GROUPS}
    group_diag = {g: sp.diags(m.astype(np.float32)) for g, m in group_masks.items()}
    settings = ["all books"] + (["no same-series"] if series is not None else [])
    acc = {(s, g): {"hit": [], "recall": [], "ndcg": []} for s in settings for g in ("all",) + GROUPS}
    rec_counts = {s: np.zeros(bp.n_books) for s in settings}

    for start in range(0, len(test.users), batch_size):
        users = test.users[start:start + batch_size]
        seeds = bp.liked[users]
        read = bp.read[users]
        rel = test.relevant[start:start + batch_size]
        base_scores = recommender.scores(seeds)
        for setting in settings:
            sc = base_scores.copy() if setting != settings[-1] else base_scores
            masks = [read]
            rel_s = rel
            if setting == "no same-series":
                sm = _series_mask(read, series)
                masks.append(sm)
                rel_s = (rel - rel.multiply(sm)).tocsr()
                rel_s.eliminate_zeros()
            for m in masks:
                m = m.tocoo()
                sc[m.row, m.col] = -np.inf
            top = np.argpartition(-sc, k, axis=1)[:, :k]
            order = np.argsort(-np.take_along_axis(sc, top, axis=1), axis=1)
            top = np.take_along_axis(top, order, axis=1)
            np.add.at(rec_counts[setting], top.ravel(), 1)
            for g in ("all",) + GROUPS:
                if g == "all":
                    rg = rel_s
                else:
                    rg = (rel_s @ group_diag[g]).tocsr()
                    rg.eliminate_zeros()
                for name, vals in _metrics_for(top, rg, k).items():
                    acc[(setting, g)][name].append(vals)

    rows, per_user = [], {}
    for (setting, g), m in acc.items():
        vals = {name: np.concatenate(v) for name, v in m.items()}
        per_user[(setting, g)] = vals
        share = rec_counts[setting][group_masks[g]].sum() / rec_counts[setting].sum() if g != "all" else 1.0
        rows.append({"recommender": recommender.name, "setting": setting, "group": g,
                     "users": int(np.sum(~np.isnan(vals["hit"]))),
                     f"hit@{k}": np.nanmean(vals["hit"]), f"recall@{k}": np.nanmean(vals["recall"]),
                     f"ndcg@{k}": np.nanmean(vals["ndcg"]), "rec_share": share,
                     "coverage": (rec_counts[setting] > 0).mean() if g == "all" else np.nan})
    df = pd.DataFrame(rows)
    return (df, per_user) if return_per_user else df


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean difference a - b of a per-user metric with a 95% paired-bootstrap confidence interval
    (users resampled with replacement; users with NaN in either array are dropped)."""
    ok = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[ok]
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"users": int(ok.sum()), "mean_a": float(a[ok].mean()), "mean_b": float(b[ok].mean()),
            "diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "relative_pct": float(100 * d.mean() / b[ok].mean()) if b[ok].mean() > 0 else np.nan,
            "significant": bool(lo > 0 or hi < 0)}
