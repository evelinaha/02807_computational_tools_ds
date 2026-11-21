"""Graph clustering with Ana Rita's description similarity (SBERT all-MiniLM-L6-v2, cosine).

Reuses the cached preprocessing of the main run, rebuilds the main backbone
(co-liked / Jaccard / disparity) and adds a content graph: every book's 10 most similar books
by description. Three graphs are then clustered with Louvain and evaluated exactly like the
main notebook (same test users, popularity groups and same-series filter):

- ``co-liked``  : the main backbone alone (reference, should reproduce the notebook)
- ``content``   : the description-similarity graph alone
- ``combined``  : (1 - beta) * co-liked / mean + beta * content / mean

beta, the resolution and the hybrid lambda are chosen on the validation users; everything
reported is on the held-out users. Outputs: ``docs/results/<run>/content_*.csv`` and
``content_summary.json``; the content edge list goes to the cache folder as
``content_knn_edges.parquet`` (the notebook's ``TEXT_GRAPH_PATH`` format).

Usage (from the repo root, after ``uv sync --group text``):
    PYTHONPATH=src uv run python scripts/content_graph_experiment.py
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctds.graph.build import cooccurrence_graph
from ctds.graph.cluster import louvain, modularity
from ctds.graph.config import REPO_ROOT, Config
from ctds.graph.content import (MODEL_NAME, combine_graphs, description_embeddings, knn_similarity_graph,
                                to_edge_list)
from ctds.graph.data import prepare
from ctds.graph.evaluate import genre_nmi, size_summary
from ctds.graph.matrix import Bipartite
from ctds.graph.metadata import load_book_metadata
from ctds.graph.recommend import (CommunityRecommender, HybridRecommender, ItemKNNRecommender, PopularityRecommender,
                                  book_popularity, evaluate, make_test_set, paired_bootstrap, popularity_groups,
                                  series_onehot, split_test_users)
from ctds.graph.sparsify import backbone_stats, disparity_alpha_for_edges, disparity_filter, top_k_neighbours
from ctds.graph.utils import timed
from ctds.graph.weights import apply_weighting

SEED, K, TOP_K, MIN_SHARED = 42, 10, 10, 2
BETAS = [0.1, 0.25, 0.5, 0.75]
RESOLUTIONS = [1, 5]
LAMBDAS = [0.3, 0.5, 0.7]

cfg = Config()
RESULTS = REPO_ROOT / "docs" / "results" / cfg.tag
RESULTS.mkdir(parents=True, exist_ok=True)
SUMMARY: dict = {"model": MODEL_NAME, "knn_k": TOP_K}


@contextmanager
def wall(step: str):
    """Plain wall-clock timer. ``timed`` runs tracemalloc, which slows long Python loops ~40x."""
    t0 = time.perf_counter()
    yield
    print(f"[{step}] {time.perf_counter() - t0:.1f}s", flush=True)


def save(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.to_csv(RESULTS / f"content_{name}.csv", index=False)
    return df


# ---------------------------------------------------------------- data and the main backbone
data = prepare(cfg)
train, test = data["train"], data["test"]
bp = Bipartite.load(cfg.run_dir)
meta = load_book_metadata(cfg, bp.book_ids)
pop = book_popularity(bp)

with timed("graphs"):
    co_reader = cooccurrence_graph(bp.read, min_shared=MIN_SHARED, seed=SEED, name="co-reader")
    co_liked = cooccurrence_graph(bp.liked, min_shared=MIN_SHARED, seed=SEED, name="co-liked")
    W = apply_weighting(co_liked.weights, "jaccard", co_liked.strength)
    n_edges = top_k_neighbours(W, TOP_K).nnz // 2
    D = disparity_filter(W, disparity_alpha_for_edges(W, n_edges))
    del W

# ---------------------------------------------------------------- Ana Rita's similarity as a graph
with timed("sbert embeddings"):
    nodes, E = description_embeddings(meta, cache=cfg.run_dir / "sbert_descriptions.npz")
with timed("content knn graph"):
    T = knn_similarity_graph(nodes, E, bp.n_books, k=TOP_K)
to_edge_list(T, bp.book_ids).to_parquet(cfg.run_dir / "content_knn_edges.parquet", index=False)

overlap = D.multiply(T > 0)
graph_rows = [{"graph": "co-liked backbone", **backbone_stats(D, D)},
              {"graph": "content kNN", **backbone_stats(T, T)}]
SUMMARY["coverage"] = {
    "books": int(bp.n_books), "books_with_description": int(len(nodes)),
    "pct_with_description": round(100 * len(nodes) / bp.n_books, 1),
    "content_edges": int(T.nnz // 2), "backbone_edges": int(D.nnz // 2),
    "shared_edges": int(overlap.nnz // 2),
    "shared_pct_of_content": round(100 * overlap.nnz / T.nnz, 2),
    "mean_cosine": float(T.data.mean()),
}
print(SUMMARY["coverage"])

# ---------------------------------------------------------------- test set (identical to the notebook)
test_all = make_test_set(test, bp)
val, final = split_test_users(test_all, val_frac=0.2, seed=SEED)
groups = popularity_groups(pop)
S = series_onehot(meta["series"])
knn = ItemKNNRecommender(co_reader.shared, pop, name="item-kNN (co-reader)")
popular = PopularityRecommender(pop)
print(f"test users: validation {len(val.users)}, final {len(final.users)}")


def partition_stats(G: sp.csr_matrix, lab: np.ndarray, name: str) -> dict:
    has_edge = np.flatnonzero(np.diff(G.indptr) > 0)
    coo = G.tocoo()
    intra = (lab[coo.row] == lab[coo.col])
    return {"graph": name, "modularity": modularity(G, lab), **size_summary(lab, has_edge),
            **genre_nmi(lab, meta, has_edge, seed=SEED),
            "intra_weight_pct": round(100 * coo.data[intra].sum() / coo.data.sum(), 1)}


# ---------------------------------------------------------------- validation: beta and resolution
graphs = {"co-liked": D, "content": T}
graphs.update({f"combined b={b}": combine_graphs(D, T, b) for b in BETAS})
labels, val_rows, part_rows = {}, [], []
with wall("validation"):
    r = evaluate(knn, bp, val, groups, k=K).set_index("group")
    val_rows.append({"graph": "item-kNN", "resolution": np.nan, "ranking": "-",
                     **{f"recall_{g}": r.loc[g, "recall@10"] for g in ["all", "head", "mid", "tail"]}})
    for name, G in graphs.items():
        for res in RESOLUTIONS:
            lab = louvain(G, res, seed=0)
            labels[(name, res)] = lab
            part_rows.append({**partition_stats(G, lab, name), "resolution": res})
            # Rank inside communities with the graph itself, and - for the combined graphs - also with the
            # co-liked backbone only, which isolates what the content edges change in the communities.
            rankers = {"own graph": G} if name == "co-liked" else {"own graph": G, "co-liked edges": D}
            for ranking, Wr in rankers.items():
                rec = CommunityRecommender(lab, Wr, pop, within="personal")
                r = evaluate(rec, bp, val, groups, k=K).set_index("group")
                val_rows.append({"graph": name, "resolution": res, "ranking": ranking,
                                 **{f"recall_{g}": r.loc[g, "recall@10"] for g in ["all", "head", "mid", "tail"]},
                                 "coverage": r.loc["all", "coverage"]})
                print(val_rows[-1])
val_grid = save(pd.DataFrame(val_rows), "validation_grid")
partitions = save(pd.DataFrame(part_rows), "partitions")

comb = val_grid[val_grid.graph.str.startswith("combined")].sort_values("recall_all", ascending=False).iloc[0]
BEST_GRAPH, BEST_RES, BEST_RANKING = comb.graph, int(comb.resolution), comb.ranking
best_of = lambda g: val_grid[val_grid.graph == g].sort_values("recall_all", ascending=False).iloc[0]
CONTENT_RES = int(best_of("content").resolution)
G_best = graphs[BEST_GRAPH]
W_rank = G_best if BEST_RANKING == "own graph" else D
comm_comb = CommunityRecommender(labels[(BEST_GRAPH, BEST_RES)], W_rank, pop, within="personal", name="community combined")

lam_rows = []
for lam in LAMBDAS:
    r = evaluate(HybridRecommender(knn, comm_comb, lam), bp, val, groups, k=K).set_index("group")
    lam_rows.append({"lambda": lam, **{f"recall_{g}": r.loc[g, "recall@10"] for g in ["all", "head", "mid", "tail"]}})
lam_table = save(pd.DataFrame(lam_rows), "validation_lambda")
BEST_LAM = float(lam_table.loc[lam_table.recall_all.idxmax(), "lambda"])
SUMMARY["chosen"] = {"combined_graph": BEST_GRAPH, "resolution": BEST_RES, "ranking": BEST_RANKING,
                     "content_resolution": CONTENT_RES, "lambda": BEST_LAM}
print("chosen on validation:", SUMMARY["chosen"])

# ---------------------------------------------------------------- final evaluation on held-out users
comm_main = CommunityRecommender(labels[("co-liked", 1)], D, pop, within="personal", name="community co-liked")
models = [
    popular, knn,
    comm_main,
    ItemKNNRecommender(D, pop, name="backbone-kNN co-liked"),
    HybridRecommender(knn, comm_main, 0.5, name="hybrid co-liked"),
    CommunityRecommender(labels[("content", CONTENT_RES)], T, pop, within="personal", name="community content"),
    ItemKNNRecommender(T, pop, name="backbone-kNN content"),
    comm_comb,
    ItemKNNRecommender(G_best, pop, name="backbone-kNN combined"),
    HybridRecommender(knn, comm_comb, BEST_LAM, name="hybrid combined"),
]
results, per_user = [], {}
with wall("final evaluation"):
    for rec in models:
        r, pu = evaluate(rec, bp, final, groups, S, k=K, return_per_user=True)
        results.append(r)
        per_user[rec.name] = pu
final_results = save(pd.concat(results, ignore_index=True), "final_results")
nss = final_results[final_results.setting == "no same-series"]
wide = nss.pivot_table(index="recommender", columns="group", values="recall@10")[["all", "head", "mid", "tail"]]
wide["coverage"] = nss[nss.group == "all"].set_index("recommender")["coverage"]
wide = wide.reindex([m.name for m in models])
save(wide.reset_index(), "recall10_no_same-series")
print(wide.round(4).to_string())

comparisons = [("community combined", "community co-liked"),
               ("hybrid combined", "hybrid co-liked"),
               ("community combined", "backbone-kNN combined"),
               ("community combined", "item-kNN (co-reader)"),
               ("community content", "item-kNN (co-reader)"),
               ("community content", "community co-liked"),
               ("community content", "backbone-kNN content")]
rows = []
for a, b in comparisons:
    for g in ["all", "head", "mid", "tail"]:
        key = ("no same-series", g)
        rows.append({"a": a, "b": b, "group": g,
                     **paired_bootstrap(per_user[a][key]["recall"], per_user[b][key]["recall"], seed=SEED)})
sig = save(pd.DataFrame(rows), "significance_recall")
print(sig[sig.group.isin(["all", "tail"])].round(4).to_string())

SUMMARY["final"] = wide.round(5).reset_index().to_dict("records")
SUMMARY["significance"] = sig.round(5).to_dict("records")
(RESULTS / "content_summary.json").write_text(json.dumps(SUMMARY, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print("written to", RESULTS)
