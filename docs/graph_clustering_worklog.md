# Graph clustering: work log

A record of the work done on the graph-clustering part (owner: Vishwaa), in the order it happened, so the group can
see what exists, how it was produced and what is still open. Details are in two companion documents:
- [`graph_clustering_decisions.md`](graph_clustering_decisions.md): every decision (D1–D25) and why it was made.
- [`graph_clustering_results.md`](graph_clustering_results.md): all numbers, tables and figure paths.

## Starting point (2026-10-31)

- The task was specified in the group plan for the graph part (`graph_clustering_prompt.md`, kept outside the
  repo). It covers the bipartite matrix, the book–book graphs, our own weighting and sparsification, Louvain and
  spectral clustering, community evaluation and a community-based recommender.
- The data is `books_50pct.zip`, from the teammate who ran the pull: a user-based pull of about 50% of Amazon Books
  users, with all of each sampled user's book entries. Books and users with fewer than 2 entries were removed. The
  pull took more than 2 hours, and the data is too big for the repo. It is unzipped to `../data/books_50pct/`, next
  to the repo.
- The repo had one commit (`f1e95be`, the template).

## Step 0: data inspection, before any pipeline code

The data was profiled with DuckDB, out of core: sizes, schema, sample rows, missing values, distributions, date
range and k-core simulations. The results were reported back, and these choices were confirmed before any thresholds
were set.

The data differs from the group plan's assumptions in three ways:
- **It is the Amazon 2014 release, not 2023.** Reviews run from 1996-05-20 to 2014-07-23, `timestamp` is in
  seconds, and the key is `item_id`, not `parent_asin`. The plan's 2015–2023 window and 2022 cut-off would leave
  zero reviews.
- **There is no author field** (`brand` is empty for every book). Only about 18% of the books that survive the k-core
  have a genre label.
- The entries are reviews, which we use as a proxy for purchases.

The confirmed choices:
- all years, with the cut-off at 2014-01-01 (D1)
- k-core users ≥ 5, books ≥ 15 (D2)
- the same-author check replaced by a same-series check based on titles (D5)
- NMI only on books that have a genre (D6)
- the package renamed to `ctds` (D7)
- the data path `../data/books_50pct/` (D8)

## Step 1: package, preprocessing, metadata (commit `367391d`)

- Renamed the package `02807_computational_tools_ds` → `ctds`, because a module name cannot start with a digit.
  Dependencies are managed with `uv`. Added a `.gitignore` that blocks data, zips, Parquet, npz and caches.
- `ctds.graph.config`: one `Config` holds all the shared thresholds, plus a sample mode (15% of users).
- `ctds.graph.data`: DuckDB loading of only the needed columns, the optional duplicate-edition mapping, an iterative
  k-core that logs every step, the "liked" flag (rating ≥ 4, plus optional sentiment), the date split and a Parquet
  cache. The k-core was switched to integer codes, which cut its runtime from 101 s to 6 s, and its output matches
  the DuckDB profile exactly.
- `ctds.graph.metadata`: genre labels (Books level 2 / Kindle eBooks level 3) and series keys parsed from titles,
  with imprints and format notes filtered out.
- Wrote the first part of the decisions document: how the data was collected, what the files contain, and D1–D9.

## Step 2: graphs, weighting, sparsification, clustering, tests (commit `2982b09`)

- `matrix`: user × book CSR matrices (read / liked), with the index mappings saved.
- `build`: B^T W B co-occurrence computed in blocks of books, with 1/log(1+d) down-weighting of heavy users, at least
  2 shared users per edge, an optional degree cap, and an adapter for external edge lists (for the teammate's
  review-text graph). Checked against a dense computation (maximum difference 4e-6).
- `weights` (ours): raw counts, Jaccard and PPMI.
- `sparsify` (ours): the disparity filter (Serrano et al. 2009), α matched to an edge budget, a top-k baseline, and
  backbone statistics.
- `cluster`: our modularity function, Louvain (igraph, plus networkx as a check) with a resolution and seed sweep,
  and our spectral clustering (eigsh + k-means).
- `tests/test_graph.py`: tests on toy graphs with hand-computed answers for PPMI, Jaccard, co-occurrence, the
  disparity filter (formula, OR/AND rules), top-k, modularity (by hand, and against networkx on Karate),
  Louvain/spectral on planted partitions, the k-core cascade and the edge-list adapter.
- Found that a fixed α means very different things for different weightings: at α = 0.05, PPMI keeps 3% of books
  and Jaccard 80%. So backbones are compared at a matched edge budget instead (D13).

## Step 3: evaluation, recommender, notebook (commit `3ad66c0`)

- `evaluate`: community sizes, NMI/ARI against genres with a random baseline, and a table of genres per community
  with a "mixed" flag.
- `recommend`: a test set that is the same for every variant, popularity groups (head / mid / tail by thirds of the
  interactions), and these recommenders:
  - most popular
  - item-kNN, "readers also bought" (added as the real baseline, D21)
  - community, with strength, popularity or personal ranking
  - hybrid
  - a backbone-kNN ablation

  Metrics: Hit / Recall / NDCG@10, per group, with and without same-series books. Also a validation split of the
  users and paired-bootstrap confidence intervals.
- Found that the community recommender exactly as specified in the plan (rank by centrality inside the community)
  scores below "most popular". The personal ranking inside communities fixes this (D22). All choices are made on
  20% validation users, never on the reported users (D23).
- `plotting`: a shared, colour-blind-safe style for the figures.
- `notebooks/graph_clustering.ipynb`: the whole part end to end, with run instructions. `CTDS_SAMPLE=1` (the
  default) takes about 25 s; `CTDS_SAMPLE=0` runs on the full data.

## Step 4: full-data run and write-up (commit `602b36c`)

- Ran the notebook on the full data: 34.6 min, peak memory 3.2 GB, on a 16 GB MacBook. The executed notebook is
  committed with its outputs. Tables are in `docs/results/full_fromall_cut2014-01-01_u5b15/` and figures in
  `docs/figures/full_fromall_cut2014-01-01_u5b15/`.
- Wrote `graph_clustering_results.md` and added D10–D25 to the decisions document.

### Main findings

- On 59,392 held-out users (no same-series books), the community recommender on the co-liked / Jaccard / disparity
  graph beats "readers also bought":
  - overall recall@10: +9%
  - tail recall: 3.1× (0.0272 vs 0.0089)
  - catalogue coverage: 82% vs 47%

  The hybrid (λ = 0.5) gives +16% overall. All three are significant.
- **The community restriction adds nothing over recommending straight from the same backbone** (difference +0.0001,
  not significant). At resolution 1, 91% of the backbone's edge weight lies inside communities. The long-tail gain
  comes from the graph construction. This affects how the report can present graph clustering as "essential"; see
  the last section of the results document.
- The communities are meaningful: modularity 0.84, NMI against genres 0.29 (random baseline 0.06), and 12 mixed-genre
  communities, e.g. post-apocalyptic fiction with prepper non-fiction, clean romance with Christian fiction, and
  M/M romance.

## Practical notes

- On macOS the venv's `.pth` file gets the "hidden" flag after `uv sync`, and Python 3.13 then skips it. The notebook
  adds `src/` to `sys.path` itself and pytest is configured with `pythonpath = ["src"]`. For scripts, use
  `PYTHONPATH=src uv run python ...`.
- Caches go to `../data/cache/graph/<run tag>/`, outside the repo.
- Reproduce:

  ```bash
  uv sync
  uv run pytest
  CTDS_SAMPLE=1 uv run jupyter nbconvert --to notebook --execute --inplace notebooks/graph_clustering.ipynb   # ~25 s
  CTDS_SAMPLE=0 uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 notebooks/graph_clustering.ipynb   # ~35 min
  ```

## Open items / hand-offs

- **LSH teammate:** pass the `asin → canonical_id` Parquet as `Config(canonical_map_path=...)`. Duplicate editions
  are visible today: *Catching Fire* appears twice among *The Hunger Games*' neighbours. For the review-text graph,
  set `TEXT_GRAPH_PATH` in the notebook (columns `book_a, book_b, weight`). It then goes through the same pipeline.
- **Sentiment teammate:** pass a Parquet with `user_id, item_id, sentiment` as `sentiment_path` and set
  `exclude_negative_sentiment=True`.
- **Cold-start teammate:** placing a brand-new book in a community through the text graph is where communities are
  actually needed, since a new book has no co-reader edges. That would make graph clustering essential.
- **Whole group:** use `item_id` and Unix-seconds timestamps, and the same cut-off and k-core (D1, D2), so the test
  items match across parts.
