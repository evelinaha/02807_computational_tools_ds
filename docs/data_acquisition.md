# Data: how it was obtained, inspected and loaded

This covers the path from Amazon's review data to the inputs of the graph-clustering notebook. What the data looks
like and why each preprocessing choice was made are in [`graph_clustering_decisions.md`](graph_clustering_decisions.md)
(sections 1–2).

## 1. The pull (done by a teammate)

- Source: the Amazon Books reviews and metadata. The files turned out to be the **2014 release** (McAuley et al.):
  reviews from 1996-05-20 to 2014-07-23.
- **Sampling by user, not by review:** about 50% of all Books users were sampled, and *all* book reviews of each
  sampled user were kept. This keeps each reader's whole history, which co-reader graphs, clusters and per-user
  recommendations need. A random sample of reviews would break histories apart.
- Books and users with fewer than 2 entries were removed. The pull took more than 2 hours.
- Output: `books_50pct.zip` (2.2 GB), holding two Parquet files:
  - `books_50pct_users_reviews.parquet`: 2.3 GB, 4,408,904 reviews. Columns `user_id, item_id, rating, summary,
    review_text, helpful_votes, timestamp`.
  - `books_50pct_users_meta.parquet`: 158 MB, 349,741 books. Columns `item_id, title, description, price, brand,
    categories`.
- The pull script is not in this repo. The teammate who ran it can add it under `scripts/` so the sample is fully
  reproducible.

## 2. Getting the files in place

The data is too big for git and is shared outside the repo. Put it next to the repo:

```bash
# from the folder that contains the repo
mkdir -p data/books_50pct
unzip -q books_50pct.zip -x "__MACOSX/*" -d data/books_50pct
```

```text
<parent>/
├── data/books_50pct/books_50pct_users_reviews.parquet
├── data/books_50pct/books_50pct_users_meta.parquet
└── 02807_computational_tools_ds/        (this repo)
```

The code finds the data at `../data/books_50pct/` by default. Set `CTDS_DATA_DIR=/path/to/books_50pct` to use
another location. `.gitignore` blocks `data/`, `*.zip`, `*.parquet` and `*.npz`, so none of it can be committed by
accident.

## 3. Inspection (Step 0, before any thresholds were chosen)

```bash
uv run python scripts/inspect_data.py > docs/results/data_inspection.txt   # ~15 s
```

[`scripts/inspect_data.py`](../scripts/inspect_data.py) reads both files out of core with DuckDB, so the 2.3 GB of
review text is never loaded. It prints:
- schema and row counts, plus sample rows
- missing values, duplicate pairs, and the rating and year distributions
- reviews per user and per book
- metadata coverage: titles, descriptions, author/brand, and genres from `categories`
- iterative k-core simulations for candidate thresholds and cut-offs

The saved output is [`results/data_inspection.txt`](results/data_inspection.txt). These numbers were the basis for
D1–D6: the cut-off 2014-01-01, the k-core users ≥ 5 / books ≥ 15, the same-series check instead of same-author, and
NMI only on books with a genre.

## 4. Loading in the pipeline

`ctds.graph.data.prepare(cfg)` does the shared preprocessing and caches the result:

1. DuckDB reads only `user_id, item_id, rating, timestamp` from the reviews file. This step also applies the
   optional time window, the optional user sample (`sample_config()`, 15% of users) and the optional
   `asin → canonical_id` mapping from the LSH part.
2. Iterative k-core (users ≥ 5, books ≥ 15), with a log of every step.
3. The `liked` flag: rating ≥ 4, and optionally sentiment not negative.
4. Split at 2014-01-01 into train and test.
5. Results are cached as Parquet in `../data/cache/graph/<run tag>/` (`train`, `test`, `kcore_log`). The user × book
   matrices and their index mappings (`bipartite_*.npz`, `user_index.parquet`, `book_index.parquet`) are saved
   there too, so other parts can reuse the same ids.

The metadata is loaded per book with `ctds.graph.metadata.load_book_metadata`, which adds genre labels and series
keys.
