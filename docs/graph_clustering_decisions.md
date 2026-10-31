# Graph clustering: data and design decisions

Decision log for the graph-clustering part. Each entry gives the decision, the reason and,
where relevant, what it means for the rest of the group. Results are in
[`graph_clustering_results.md`](graph_clustering_results.md).

## 1. The data

### How it was collected (as described by the teammate who ran the pull)

- About 50% of all Amazon Books users were sampled. The pull took more than 2 hours.
- The sample is **user-based, not review-based**: for every sampled user, *all* of that
  user's book entries were pulled, so we see each reader's whole history. That is what makes
  co-reader graphs, clusters and per-user recommendations possible. A random sample of
  reviews would break user histories apart.
- Books and users with fewer than 2 entries were removed.
- The data is too big for the repo. It is shared separately as `books_50pct.zip`.

### What the files contain (checked in Step 0)

| File | Rows | Columns |
|---|---|---|
| `books_50pct_users_reviews.parquet` (2.3 GB) | 4,408,904 reviews, 301,553 users, 349,741 books | `user_id, item_id, rating, summary, review_text, helpful_votes, timestamp` |
| `books_50pct_users_meta.parquet` (158 MB) | 349,741 books (one row for every reviewed book) | `item_id, title, description, price, brand, categories` |

- **This is the Amazon 2014 Books release (McAuley et al.), not the 2023 release.**
  Reviews run from 1996-05-20 to 2014-07-23. `timestamp` is Unix **seconds**,
  `helpful_votes` is `[helpful, total]`, and the join key is `item_id`, since there is no
  `parent_asin`. The notes in the project plan about the 2023 format (milliseconds,
  `parent_asin`, `bought_together`) do not apply.
- The entries are **reviews**, which we use as a proxy for "bought/read". Amazon only lets
  buyers write some reviews, but not all reviewers are verified buyers, and many buyers never
  review.
- Quality: no missing ratings, no duplicate (user, book) pairs, and 314 empty review texts.
  After the source filter, the smallest user has 3 reviews and the smallest book has 2.
- Ratings: 56% are 5★, 25% are 4★, 11% are 3★ and 8% are 1–2★.
- The data is skewed. Reviews per user: median 8, p99 129, max 15,761 (38 users have
  1,000 or more). Reviews per book: median 6, p99 116, max 3,722.
- The years are uneven. 78% of reviews are from 2012 onward (2013 alone is 34%, and
  January–July 2014 is 22%).
- **Metadata gaps:** `brand` is empty for every book, so there is **no author**. Titles are
  missing for 22% of books and descriptions for 35%. 78% of books carry only the category
  `[[Books]]`, with no genre.

## 2. Preprocessing decisions

These replace the group's original plan, which was written for the 2023 data.

| # | Decision | Why |
|---|---|---|
| D1 | **Time window: all years (1996 to July 2014). Cut-off: 2014-01-01.** Graphs are built from reviews before 2014; Jan–Jul 2014 reviews are the test set. | The 2015–2023 window and the 2022 cut-off would leave zero reviews. A 2014 cut-off keeps the "train on the past, test on the most recent ~7 months" design, and gives 24% test reviews. A 2013-07-01 cut-off would give 43%, which is too much of the data. Keeping pre-2010 reviews keeps about 25% more books after the k-core. Those are mostly long-tail books, which are the books this project is about. |
| D2 | **Iterative k-core: users ≥ 5 reviews, books ≥ 15 reviews**, applied to the whole window *before* the date split. | Result: 2,287,187 reviews, 165,151 users, 51,981 books (15 iterations, log in `kcore_log.parquet`). Books with 10 or more reviews would keep 87k books, but many of them would have only a handful of co-readers, which gives noisy edges. Books with 20 or more would leave only 35k books and cut deeper into the long tail. Applying the k-core before the split keeps the same set of books for train and test. |
| D3 | **"Liked" = rating ≥ 4**, with an optional switch to also require "sentiment not negative" once the sentiment column exists (`Config.exclude_negative_sentiment`, `sentiment_path`). | As agreed. 80% of training reviews count as liked. |
| D4 | **Duplicate editions:** an optional `asin → canonical_id` mapping (`Config.canonical_map_path`, Parquet or CSV). Without it, every book maps to itself. When two merged editions were reviewed by the same user, we keep the higher rating and the earlier date. | This lets the LSH teammate's output plug straight in. |
| D5 | **Same-author ablation → same-series ablation.** A series key is parsed from the title, e.g. "Kiss of Fire (Dragonfire, Book 1)" → `dragonfire`. Imprints and format notes ("Modern Library", "w/audio CD", "Berkley Sensation") are filtered out. | There is no author field. Series are the main thing that inflates "same author" hits anyway. Coverage: 6,931 of the 47,937 training books (14%) are in 2,036 series. This is an approximation: books without a title, or without a series marker in the title, are missed. |
| D6 | **Genre labels for NMI:** second-level genre from `Books > X` or `Kindle Store > Kindle eBooks > X` (the names are the same). A book with several genres gets the **least frequent** one as its primary label. NMI is computed only on books that have a genre (9,349 of 51,981, 18%), with a robustness check on books that have exactly one genre. | Most books have no genre, so NMI can only be measured on the labelled subset. The report has to say so. "Literature & Fiction" is attached to almost everything, so taking the rarest label keeps the informative one (e.g. Romance or Sci-fi). |
| D7 | **Package renamed** `02807_computational_tools_ds` → `ctds` (`import ctds`). The distribution name stays the same. | A Python module name cannot start with a digit. |
| D8 | **Data location:** default `../data/books_50pct/` (next to the repo), overridable with the env var `CTDS_DATA_DIR`. Caches go to `../data/cache/graph/<config tag>/`. `.gitignore` blocks data, zips, Parquet and npz files. | The data must never be committed. |
| D9 | **Sample mode** (`sample_config()`): a deterministic 15% of users (hash of `user_id`), with books ≥ 10 instead of ≥ 15. This gives 107k reviews, 9.3k users and 5.3k books. | This lets the whole notebook run in a few minutes. The lower book threshold is needed because a 15% user sample leaves few books with 15 or more reviews. Sample results are only for checking that the code works. |

**For the rest of the group:** the data is the 2014 release, so the sentiment, LSH and cold-start
parts should use `item_id` as the key and Unix-seconds timestamps, and should use the same cut-off
and k-core (D1, D2) so that the test items are identical.

## 3. Graph and clustering decisions

| # | Decision | Why |
|---|---|---|
| D10 | **Heavy users are down-weighted by w_u = 1/log(1 + d_u)**, not removed or capped. A cap (`max_user_books`, random subsample) exists but is off by default. | After the k-core, the heaviest training user has 2,049 books. The 34 users (0.02%) with more than 500 books create 17.5% of all co-occurrence pairs. A user who reviews everything says little about whether two books are similar, so these users count less (the same idea as IDF / Adamic–Adar). The full product B^T B fits in memory (about 53M pair contributions), so no cap is needed for memory. |
| D11 | **Edges need at least 2 shared users** (`min_shared=2`). | With the full data, a threshold of 1 gives 40M (co-reader) and 28M (co-liked) edges, most of them single coincidences. With 2 the counts are 7.0M and 4.0M. |
| D12 | **The co-occurrence product is computed in blocks of 4,096 books**, thresholded per block. | This bounds memory and never forms a dense all-pairs matrix. Checked against a dense computation on the sample, the maximum difference is 4e-6 (float32). |
| D13 | **Disparity filter with the "OR" rule** (keep an edge if it is significant for at least one endpoint, as in Serrano et al.). For comparisons, **α is chosen per graph to match the edge budget of top-10**. A fixed-α sweep (0.3, 0.1, 0.05, 0.01) is reported separately. | The disparity p-values depend heavily on the weighting. At α = 0.05 on the co-liked graph, raw counts keep 47% of books, Jaccard 80% and PPMI 3%, because PPMI weights are almost uniform around each node. Comparing at a fixed α would compare backbones of very different sizes. |
| D14 | **Louvain uses igraph** (`community_multilevel`, C implementation). networkx's `louvain_communities` is run once on the main graph as a check. Resolutions 0.5–20, 5 seeds each. Stability is the mean pairwise NMI/ARI across seeds. | igraph is about 100x faster, which makes the sweep over 12 graph variants cheap. Both libraries reach the same modularity. |
| D15 | **Spectral clustering runs on the giant component of the main backbone at full size** (no subsample). eigsh finds the largest eigenvalues of D^-1/2 W D^-1/2, i.e. the smallest of L_sym, then rows are normalised and clustered with k-means (Ng–Jordan–Weiss). k = 20, 40, 60. | The backbone is sparse enough (about 300k edges, 44k nodes) that eigsh for 60 vectors takes about 5 s. A disconnected graph would add one zero eigenvalue per component. |
| D16 | **Main variant for the detailed analysis: co-liked / Jaccard / disparity**, fixed before looking at results. | "Liked" encodes positive taste. Jaccard corrects for popularity without PMI's rare-pair bias. The disparity filter is the main backbone method. All 12 variants (2 sources × 3 weightings × 2 sparsifiers) are still compared in the tables. |
| D17 | **Our modularity function** implements Q = Σ_c [in_c/2m − γ(d_c/2m)²] and is checked against networkx and igraph. | They agree to about 1e-8 on the full graph. |
| D18 | **PPMI without smoothing.** | PPMI gives popular books rare neighbours, e.g. books with about 15 readers become The Hunger Games' top neighbours. Context-distribution smoothing (0.75) did not change this. We report it as a finding rather than tuning it away. For long-tail books PPMI and Jaccard agree. |

## 4. Recommender and evaluation decisions

| # | Decision | Why |
|---|---|---|
| D19 | **Test set:** users with at least one liked training book and at least one liked test book (Jan–Jul 2014) that exists in the training graph and wasn't read before. The same users and items are used for every variant. | Books first reviewed after the cut-off can't be reached by any graph. They belong to the cold-start part. |
| D20 | **Popularity groups:** books sorted by training readers and split so that each of head / mid / tail holds one third of all training interactions. Per group, only the user's relevant books in that group count. | The cut is defined by interaction mass, so it reflects real exposure. The head is a few thousand bestsellers and the tail is most of the catalogue. |
| D21 | **Added a "readers also bought" item-kNN baseline** (sum of raw co-reader counts with the user's liked books), alongside "most popular". | The project question is whether communities beat *purchase overlap alone*. Popularity alone is too weak a baseline for that. |
| D22 | **Three ways to rank books within a community:** weighted degree (`strength`, as in the plan), `popularity`, and `personal` (backbone links to the user's own liked books, restricted to the same community). Also a **hybrid** (1 − λ)·item-kNN + λ·community, and a **backbone-kNN ablation** (the same graph with no community restriction). | The plan's version (community × centrality) gives every user in a community the same hub books. Recall was 0.0036, below "most popular". The personal ranking keeps the community as the candidate set and the graph for the order. Backbone-kNN isolates what the clustering step itself adds. |
| D23 | **No tuning on the test set:** a random 20% of the test users form a validation set. It is used to choose the resolution, the within-community ranking and λ on the main variant (criterion: recall@10). All reported numbers are on the other 80%. | The test period is the only held-out data, so the choices have to be made on users we don't report. |
| D24 | **Significance:** paired bootstrap over users (2,000 resamples, 95% CI) of per-user recall@10. | Differences of a few thousandths need intervals. |
| D25 | **Metrics:** Hit@10, Recall@10 (hits / number of relevant items), NDCG@10 (binary relevance, ideal DCG over min(#relevant, 10)), and catalogue coverage. | These are standard top-k metrics. |

## 5. Practical notes

- On this Mac the project venv's `.pth` file gets the macOS "hidden" flag after `uv sync`, and Python 3.13 skips
  hidden `.pth` files, so `import ctds` fails. The notebook adds `src/` to `sys.path` itself and pytest has
  `pythonpath = ["src"]`. For scripts, use `PYTHONPATH=src uv run python ...` (or `chflags nohidden .venv/lib/python3.13/site-packages/*.pth`).
- The sample run (`CTDS_SAMPLE=1`, the default) takes about 25 s. Sample-run outputs are git-ignored.
