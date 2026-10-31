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
