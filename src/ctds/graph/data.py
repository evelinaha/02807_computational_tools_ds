"""Loading and the shared preprocessing: time window, duplicate editions, iterative k-core,
train/test split by date, and the "liked" flag.

DuckDB reads only the four columns we need straight from Parquet (the 2.3 GB file is
mostly review text), so the full table never has to be loaded into pandas.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .config import Config

COLUMNS = ["user_id", "item_id", "rating", "timestamp"]


def _read_any(path: Path) -> str:
    """DuckDB table expression for a Parquet or CSV file."""
    path = Path(path)
    if path.suffix == ".csv":
        return f"read_csv_auto('{path}')"
    return f"read_parquet('{path}')"


def load_reviews(cfg: Config, con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """Load (user_id, item_id, rating, timestamp) with the time window, the optional user sample
    and the optional duplicate-edition mapping applied.

    ``timestamp`` is Unix *seconds* in this dataset (Amazon 2014 release).

    If a canonical mapping merges two editions that the same user reviewed, the pair is kept
    once with the highest rating and the earliest timestamp.
    """
    con = con or duckdb.connect()
    where = ["true"]
    if cfg.window_start:
        where.append(f"timestamp >= epoch(timestamp '{cfg.window_start}')")
    if cfg.sample:
        # Deterministic user sample: same users every run, and all of each user's reviews are kept.
        where.append(f"hash(user_id || '{cfg.seed}') % 100 < {cfg.sample_user_pct}")
    base = f"select {', '.join(COLUMNS)} from {_read_any(cfg.reviews_path)} where {' and '.join(where)}"

    if cfg.canonical_map_path:
        sql = f"""
            with r as ({base}),
                 m as (select asin, canonical_id from {_read_any(cfg.canonical_map_path)})
            select r.user_id, coalesce(m.canonical_id, r.item_id) as item_id,
                   max(r.rating) as rating, min(r.timestamp) as timestamp
            from r left join m on r.item_id = m.asin
            group by 1, 2"""
    else:
        sql = base
    df = con.execute(sql).fetchdf()
    df["rating"] = df["rating"].astype("float32")
    return df


def iterative_kcore(df: pd.DataFrame, min_user: int, min_book: int, max_iter: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeatedly drop users with < ``min_user`` reviews and books with < ``min_book`` reviews
    until both conditions hold.

    Returns the filtered reviews and a log with one row per filtering step (what was removed
    and what remains) - the numbers go straight into the report.
    """
    # Work on integer codes: counting with bincount is ~50x faster than value_counts on strings.
    codes = {"user_id": pd.factorize(df["user_id"])[0], "item_id": pd.factorize(df["item_id"])[0]}
    alive = np.ones(len(df), dtype=bool)

    def n_distinct(col: str) -> int:
        return int(np.count_nonzero(np.bincount(codes[col][alive])))

    log = [{"iteration": 0, "step": "start", "reviews_removed": 0, "users_removed": 0, "books_removed": 0,
            "reviews": len(df), "users": n_distinct("user_id"), "books": n_distinct("item_id")}]
    for it in range(1, max_iter + 1):
        changed = False
        for step, col, k in [("users", "user_id", min_user), ("books", "item_id", min_book)]:
            counts = np.bincount(codes[col][alive], minlength=codes[col].max() + 1)
            drop = alive & (counts[codes[col]] < k)
            if not drop.any():
                continue
            changed = True
            before = log[-1]
            alive &= ~drop
            users, books = n_distinct("user_id"), n_distinct("item_id")
            log.append({"iteration": it, "step": f"drop {step} < {k}",
                        "reviews_removed": int(drop.sum()),
                        "users_removed": before["users"] - users,
                        "books_removed": before["books"] - books,
                        "reviews": int(alive.sum()), "users": users, "books": books})
        if not changed:
            break
    return df[alive].reset_index(drop=True), pd.DataFrame(log)


def add_liked(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add the boolean ``liked`` column: rating >= threshold, and (optionally) the sentiment
    teammate's label is not ``negative``. Without a sentiment file only the rating is used."""
    liked = df["rating"] >= cfg.like_threshold
    if cfg.exclude_negative_sentiment:
        if cfg.sentiment_path is None:
            raise ValueError("exclude_negative_sentiment=True needs cfg.sentiment_path")
        sent = pd.read_parquet(cfg.sentiment_path, columns=["user_id", "item_id", "sentiment"])
        df = df.merge(sent, on=["user_id", "item_id"], how="left")
        liked &= df["sentiment"].astype("string").str.lower().ne("negative").fillna(True)
    return df.assign(liked=liked.to_numpy())


def train_test_split_by_date(df: pd.DataFrame, cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reviews strictly before ``cutoff`` are training data; reviews on/after it are test data."""
    ts = int(pd.Timestamp(cutoff, tz="UTC").timestamp())
    is_train = df["timestamp"] < ts
    return df[is_train].reset_index(drop=True), df[~is_train].reset_index(drop=True)


def prepare(cfg: Config, force: bool = False, verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Run the full preprocessing and cache the result as Parquet in ``cfg.run_dir``.

    Returns a dict with ``train``, ``test`` (both with a ``liked`` column) and ``kcore_log``.
    The k-core is applied to the whole window before the date split, as agreed in the group,
    so the set of books is the same for train and test.
    """
    out = cfg.run_dir
    files = {name: out / f"{name}.parquet" for name in ["train", "test", "kcore_log"]}
    if not force and all(f.exists() for f in files.values()):
        if verbose:
            print(f"loading cached preprocessing from {out}")
        return {name: pd.read_parquet(f) for name, f in files.items()}

    raw = load_reviews(cfg)
    core, log = iterative_kcore(raw, cfg.min_user_reviews, cfg.min_book_reviews)
    core = add_liked(core, cfg)
    train, test = train_test_split_by_date(core, cfg.cutoff)
    result = {"train": train, "test": test, "kcore_log": log}
    for name, frame in result.items():
        frame.to_parquet(files[name], index=False)
    if verbose:
        print(f"raw {len(raw):,} reviews -> k-core {len(core):,} "
              f"({core.user_id.nunique():,} users, {core.item_id.nunique():,} books); "
              f"train {len(train):,}, test {len(test):,}")
    return result
