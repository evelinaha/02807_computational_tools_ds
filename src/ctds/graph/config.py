"""Configuration for the graph-clustering pipeline.

All thresholds that are shared with the rest of the group live here, so that a change
is made in one place. See ``docs/graph_clustering_decisions.md`` for why each default
was chosen.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# The data is too big for the repo; by default it sits next to it in ../data/books_50pct.
DEFAULT_DATA_DIR = REPO_ROOT.parent / "data" / "books_50pct"

REVIEWS_FILE = "books_50pct_users_reviews.parquet"
META_FILE = "books_50pct_users_meta.parquet"


@dataclass(frozen=True)
class Config:
    """Pipeline parameters.

    Attributes:
        data_dir: folder with the two parquet files (env var ``CTDS_DATA_DIR`` overrides).
        cache_dir: where intermediate Parquet/npz files are written (outside the repo).
        window_start: first date kept (``None`` = keep all years, the agreed choice for the
            2014 data because pre-2010 reviews hold ~25% of the long-tail co-reader signal).
        cutoff: graphs are built from reviews strictly before this date; later reviews are test.
        min_user_reviews / min_book_reviews: iterative k-core thresholds.
        like_threshold: rating >= this counts as "liked".
        exclude_negative_sentiment: once the sentiment teammate's column exists, a liked review
            must also not be classified negative.
        sentiment_path: optional Parquet with columns ``user_id, item_id, sentiment``.
        canonical_map_path: optional Parquet/CSV ``asin, canonical_id`` from the LSH teammate
            (duplicate editions). ``None`` = identity mapping.
        sample: run on a deterministic user sample so the whole notebook runs in minutes.
        sample_user_pct: percentage of users kept when ``sample`` is True.
        seed: global random seed.
    """

    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("CTDS_DATA_DIR", DEFAULT_DATA_DIR)))
    cache_dir: Path | None = None
    window_start: str | None = None
    cutoff: str = "2014-01-01"
    min_user_reviews: int = 5
    min_book_reviews: int = 15
    like_threshold: float = 4.0
    exclude_negative_sentiment: bool = False
    sentiment_path: Path | None = None
    canonical_map_path: Path | None = None
    sample: bool = False
    sample_user_pct: int = 15
    seed: int = 42

    @property
    def reviews_path(self) -> Path:
        return Path(self.data_dir) / REVIEWS_FILE

    @property
    def meta_path(self) -> Path:
        return Path(self.data_dir) / META_FILE

    @property
    def tag(self) -> str:
        """Short name of this configuration, used as the cache sub-folder."""
        parts = [
            f"sample{self.sample_user_pct}" if self.sample else "full",
            f"from{self.window_start or 'all'}",
            f"cut{self.cutoff}",
            f"u{self.min_user_reviews}b{self.min_book_reviews}",
        ]
        if self.canonical_map_path:
            parts.append("canon")
        return "_".join(parts)

    @property
    def run_dir(self) -> Path:
        """Cache folder for this configuration (created on first access)."""
        base = Path(self.cache_dir) if self.cache_dir else Path(self.data_dir).parent / "cache" / "graph"
        path = base / self.tag
        path.mkdir(parents=True, exist_ok=True)
        return path

    def with_(self, **kwargs) -> "Config":
        """Return a copy with some fields changed."""
        return replace(self, **kwargs)

    def as_dict(self) -> dict:
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}


def sample_config(**kwargs) -> Config:
    """Config for the quick sample run (15% of users, lower book threshold so enough books survive)."""
    return Config(sample=True, min_book_reviews=10, **kwargs)
