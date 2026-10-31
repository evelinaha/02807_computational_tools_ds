"""Book metadata: titles, genre labels (for NMI) and an approximate series key.

The metadata file has no author field (``brand`` is empty for every book), so the
"same-author" ablation from the project plan is replaced by a *same-series* ablation
that uses a series key parsed from the title, e.g.
``"Kiss of Fire (Dragonfire, Book 1)" -> "dragonfire"``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .config import Config

# Kindle eBooks level-3 labels use the same names as Books level-2, except this one.
GENRE_ALIASES = {"Children's eBooks": "Children's Books"}

_SERIES_CUES = re.compile(
    r"\b(?:book|books|series|trilogy|quartet|saga|chronicles?|novels?|mysteries|mystery|volume|vol|no|part|cycle)\b|#\s*\d",
    re.IGNORECASE,
)
# Parentheticals that are formats, imprints or editions, not series.
_NOT_SERIES = re.compile(
    r"audio|\bcd\b|edition|unabridged|abridged|paperback|hardcover|classics?|library|presents|publications|"
    r"penguin|harlequin|signet|bantam|vintage|norton|oxford|dover|reprint|large print|illustrated|annotated|"
    r"berkley|kimani|sensation|oprah|book club|guide|for dummies|study|workbook|circle|"
    r"p\.s\.|sermon|^volume\s*\d+$|^vol\.?\s*\d+$|^book\s*\d+$|^part\s*\d+$|^no\.?\s*\d+$",
    re.IGNORECASE,
)
_NUMBERING = re.compile(
    r"\b(book|volume|vol|no|number|part|episode)\.?\s*(\d+|[ivxlc]+|one|two|three|four|five|six|seven|eight|nine|ten)\b"
    r"(\s+of\s+(the\s+)?)?|#\s*\d+(\.\d+)?|\b\d+\b",
    re.IGNORECASE,
)
_FILLER = re.compile(r"\b(a|an|the|novel|novels|series|trilogy|quartet|saga|book|books|mystery|mysteries|of)\b", re.IGNORECASE)


def _normalise_series(text: str) -> str:
    text = _NUMBERING.sub(" ", text)
    text = _FILLER.sub(" ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def series_candidate(title: str | None) -> str | None:
    """Raw series key for one title, or ``None``.

    Looks at the last parenthetical ("Steel World (Undying Mercenaries) (Volume 1)" uses
    "Undying Mercenaries" because pure "Volume 1" is skipped) and at "Title: Series #5".
    """
    if not isinstance(title, str) or not title:
        return None
    for frag in reversed(re.findall(r"\(([^()]{2,80})\)", title)):
        if _NOT_SERIES.search(frag.strip()):
            continue
        key = _normalise_series(frag)
        if len(key) >= 3:
            return key
    m = re.search(r":\s*([^:()]{3,60}?)\s*#\s*\d+", title)
    if m:
        key = _normalise_series(m.group(1))
        if len(key) >= 3:
            return key
    return None


def series_keys(titles: pd.Series, max_books_without_cue: int = 30) -> pd.Series:
    """Series key per book (index preserved), ``<NA>`` when none.

    A parenthetical without a series cue word ("Carpathian", "Little House") is accepted only
    if 2..``max_books_without_cue`` books share it - this keeps real series and drops one-off
    subtitles and large imprints.
    """
    raw = titles.map(series_candidate)
    has_cue = titles.fillna("").str.contains(_SERIES_CUES)
    size = raw.map(raw.value_counts())
    ok = raw.notna() & (size >= 2) & (has_cue | (size <= max_books_without_cue))
    return raw.where(ok).astype("string")


def parse_genres(categories: Iterable) -> list[str]:
    """Second-level genres of one book: ``Books > X`` and ``Kindle Store > Kindle eBooks > X``."""
    out: set[str] = set()
    if categories is None:
        return []
    for path in categories:
        path = list(path)
        if len(path) >= 2 and path[0] == "Books":
            out.add(path[1])
        elif len(path) >= 3 and path[0] == "Kindle Store" and path[1] == "Kindle eBooks":
            out.add(path[2])
    return sorted(GENRE_ALIASES.get(g, g) for g in out)


def load_book_metadata(cfg: Config, item_ids: Iterable[str]) -> pd.DataFrame:
    """Metadata for the given books: ``item_id, title, description, genres, n_genres,
    primary_genre, series``.

    ``primary_genre``: for a book with several genres we take the *least frequent* one (counted
    over the given books), because the frequent labels ("Literature & Fiction") are attached to
    almost everything and would hide the specific genre. Books without a genre get ``<NA>``.
    """
    ids = pd.Index(pd.unique(np.asarray(list(item_ids))), name="item_id")
    meta = pd.read_parquet(cfg.meta_path, columns=["item_id", "title", "description", "categories"])
    meta = meta[meta.item_id.isin(ids)].set_index("item_id").reindex(ids)
    meta["title"] = meta["title"].replace("", pd.NA)
    meta["genres"] = meta["categories"].map(parse_genres)
    meta["n_genres"] = meta["genres"].map(len)
    freq = pd.Series([g for gs in meta["genres"] for g in gs]).value_counts()
    meta["primary_genre"] = meta["genres"].map(
        lambda gs: min(gs, key=lambda g: (freq[g], g)) if gs else pd.NA).astype("string")
    meta["series"] = series_keys(meta["title"])
    return meta.drop(columns="categories").reset_index()
