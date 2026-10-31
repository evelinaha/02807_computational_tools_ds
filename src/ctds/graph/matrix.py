"""Sparse user x book matrices built from the training reviews, with integer index mappings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class Bipartite:
    """User x book incidence matrices (CSR, float32).

    Attributes:
        read: 1 where the user reviewed the book before the cut-off.
        liked: 1 where the review counts as liked (see ``data.add_liked``).
        user_ids / book_ids: the original id of each row / column.
    """

    read: sp.csr_matrix
    liked: sp.csr_matrix
    user_ids: np.ndarray
    book_ids: np.ndarray

    @property
    def n_users(self) -> int:
        return self.read.shape[0]

    @property
    def n_books(self) -> int:
        return self.read.shape[1]

    def book_index(self) -> pd.Series:
        """Map original book id -> column index."""
        return pd.Series(np.arange(self.n_books), index=self.book_ids)

    def user_index(self) -> pd.Series:
        """Map original user id -> row index."""
        return pd.Series(np.arange(self.n_users), index=self.user_ids)

    def save(self, folder: Path) -> None:
        """Write ``bipartite_read.npz``, ``bipartite_liked.npz`` and the two index Parquet files."""
        folder = Path(folder)
        sp.save_npz(folder / "bipartite_read.npz", self.read)
        sp.save_npz(folder / "bipartite_liked.npz", self.liked)
        pd.DataFrame({"user_idx": np.arange(self.n_users), "user_id": self.user_ids}).to_parquet(folder / "user_index.parquet", index=False)
        pd.DataFrame({"book_idx": np.arange(self.n_books), "item_id": self.book_ids}).to_parquet(folder / "book_index.parquet", index=False)

    @classmethod
    def load(cls, folder: Path) -> "Bipartite":
        folder = Path(folder)
        return cls(
            read=sp.load_npz(folder / "bipartite_read.npz").tocsr(),
            liked=sp.load_npz(folder / "bipartite_liked.npz").tocsr(),
            user_ids=pd.read_parquet(folder / "user_index.parquet")["user_id"].to_numpy(),
            book_ids=pd.read_parquet(folder / "book_index.parquet")["item_id"].to_numpy(),
        )

    @classmethod
    def exists(cls, folder: Path) -> bool:
        return (Path(folder) / "bipartite_liked.npz").exists()


def build_bipartite(train: pd.DataFrame) -> Bipartite:
    """Build the user x book matrices from training reviews (columns ``user_id, item_id, liked``).

    Users and books are indexed in sorted id order, so the mapping is reproducible.
    """
    users, u = np.unique(train["user_id"].to_numpy(), return_inverse=True)
    books, b = np.unique(train["item_id"].to_numpy(), return_inverse=True)
    shape = (len(users), len(books))
    ones = np.ones(len(train), dtype=np.float32)
    read = sp.csr_matrix((ones, (u, b)), shape=shape)
    liked_mask = train["liked"].to_numpy()
    liked = sp.csr_matrix((ones[liked_mask], (u[liked_mask], b[liked_mask])), shape=shape)
    return Bipartite(read=read, liked=liked, user_ids=users, book_ids=books)
