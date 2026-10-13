"""
collaborative.py -- "People who liked your movies also liked..." (collaborative filtering).

The TF-IDF model (recommender.py) knows what a movie is ABOUT.  This model
knows nothing about the movies themselves; it learns only from BEHAVIOUR:
which movies the same people go for.  Two films are close here when the same
viewers like both, even if they share no genre, actor or keyword.

WHAT IT LEARNS FROM
    1. MovieLens ratings (movies data/ratings_small.csv): 100,000 ratings by
       ~670 real viewers.  links_small.csv maps MovieLens ids to TMDB ids.
       Each rating is centred on that viewer's own average, so +1.5 means
       "liked it more than they usually like things" (a harsh critic's 3.5
       and a generous viewer's 4.5 can mean the same thing).
    2. Every CineMatch account -- users and admins alike.  The interest
       scores activity.py learns from what each person searches, opens, picks
       and saves become one more "viewer", so the model keeps learning as
       people use the site.  The person being recommended to is left out, so
       their own clicks cannot simply echo back at them.

HOW IT RECOMMENDS  (item-item cosine similarity)
    The behaviour is a matrix with one row per catalogue movie and one column
    per viewer.  Each row is scaled to length 1.  A user's taste is the
    weighted sum of the rows of the movies they showed interest in, and every
    movie is scored by its cosine similarity to that taste.  Movies nobody
    has rated score 0, so the TF-IDF part still covers them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from . import config, file_manager

# An app user's interest scores are rescaled into the same range as a
# MovieLens viewer's centred ratings: their strongest movie becomes +2, and a
# movie they rated low on My List becomes -2.
APP_USER_SCALE = 2.0


class BehaviourModel:
    """The item x viewer matrix, ready to score a taste profile."""

    def __init__(self, item_user: sparse.csr_matrix, info: dict):
        norms = np.sqrt(np.asarray(item_user.multiply(item_user).sum(axis=1)).ravel())
        self.has_data = norms > 0
        norms[~self.has_data] = 1.0
        self.matrix = sparse.diags(1.0 / norms) @ item_user
        self.matrix = self.matrix.tocsr()
        self.info = info

    def scores(self, rows: list[int], weights: list[float]) -> np.ndarray | None:
        """Cosine similarity of every movie to the weighted taste; None if nothing is known."""
        usable = [(r, w) for r, w in zip(rows, weights)
                  if 0 <= r < len(self.has_data) and self.has_data[r] and w > 0]
        if not usable:
            return None
        picked, w = zip(*usable)
        taste = np.asarray(self.matrix[list(picked)].T @ np.asarray(w, dtype=float)).ravel()
        length = np.linalg.norm(taste)
        if length == 0:
            return None
        return np.asarray(self.matrix @ (taste / length)).ravel()

    def known(self, rows: list[int]) -> int:
        """How many of these movies the model has behaviour for."""
        return int(sum(1 for r in rows if 0 <= r < len(self.has_data) and self.has_data[r]))


# ---------------------------------------------------------------------------
# TRAINING DATA
# ---------------------------------------------------------------------------

def load_movielens(movies: pd.DataFrame) -> pd.DataFrame:
    """READ the MovieLens ratings as (row, viewer, value) for catalogue movies.

    row is the movie's position in the catalogue (the same numbering the
    TF-IDF matrix uses); value is the rating minus that viewer's average.
    Missing files simply mean no MovieLens part.
    """
    empty = pd.DataFrame(columns=["row", "viewer", "value"])
    if not (config.RATINGS_CSV.exists() and config.LINKS_CSV.exists()):
        return empty
    ratings = file_manager.read_csv_file(config.RATINGS_CSV,
                                         usecols=["userId", "movieId", "rating"])
    links = file_manager.read_csv_file(config.LINKS_CSV, usecols=["movieId", "tmdbId"])
    tmdb = pd.to_numeric(links.set_index("movieId")["tmdbId"], errors="coerce")

    position = {int(movie_id): n for n, movie_id in enumerate(movies["id"])}
    ratings["row"] = ratings["movieId"].map(tmdb).map(position)
    ratings = ratings.dropna(subset=["row"])
    ratings["value"] = ratings["rating"] - ratings.groupby("userId")["rating"].transform("mean")
    ratings = ratings[ratings["value"] != 0]
    return pd.DataFrame({"row": ratings["row"].astype(int),
                         "viewer": "ml_" + ratings["userId"].astype(str),
                         "value": ratings["value"].astype(float)})


def app_user_ratings(profiles: dict[str, pd.DataFrame], movies: pd.DataFrame) -> pd.DataFrame:
    """Turn every CineMatch account's learned interest scores into (row, viewer, value)."""
    position = {int(movie_id): n for n, movie_id in enumerate(movies["id"])}
    parts = []
    for name, profile in profiles.items():
        if profile.empty:
            continue
        top = float(profile["score"].max()) or 1.0
        value = (profile["score"] / top * APP_USER_SCALE).where(~profile["disliked"],
                                                                 -APP_USER_SCALE)
        part = pd.DataFrame({"row": profile["movie_id"].map(position),
                             "viewer": "app_" + name, "value": value})
        parts.append(part[part["value"] != 0].dropna(subset=["row"]))
    if not parts:
        return pd.DataFrame(columns=["row", "viewer", "value"])
    combined = pd.concat(parts, ignore_index=True)
    combined["row"] = combined["row"].astype(int)
    return combined


def build_model(n_movies: int, *parts: pd.DataFrame) -> BehaviourModel:
    """Stack the (row, viewer, value) parts into the item x viewer matrix."""
    data = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(
        len(p) for p in parts) else pd.DataFrame(columns=["row", "viewer", "value"])
    viewers = pd.Index(data["viewer"].unique())
    matrix = sparse.csr_matrix(
        (data["value"].astype(float).to_numpy(),
         (data["row"].astype(int).to_numpy(), viewers.get_indexer(data["viewer"]))),
        shape=(n_movies, max(len(viewers), 1)))
    info = {
        "viewers": len(viewers),
        "movielens_viewers": int(viewers.str.startswith("ml_").sum()),
        "app_viewers": int(viewers.str.startswith("app_").sum()),
        "ratings": len(data),
        "movies_covered": int((np.diff(matrix.indptr) > 0).sum()),
    }
    return BehaviourModel(matrix, info)
