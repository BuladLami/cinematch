"""
recommender.py -- The machine-learning part of CineMatch.

CONTENT-BASED FILTERING in three steps
--------------------------------------
1. Every movie is described by one block of text (the "soup" built in
   preprocessing.py) containing its genres, keywords, cast, director, overview
   and tagline.

2. TF-IDF (Term Frequency - Inverse Document Frequency) turns each block of
   text into a vector of numbers.  A word scores high for a movie when it
   appears often in THAT movie but rarely across all the others -- so common
   words like "the" score near zero, while a distinctive word like
   "sciencefiction" or "christophernolan" scores high.

3. COSINE SIMILARITY measures the angle between two movie vectors.
   The result is between 0 and 1:
        1.0  = identical wording
        0.0  = no words in common
   The movies with the highest cosine similarity to the one you picked are the
   recommendations.

Why this model and not a neural network?  Because it is fast, needs no training
labels, works from the catalogue alone, and every score can be explained by
pointing at the shared words.  For a ~9,000 movie catalogue it is the right tool.

"RECOMMENDED FOR YOU" uses the same model: the vectors of the movies on the
user's list are averaged into one "taste" vector, and the catalogue is ranked
against that instead of against a single film.

CACHING
-------
Building the TF-IDF matrix takes a few seconds, so it is saved to
data/cache/tfidf_cache.pkl together with a "fingerprint" of the data.  On the
next run the cache is reused, unless a movie was added, edited or deleted -- the
fingerprint changes and the model is rebuilt automatically.
"""

from __future__ import annotations

import hashlib
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config, preprocessing


class RecommenderError(Exception):
    """Raised when a recommendation cannot be produced."""


def data_fingerprint(df: pd.DataFrame) -> str:
    """Make a short signature of the text the model was trained on.

    If a single movie's soup changes, this string changes, which tells the
    program that the cached model is stale and must be rebuilt.
    """
    joined = "|".join(df["soup"].astype(str).tolist())
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


class MovieRecommender:
    """Holds the movie table, the TF-IDF model and the search/recommend logic."""

    def __init__(self, movies: pd.DataFrame):
        if "soup" not in movies.columns:
            raise RecommenderError(
                "The movie table has not been preprocessed "
                "(no 'soup' column). Call preprocessing.prepare() first."
            )
        self.movies = movies.reset_index(drop=True)
        self.fingerprint = data_fingerprint(self.movies)
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.loaded_from_cache = False

    # -- model building ----------------------------------------------------

    def fit(self) -> "MovieRecommender":
        """Learn the vocabulary and build the TF-IDF matrix (rows = movies).

        min_df=2 drops words that only one movie uses, because a word that
        appears exactly once cannot make two movies similar -- it only makes the
        model bigger.  On a very small dataset (an imported file with a handful
        of movies) that rule would throw away everything, so it is relaxed to 1.
        """
        documents = self.movies["soup"].astype(str)
        min_df = 2 if len(self.movies) >= config.MIN_ROWS_FOR_MIN_DF else 1

        for attempt_min_df in (min_df, 1):
            self.vectorizer = TfidfVectorizer(
                stop_words="english",             # ignore "the", "and", "of", ...
                max_features=config.MAX_TFIDF_FEATURES,   # keep the 20k best words
                min_df=attempt_min_df,
                sublinear_tf=True,                # soften very repetitive words
            )
            try:
                self.matrix = self.vectorizer.fit_transform(documents)
                self.loaded_from_cache = False
                return self
            except ValueError:
                continue      # no words survived; retry with the looser setting

        raise RecommenderError(
            "The dataset has no usable text, so no model can be built. "
            "Check that the genres, keywords and overview columns are filled in."
        )

    def save_cache(self, path=None) -> None:
        """Save the fitted model so the next run does not have to refit."""
        path = path or config.TFIDF_CACHE
        config.ensure_dirs()
        payload = {
            "fingerprint": self.fingerprint,
            "vectorizer": self.vectorizer,
            "matrix": self.matrix,
            "titles": self.movies["title"].tolist(),
        }
        with open(path, "wb") as handle:
            pickle.dump(payload, handle)

    def try_load_cache(self, path=None) -> bool:
        """Reuse the saved model if it matches the current data.

        Returns True when the cache was used, False when a rebuild is needed.
        """
        path = path or config.TFIDF_CACHE
        if not path.exists():
            return False
        try:
            with open(path, "rb") as handle:
                payload = pickle.load(handle)
        except (pickle.UnpicklingError, EOFError, AttributeError, ModuleNotFoundError):
            return False        # corrupt or written by another version -> refit

        if payload.get("fingerprint") != self.fingerprint:
            return False        # the data changed -> refit
        if payload["matrix"].shape[0] != len(self.movies):
            return False        # row count mismatch -> refit

        self.vectorizer = payload["vectorizer"]
        self.matrix = payload["matrix"]
        self.loaded_from_cache = True
        return True

    def ensure_model(self, use_cache: bool = True) -> "MovieRecommender":
        """Load the cached model, or build and cache a new one."""
        if use_cache and self.try_load_cache():
            return self
        self.fit()
        if use_cache:
            self.save_cache()
        return self

    # -- searching ---------------------------------------------------------

    def search(self, query: str, limit: int = 25) -> pd.DataFrame:
        """Find movies whose title contains the typed text (case-insensitive).

        Results are sorted so that titles STARTING with the query come first,
        then by popularity, which puts the obvious match at the top.
        """
        query = str(query).strip().lower()
        if not query:
            return self.movies.head(limit)

        titles = self.movies["title"].astype(str).str.lower()
        hits = self.movies[titles.str.contains(query, regex=False)].copy()
        if hits.empty:
            return hits

        hits["_starts"] = (~titles[hits.index].str.startswith(query)).astype(int)
        sort_columns = ["_starts"] + (["popularity"] if "popularity" in hits else [])
        hits = hits.sort_values(sort_columns, ascending=[True, False][: len(sort_columns)])
        return hits.drop(columns="_starts").head(limit)

    def find_index(self, title: str) -> int:
        """Return the row number of an exact title, or raise a clear error."""
        titles = self.movies["title"].astype(str).str.lower()
        matches = self.movies.index[titles == str(title).strip().lower()]
        if len(matches) == 0:
            raise RecommenderError(f'"{title}" is not in the dataset.')
        return int(matches[0])

    def index_of_id(self, movie_id) -> int | None:
        """Row number of a movie id, or None if it is not in the catalogue.

        Titles are not unique (there are several "Hamlet"s), so anything that
        must point at exactly one film uses the id.
        """
        if not hasattr(self, "_id_lookup"):
            self._id_lookup = {int(i): n for n, i in enumerate(self.movies["id"])}
        try:
            return self._id_lookup.get(int(movie_id))
        except (TypeError, ValueError):
            return None

    def get_movie(self, index: int) -> pd.Series:
        """Return one movie row by its position in the table."""
        if index < 0 or index >= len(self.movies):
            raise RecommenderError(f"Row {index} is outside the dataset.")
        return self.movies.iloc[index]

    # -- recommending ------------------------------------------------------

    def recommend(self, index: int, top_n: int = config.DEFAULT_TOP_N) -> pd.DataFrame:
        """Return the top_n most similar movies to the one at "index".

        Only ONE row of the similarity matrix is computed -- the chosen movie
        against all the others.  The full 9,000 x 9,000 matrix would need about
        650 MB of memory, and we never need more than one row at a time.
        """
        if self.matrix is None:
            self.ensure_model()
        if index < 0 or index >= self.matrix.shape[0]:
            raise RecommenderError(f"Row {index} is outside the model.")

        # cosine_similarity returns a 1 x n_movies array; [0] takes that row.
        scores = cosine_similarity(self.matrix[index], self.matrix)[0]

        ranking = pd.Series(scores).sort_values(ascending=False)
        ranking = ranking.drop(index)                 # a movie is not its own match
        ranking = ranking[ranking > 0].head(top_n)    # skip zero-similarity rows

        if ranking.empty:
            return pd.DataFrame(columns=list(self.movies.columns) + ["similarity"])

        results = self.movies.loc[ranking.index].copy()
        results["similarity"] = ranking.values
        results["similarity_pct"] = (ranking.values * 100).round(1)
        return results.reset_index(drop=True)

    def profile_scores(self, indices: list[int], weights: list[float] | None = None):
        """Cosine similarity of EVERY movie to the weighted average of some movies.

        Returns (scores, kept_indices); scores is None when no index is usable.
        The watch-pattern recommender (patterns.py) combines these scores with
        the pattern found in the user's history.
        """
        if self.matrix is None:
            self.ensure_model()
        pairs = [(i, w) for i, w in zip(indices, weights or [1.0] * len(indices))
                 if i is not None and 0 <= i < self.matrix.shape[0]]
        if not pairs:
            return None, []
        indices = [i for i, _ in pairs]
        weights = np.asarray([w for _, w in pairs], dtype=float)
        profile = np.asarray(self.matrix[indices].T @ weights).ravel() / weights.sum()
        return cosine_similarity(profile.reshape(1, -1), self.matrix)[0], indices

    def recommend_for_profile(self, indices: list[int], weights: list[float] | None = None,
                              top_n: int = 12) -> pd.DataFrame:
        """Rank the catalogue against the average of several movies ("your taste").

        weights lets a movie rated 10 pull harder than one rated 6.  The input
        movies themselves are never recommended back.
        """
        scores, indices = self.profile_scores(indices, weights)
        if scores is None:
            return pd.DataFrame(columns=list(self.movies.columns) + ["similarity"])

        ranking = pd.Series(scores).drop(indices).sort_values(ascending=False)
        ranking = ranking[ranking > 0].head(top_n)
        results = self.movies.loc[ranking.index].copy()
        results["similarity"] = ranking.values
        results["similarity_pct"] = (ranking.values * 100).round(1)
        return results.reset_index(drop=True)

    def recommend_by_title(self, title: str, top_n: int = config.DEFAULT_TOP_N) -> pd.DataFrame:
        """Same as recommend(), but takes a movie title instead of a row number."""
        return self.recommend(self.find_index(title), top_n=top_n)

    # -- explaining --------------------------------------------------------

    def shared_terms(self, index_a: int, index_b: int, limit: int = 8) -> list[str]:
        """List the words that made two movies look similar.

        Used on the Recommendations page to show WHY a movie was suggested,
        which turns the score from a mystery number into an explanation.
        """
        if self.matrix is None or self.vectorizer is None:
            return []

        vocabulary = self.vectorizer.get_feature_names_out()
        row_a = self.matrix[index_a].toarray()[0]
        row_b = self.matrix[index_b].toarray()[0]

        overlap = row_a * row_b                       # non-zero only for shared words
        top_positions = overlap.argsort()[::-1][:limit]
        return [vocabulary[p] for p in top_positions if overlap[p] > 0]

    def readable(self, terms: list[str]) -> list[str]:
        """Turn model tokens back into display text: "christophernolan" -> "Christopher Nolan".

        Names and keyword phrases were squashed into single tokens for the
        model (see preprocessing.squash_name); this reverses that for display.
        Plain overview words have no entry and are shown as they are.
        """
        if not hasattr(self, "_labels"):
            self._labels = {}
            for column in ("genres", "keywords", "cast", "director", "collection"):
                if column in self.movies.columns:
                    for value in self.movies[column]:
                        for item in preprocessing.split_list(value):
                            self._labels.setdefault(preprocessing.squash_name(item), item)
        return [self._labels.get(term, term) for term in terms]

    def model_info(self) -> dict:
        """Facts about the fitted model, shown on the Dashboard page."""
        if self.matrix is None:
            return {"fitted": False}
        rows, columns = self.matrix.shape
        return {
            "fitted": True,
            "movies": rows,
            "vocabulary": columns,
            "stored_values": int(self.matrix.nnz),
            "density_pct": round(100 * self.matrix.nnz / (rows * columns), 3),
            "from_cache": self.loaded_from_cache,
        }


def build_recommender(movies: pd.DataFrame, use_cache: bool = True) -> MovieRecommender:
    """Convenience function: create the recommender and make sure it has a model."""
    return MovieRecommender(movies).ensure_model(use_cache=use_cache)
