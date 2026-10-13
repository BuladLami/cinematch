"""
ui/state.py -- Shared, cached data loading for the Streamlit pages.

Streamlit re-runs the whole script on every click.  Without caching, the
catalogue would be re-read and the TF-IDF model rebuilt on every interaction.

Two caches are used:

  @st.cache_data      for the movie table (a value that gets copied)
  @st.cache_resource  for the recommender  (one heavy object shared by all pages)

Both are keyed on a "file signature" (size + last-modified time) of
data/catalogue.csv, so if the catalogue is ever rebuilt, fresh data is loaded
automatically.  My List is NOT cached: it is a tiny file that changes with
every click, so it is simply read whenever a page needs it.

The collaborative model (collaborative.py) caches its MovieLens half the same
way; the CineMatch accounts' half is rebuilt on every call, so the model
picks up what people did a moment ago.
"""

from __future__ import annotations

import streamlit as st

from .. import accounts, activity, collaborative, config, file_manager, recommender


def catalogue_signature() -> tuple:
    """A small value that changes whenever the catalogue file is (re)built."""
    if not file_manager.catalogue_is_current():
        with st.status("Building the movie catalogue (first run only)...") as status:
            file_manager.build_catalogue(progress=status.write)
            status.update(label="Catalogue ready", state="complete")
    stat = config.CATALOGUE_CSV.stat()
    return (stat.st_size, stat.st_mtime_ns)


@st.cache_data(show_spinner="Loading the catalogue...")
def load_movies(signature: tuple):
    """READ + PROCESS the catalogue.  Cached until the file changes."""
    return file_manager.load_prepared_movies()


@st.cache_resource(show_spinner="Preparing recommendations...")
def load_recommender(signature: tuple, _movies):
    """Build (or reuse the cached) TF-IDF model.  One copy for all pages."""
    return recommender.build_recommender(_movies)


@st.cache_resource(show_spinner="Learning from viewers' ratings...")
def load_movielens(signature: tuple, _movies):
    """The MovieLens ratings, mapped onto the catalogue.  Read once."""
    return collaborative.load_movielens(_movies)


def behaviour_model(movies, exclude: str | None = None) -> collaborative.BehaviourModel:
    """The collaborative model: MovieLens viewers + every account except "exclude"."""
    profiles = {name: activity.profile_for(name, movies)[0]
                for name in accounts.read_users()["username"] if name != exclude}
    return collaborative.build_model(
        len(movies), load_movielens(catalogue_signature(), movies),
        collaborative.app_user_ratings(profiles, movies))


def current_user() -> dict | None:
    """The signed-in account ({"username", "role"}) or None."""
    return st.session_state.get("user")


def username() -> str | None:
    user = current_user()
    return user["username"] if user else None


def is_admin() -> bool:
    user = current_user()
    return bool(user and user["role"] == "admin")


def get_data():
    """The one call every page uses: returns (movies, report, recommender)."""
    signature = catalogue_signature()
    movies, report = load_movies(signature)
    engine = load_recommender(signature, movies)
    return movies, report, engine
