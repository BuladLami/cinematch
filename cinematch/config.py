"""
config.py -- All the settings and file paths CineMatch uses.

Keeping paths and constants in one small module means the rest of the program
never has to hard-code a filename.  If a file needs to move, it moves here.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# FOLDERS AND FILES
# ---------------------------------------------------------------------------

# Project root = the folder that contains this "cinematch" package.
BASE_DIR = Path(__file__).resolve().parent.parent

# The SOURCE dataset ("The Movies Dataset", TMDB + MovieLens).  Everything in
# this folder is opened for READING ONLY -- CineMatch never writes into it.
SOURCE_DIR = BASE_DIR / "movies data"
METADATA_CSV = SOURCE_DIR / "movies_metadata.csv"   # titles, genres, overview, posters
CREDITS_CSV = SOURCE_DIR / "credits.csv"            # full cast and crew per movie
KEYWORDS_CSV = SOURCE_DIR / "keywords.csv"          # plot keywords per movie
RATINGS_CSV = SOURCE_DIR / "ratings_small.csv"      # 100k MovieLens viewer ratings
LINKS_CSV = SOURCE_DIR / "links_small.csv"          # MovieLens id -> TMDB id

# Everything the program creates lives inside "data/".
DATA_DIR = BASE_DIR / "data"

# The CATALOGUE: the three source files merged, filtered and flattened into one
# small CSV.  Built automatically on the first run (it takes a little while,
# because credits.csv alone is ~190 MB) and simply read on every run after.
CATALOGUE_CSV = DATA_DIR / "catalogue.csv"

# The user's own list of movies.  This is the file the CRUD actions change:
# add a movie, read the list, update its status / rating / note, remove it.
MY_LIST_CSV = DATA_DIR / "my_list.csv"

# The saved TF-IDF model, so it is not recomputed on every single run.
CACHE_DIR = DATA_DIR / "cache"
TFIDF_CACHE = CACHE_DIR / "tfidf_cache.pkl"

# ACCOUNTS.  A database with one table, "users": one row per person who can
# sign in (passwords are stored hashed).  Without a DATABASE_URL (see
# database_url() below) it is this SQLite file: built into Python, nothing to
# install -- but it only exists on this one computer.
DATABASE = DATA_DIR / "cinematch.db"

# Every user gets a folder of their own: data/users/<username>/
#     my_list.csv          their My List
#     activity.csv         what they search, open and pick -- CineMatch learns
#                          their watch pattern from it (see activity.py)
USERS_DIR = DATA_DIR / "users"

# One line per sign-in, account change and admin action -- the admin reads this.
ACTIVITY_LOG_CSV = DATA_DIR / "activity_log.csv"

ALL_DIRS = (DATA_DIR, CACHE_DIR, USERS_DIR)


def user_dir(username: str) -> Path:
    return USERS_DIR / username


def my_list_path(username: str | None = None) -> Path:
    """The user's My List file; without a username, the original shared file."""
    return user_dir(username) / "my_list.csv" if username else MY_LIST_CSV


def activity_path(username: str) -> Path:
    return user_dir(username) / "activity.csv"


def ensure_dirs() -> None:
    """Create the data folders if they do not exist yet (safe to call often)."""
    for folder in ALL_DIRS:
        folder.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# WHICH MOVIES MAKE THE CATALOGUE
# ---------------------------------------------------------------------------
# The source has ~45,000 movies, most of them obscure with a handful of votes.
# Keeping released films with at least this many votes, a plot summary and a
# poster leaves ~9,000 titles people will actually recognise.
MIN_VOTES = 50

CAST_SIZE = 5            # top-billed actors kept per movie

# Multi-value fields (genres, keywords, cast, director) are stored in the
# catalogue joined with this character, since a name can contain a comma.
LIST_SEP = "|"

# ---------------------------------------------------------------------------
# DATASET COLUMNS
# ---------------------------------------------------------------------------
CATALOGUE_COLUMNS = [
    "id",                 # TMDB movie id -> the primary key
    "imdb_id",
    "title",
    "original_title",
    "genres",             # "Animation|Comedy|Family"
    "keywords",           # "jealousy|toy|boy|friendship"
    "cast",               # "Tom Hanks|Tim Allen|Don Rickles"
    "director",           # "John Lasseter"  (several joined with LIST_SEP)
    "collection",         # "Toy Story Collection" -> links sequels together
    "overview",           # short plot summary (free text)
    "tagline",
    "release_date",       # YYYY-MM-DD
    "runtime",            # minutes
    "vote_average",       # 0-10 rating
    "vote_count",
    "popularity",
    "original_language",
    "poster_path",        # "/rhIRbceoE9lR4veEXuwCC2wARtG.jpg" on image.tmdb.org
]

# Numeric columns, listed so missing values can be filled with numbers.
NUMERIC_COLUMNS = ["runtime", "vote_average", "vote_count", "popularity"]

# ---------------------------------------------------------------------------
# RECOMMENDATION SETTINGS
# ---------------------------------------------------------------------------
# How many times each field is repeated inside the "soup".  Repeating a field
# makes TF-IDF treat it as more important.  Genres, keywords and the collection
# describe the "kind" of movie best, so they get the most weight.
FEATURE_WEIGHTS = {
    "genres": 3,
    "keywords": 3,
    "collection": 3,
    "cast": 2,
    "director": 2,
    "overview": 1,
    "tagline": 1,
}

MAX_TFIDF_FEATURES = 30000   # vocabulary cap -> keeps the model small

# Below this many movies, the "ignore words used by only one movie" rule is
# switched off -- on a tiny test table it would delete the whole vocabulary.
MIN_ROWS_FOR_MIN_DF = 50
DEFAULT_TOP_N = 5
TOP_N_CHOICES = [5, 10]

# ---------------------------------------------------------------------------
# MY LIST
# ---------------------------------------------------------------------------
LIST_STATUSES = ["Want to watch", "Watching", "Watched"]
MY_LIST_COLUMNS = ["movie_id", "title", "status", "my_rating", "notes",
                   "added_on", "updated_on"]

# A movie on the list counts as "liked" (and steers "Recommended for you")
# unless it was rated below this.  Unrated movies count as liked.
LIKED_RATING = 6

# ---------------------------------------------------------------------------
# ACCOUNTS
# ---------------------------------------------------------------------------
ROLES = ["user", "admin"]
LOG_COLUMNS = ["timestamp", "username", "action", "details"]

# Created automatically the first time the app runs, so someone can sign in
# and set up the other accounts.  Change this password after the first login.
DEFAULT_ADMIN = ("admin", "admin123")


def default_admin_password() -> str:
    """The first admin's password: CINEMATCH_ADMIN_PASSWORD if it is set, else admin123.

    A public deployment (Streamlit Community Cloud) sets it in the app's
    secrets, so the admin never starts with the well-known password -- which
    matters there, because the server's disk is wiped on every restart and the
    first admin is created again each time.
    """
    return _setting("CINEMATCH_ADMIN_PASSWORD") or DEFAULT_ADMIN[1]


def _setting(name: str) -> str | None:
    """A setting from the environment, else from .streamlit/secrets.toml, else None."""
    value = os.environ.get(name)
    if not value:
        try:
            import streamlit as st
            value = st.secrets.get(name)
        except Exception:          # no secrets file, or not running under Streamlit
            value = None
    return str(value) if value else None


def database_url() -> str | None:
    """The online PostgreSQL database the accounts are kept in, if one is set.

    DATABASE_URL, from the environment or the app's secrets, looks like
        postgresql://user:password@host/dbname?sslmode=require
    (a free database from neon.tech or supabase.com gives one).  With it, every
    device and every deployment signs in against the same accounts.  Without
    it, accounts stay in the local SQLite file, DATABASE.
    """
    return _setting("DATABASE_URL")


MIN_PASSWORD_LENGTH = 6
PASSWORD_ITERATIONS = 100_000      # PBKDF2 rounds: slow to guess, quick to check once

# ---------------------------------------------------------------------------
# LEARNING FROM ACTIVITY  (see activity.py)
# ---------------------------------------------------------------------------
ACTIVITY_COLUMNS = ["timestamp", "event", "movie_id", "title", "detail"]

# How much one action says about a user's taste.  Opening a movie is a passing
# interest; searching for it by name is deliberate; asking for "movies like
# this" says "I want more of this".  A free-text search ("batman") is spread
# over the top TEXT_SEARCH_HITS titles it finds, at the "search_text" weight.
EVENT_WEIGHTS = {"view": 1.0, "search": 2.0, "search_text": 0.5, "pick": 3.0,
                 "genre": 1.0}
TEXT_SEARCH_HITS = 3

# Putting a movie on My List is the strongest signal.  The weight is scaled by
# the user's rating (rating / 7), and a movie rated below LIKED_RATING counts
# as disliked: it is kept out of the pattern and never recommended.
LIST_STATUS_WEIGHTS = {"Want to watch": 4.0, "Watching": 5.0, "Watched": 5.0}

# Tastes drift, so an action loses half its weight every HALF_LIFE_DAYS.
HALF_LIFE_DAYS = 30

# Opening the same movie twenty times should not drown out everything else.
MAX_EVENT_SCORE = 10.0

# Below this many movies the pattern is still shown, with a note that it will
# sharpen as the user keeps browsing.
MIN_MOVIES_FOR_PATTERN = 5

# A pattern only counts as a pattern when it repeats: a director seen once is
# not "a director you keep coming back to".
MIN_REPEAT = 2

APP_NAME = "CineMatch"
APP_TAGLINE = "Find your next favourite movie"
