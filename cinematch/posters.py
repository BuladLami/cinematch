"""
posters.py -- Movie poster images from TMDB.

The catalogue already carries a "poster_path" for every movie (for example
"/rhIRbceoE9lR4veEXuwCC2wARtG.jpg"), and TMDB's image server turns that into a
picture with no key and no sign-in:

    https://image.tmdb.org/t/p/w342/rhIRbceoE9lR4veEXuwCC2wARtG.jpg

The dataset was captured in 2017, though, and TMDB has since replaced many of
those image files, so an old path can return "not found".  The page copes on
its own -- the drawn poster art shows through wherever an image fails -- but
the paths can also be refreshed from TMDB's metadata API, which needs a free
key.  Fresh paths are saved to data/cache/poster_paths.json and take priority
over the dataset's own from then on.

Refreshing (optional, once):
    1. get a free key at https://www.themoviedb.org/settings/api
    2. put it in the TMDB_API_KEY environment variable or data/tmdb_key.txt
    3. run:   python -m cinematch.posters

This product uses the TMDB API but is not endorsed or certified by TMDB.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from . import config

POSTER_CACHE = config.CACHE_DIR / "poster_paths.json"
KEY_FILE = config.DATA_DIR / "tmdb_key.txt"

API_URL = "https://api.themoviedb.org/3/movie/{movie_id}?api_key={key}"
IMAGE_BASE = "https://image.tmdb.org/t/p/"

# TMDB serves several widths.  w342 is sharp enough for a poster tile without
# downloading a full-size image for every card on the page.
GRID_SIZE = "w342"
LARGE_SIZE = "w500"

REQUEST_TIMEOUT = 10        # seconds before one lookup is given up on
MAX_WORKERS = 8             # parallel lookups; TMDB is fine with this


class PosterError(Exception):
    """Raised when posters cannot be refreshed (no key, key rejected)."""


# ---------------------------------------------------------------------------
# THE REFRESHED-PATH CACHE
# ---------------------------------------------------------------------------
# {"862": "/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg", "285": null}
# null means "TMDB was asked and has no poster", so it is not asked again.
_cache: dict[str, str | None] | None = None


def load_cache() -> dict[str, str | None]:
    """Read the refreshed poster paths (once per process, then kept in memory)."""
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(POSTER_CACHE.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            _cache = {}
    return _cache


def save_cache(cache: dict[str, str | None]) -> None:
    config.ensure_dirs()
    POSTER_CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")


# ---------------------------------------------------------------------------
# READING (what the pages call -- never touches the network)
# ---------------------------------------------------------------------------
def poster_url(movie, size: str = GRID_SIZE) -> str | None:
    """The image URL for a movie row: a refreshed path if there is one, else the dataset's."""
    movie_id = str(movie.get("id", "")).split(".")[0]
    path = load_cache().get(movie_id) or str(movie.get("poster_path", "") or "")
    return f"{IMAGE_BASE}{size}{path}" if path.startswith("/") else None


# ---------------------------------------------------------------------------
# REFRESHING (a one-off command, run from the terminal)
# ---------------------------------------------------------------------------
def find_api_key() -> str | None:
    """Look for a TMDB key in the environment or data/tmdb_key.txt."""
    key = os.environ.get("TMDB_API_KEY", "").strip()
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text(encoding="utf-8").strip()
    return key or None


def _fetch_one(movie_id: str, key: str) -> tuple[str, str | None]:
    """Ask TMDB for one movie's current poster path.  Never raises for a single film."""
    try:
        with urllib.request.urlopen(API_URL.format(movie_id=movie_id, key=key),
                                    timeout=REQUEST_TIMEOUT) as response:
            return movie_id, json.loads(response.read().decode("utf-8")).get("poster_path")
    except urllib.error.HTTPError as error:
        if error.code == 401:
            raise PosterError("TMDB rejected the API key (HTTP 401).") from error
        return movie_id, None
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError):
        return movie_id, None


def refresh_posters(movie_ids, progress=None) -> dict:
    """Ask TMDB for the current poster of every movie not refreshed yet."""
    key = find_api_key()
    if not key:
        raise PosterError("No TMDB API key found (TMDB_API_KEY or data/tmdb_key.txt).")

    cache = load_cache()
    todo = list(dict.fromkeys(str(i) for i in movie_ids if str(i) not in cache))
    found = 0
    # Each lookup is almost entirely waiting on the network, so a thread pool
    # turns an hour of one-at-a-time requests into a few minutes.
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for done, (movie_id, path) in enumerate(pool.map(lambda i: _fetch_one(i, key), todo), 1):
            cache[movie_id] = path
            found += bool(path)
            if progress and (done % 100 == 0 or done == len(todo)):
                progress(done, len(todo))
                save_cache(cache)          # keep what we have if interrupted
    save_cache(cache)
    return {"requested": len(todo), "found": found}


if __name__ == "__main__":
    from . import file_manager

    ids = file_manager.read_catalogue()["id"].tolist()
    try:
        report = refresh_posters(ids, progress=lambda d, t: print(f"  {d:,} / {t:,}"))
    except PosterError as error:
        sys.exit(str(error))
    print(f"Refreshed {report['found']:,} of {report['requested']:,} posters.")
