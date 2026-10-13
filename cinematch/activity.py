"""
activity.py -- Learning what a user likes from what they DO in CineMatch.

Nobody uploads or fills in anything.  While someone uses the site, a few of
the things they do are written to data/users/<username>/activity.csv, one line
per action:

    search   they searched the catalogue (Discover's search box).  Choosing a
             title records that movie; typed text ("batman") records the text
    view     they opened a movie's details pop-up
    pick     they pressed "Find movies like this" on a movie
    genre    they browsed a genre (Discover's filter, Home's genre pills)

My List is read as it stands rather than copied into the log: adding a movie,
its status and the rating given to it are the strongest signals of all.

build_profile() turns the log + My List into one interest score per movie:

    score =   sum of  EVENT_WEIGHTS[action] x recency   (capped at MAX_EVENT_SCORE)
            + LIST_STATUS_WEIGHTS[status] x rating / 7  (if it is on My List)

recency halves every HALF_LIFE_DAYS, so what someone looked at this week
counts for more than what they looked at in spring.  A movie rated below
config.LIKED_RATING gets a score of 0 and is marked "disliked".

patterns.py then finds the pattern in the scored movies and recommends from it.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from . import config, file_manager

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
EVENTS = ["search", "view", "pick", "genre"]

PROFILE_COLUMNS = ["movie_id", "title", "score", "views", "searches", "picks",
                   "list_status", "my_rating", "disliked", "last_seen"]


# ---------------------------------------------------------------------------
# RECORD
# ---------------------------------------------------------------------------

def record(username: str | None, event: str, movie_id=None, title: str = "",
           detail: str = "", when: datetime | None = None) -> None:
    """APPEND one action to the user's activity log.  Nobody signed in: nothing."""
    if not username or event not in EVENTS:
        return
    row = {
        "timestamp": (when or datetime.now()).strftime(TIME_FORMAT),
        "event": event,
        "movie_id": "" if movie_id is None else int(movie_id),
        "title": str(title or ""),
        "detail": str(detail or "").strip(),
    }
    file_manager.append_csv_row(row, config.activity_path(username), config.ACTIVITY_COLUMNS)


def record_movie(username: str | None, event: str, movie: pd.Series) -> None:
    """record() for a catalogue row."""
    record(username, event, movie_id=movie["id"], title=movie.get("title", ""))


def record_search(username: str | None, query: str, movies: pd.DataFrame) -> None:
    """A search: an exact catalogue title records that movie, anything else the text."""
    query = str(query or "").strip()
    if not query:
        return
    exact = movies[movies["title"].astype(str).str.lower() == query.lower()]
    if len(exact):
        record_movie(username, "search", exact.iloc[0])
    else:
        record(username, "search", detail=query)


# ---------------------------------------------------------------------------
# READ / DELETE
# ---------------------------------------------------------------------------

def read_activity(username: str) -> pd.DataFrame:
    """READ a user's activity log, oldest first (empty if they have done nothing yet)."""
    path = config.activity_path(username)
    if not path.exists():
        return pd.DataFrame(columns=config.ACTIVITY_COLUMNS)
    df = file_manager.read_csv_file(path, dtype=str).fillna("")
    for column in config.ACTIVITY_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df[config.ACTIVITY_COLUMNS].reset_index(drop=True)


def forget_movies(username: str, movie_ids: list[int]) -> str:
    """DELETE every logged action about some movies (My List is left alone)."""
    log = read_activity(username)
    ids = {str(int(i)) for i in movie_ids}
    keep = log[~log["movie_id"].isin(ids)]
    file_manager.write_csv_file(keep, config.activity_path(username))
    return (f"Forgot {len(log) - len(keep)} action(s) about {len(ids)} movie(s). "
            "Movies on My List still count until you remove them there.")


def clear_activity(username: str) -> str:
    """DELETE the whole activity log."""
    config.activity_path(username).unlink(missing_ok=True)
    return "Your activity is cleared. CineMatch will start learning again from scratch."


# ---------------------------------------------------------------------------
# LEARN
# ---------------------------------------------------------------------------

def _recency(stamps: pd.Series, now: datetime) -> pd.Series:
    """1.0 for an action just now, 0.5 after HALF_LIFE_DAYS, 0.25 after twice that."""
    when = pd.to_datetime(stamps, errors="coerce", format="mixed")
    age_days = ((pd.Timestamp(now) - when).dt.total_seconds() / 86400).clip(lower=0)
    return (0.5 ** (age_days / config.HALF_LIFE_DAYS)).fillna(0.0)


def _movie_events(log: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """One row per (action, movie): typed searches are spread over their top hits."""
    movie_rows = log[log["movie_id"].notna() & log["event"].isin(["view", "search", "pick"])]
    typed = log[(log["event"] == "search") & log["movie_id"].isna() & log["detail"].ne("")]
    if typed.empty:
        return movie_rows

    # The catalogue is sorted by popularity, so the first hits are the obvious ones.
    titles = movies["title"].astype(str).str.lower()
    spread = []
    for _, entry in typed.iterrows():
        hits = movies.loc[titles.str.contains(entry["detail"].lower(), regex=False), "id"]
        for movie_id in hits.head(config.TEXT_SEARCH_HITS):
            spread.append({**entry.to_dict(), "event": "search_text", "movie_id": movie_id})
    return pd.concat([movie_rows, pd.DataFrame(spread)], ignore_index=True) if spread else movie_rows


def build_profile(log: pd.DataFrame, my_list: pd.DataFrame, movies: pd.DataFrame,
                  now: datetime | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    """Score every movie the user has shown interest in.

    Returns (profile, genre_interest):
        profile         one row per movie (PROFILE_COLUMNS), the least recently
                        touched first, so patterns.py can tell what is LATELY
        genre_interest  {genre: weight} from the genres the user browsed
    """
    now = now or datetime.now()
    log = log.copy()
    log["movie_id"] = pd.to_numeric(log["movie_id"], errors="coerce")
    log["recency"] = _recency(log["timestamp"], now)
    log["when"] = pd.to_datetime(log["timestamp"], errors="coerce", format="mixed")

    # ------------------------------------------------------------ actions
    events = _movie_events(log, movies)
    if len(events):
        events = events.assign(points=events["event"].map(config.EVENT_WEIGHTS) * events["recency"])
        by_movie = events.groupby("movie_id").agg(
            event_score=("points", "sum"),
            views=("event", lambda e: int((e == "view").sum())),
            searches=("event", lambda e: int(e.isin(["search", "search_text"]).sum())),
            picks=("event", lambda e: int((e == "pick").sum())),
            last_event=("when", "max"),
        )
        by_movie["event_score"] = by_movie["event_score"].clip(upper=config.MAX_EVENT_SCORE)
    else:
        by_movie = pd.DataFrame(columns=["event_score", "views", "searches", "picks",
                                         "last_event"])

    # ------------------------------------------------------------ My List
    listed = my_list.set_index("movie_id") if len(my_list) else pd.DataFrame(
        columns=["status", "my_rating", "updated_on"])
    rating = pd.to_numeric(listed["my_rating"], errors="coerce")
    list_part = pd.DataFrame({
        "list_status": listed["status"],
        "my_rating": rating,
        "list_score": listed["status"].map(config.LIST_STATUS_WEIGHTS).fillna(0)
                      * (rating.fillna(7.0) / 7.0),
        "last_listed": pd.to_datetime(listed["updated_on"], errors="coerce", format="mixed"),
    })

    profile = by_movie.join(list_part, how="outer")
    if profile.empty:
        genre_interest = _genre_interest(log)
        return pd.DataFrame(columns=PROFILE_COLUMNS), genre_interest

    profile.index = profile.index.astype(int)
    profile = profile.rename_axis("movie_id").reset_index()
    for column in ("event_score", "list_score", "views", "searches", "picks"):
        profile[column] = pd.to_numeric(profile[column], errors="coerce").fillna(0)
    profile["disliked"] = profile["my_rating"] < config.LIKED_RATING
    profile["score"] = (profile["event_score"] + profile["list_score"]).where(~profile["disliked"], 0.0)
    profile["last_seen"] = profile[["last_event", "last_listed"]].max(axis=1)
    profile["list_status"] = profile["list_status"].fillna("")

    # Only movies still in the catalogue; their title comes from it.
    titles = movies.set_index("id")["title"]
    profile = profile[profile["movie_id"].isin(titles.index)].copy()
    profile["title"] = profile["movie_id"].map(titles)
    for column in ("views", "searches", "picks"):
        profile[column] = profile[column].astype(int)
    profile = profile.sort_values("last_seen", na_position="first").reset_index(drop=True)
    return profile[PROFILE_COLUMNS], _genre_interest(log)


def _genre_interest(log: pd.DataFrame) -> dict[str, float]:
    browsed = log[(log["event"] == "genre") & log["detail"].ne("")]
    weights = browsed["recency"] * config.EVENT_WEIGHTS["genre"]
    return {genre: float(w) for genre, w in weights.groupby(browsed["detail"]).sum().items()}


def profile_for(username: str, movies: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """build_profile() for a signed-in user, from their log and their My List."""
    return build_profile(read_activity(username), file_manager.read_my_list(username), movies)


def summary(log: pd.DataFrame) -> dict[str, int]:
    """How many of each action the log holds (for the page tiles and the admin)."""
    counts = log["event"].value_counts() if len(log) else pd.Series(dtype=int)
    return {event: int(counts.get(event, 0)) for event in EVENTS} | {"total": len(log)}
