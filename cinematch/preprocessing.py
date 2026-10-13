"""
preprocessing.py -- Data cleaning and feature preparation.

This module is the "data processing" part of CineMatch.  It takes the raw
catalogue DataFrame that came out of the CSV file and turns it into something
the machine-learning model can use.

The steps, in order:

  1. clean_text()          lowercase + remove punctuation
  2. fill_missing()        replace every NaN with a safe value
  3. drop_bad_rows()       remove rows with no title and duplicates
  4. add_helper_columns()  add "year" and readable, comma-separated lists
  5. build_soup()          join the chosen fields into ONE text column

Nothing here touches a file -- that is file_manager.py's job.  Keeping the two
apart means the cleaning logic can be tested on its own.
"""

from __future__ import annotations

import re

import pandas as pd

from . import config

# ---------------------------------------------------------------------------
# STEP 1: text helpers
# ---------------------------------------------------------------------------

# Apostrophes are deleted outright so "Marvel's" becomes one word, "marvels",
# instead of the two words "marvel" and "s".
_APOSTROPHE_PATTERN = re.compile(r"['’]")

# Every other non-letter/number becomes a space, which splits joined words.
# \w keeps accented letters, so "Léa Seydoux" stays readable.
_PUNCTUATION_PATTERN = re.compile(r"[^\w ]+|_")

# Matches runs of whitespace, so they can be squashed into one space.
_SPACES_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase the text and strip punctuation, leaving plain words.

    "Marvel's Avengers: Age of Ultron" -> "marvels avengers age of ultron"

    TF-IDF compares words, so "Action," and "Action" must become the same word.
    """
    if not isinstance(text, str):
        return ""
    text = _APOSTROPHE_PATTERN.sub("", text.lower())
    text = _PUNCTUATION_PATTERN.sub(" ", text)
    return _SPACES_PATTERN.sub(" ", text).strip()


def squash_name(name: str) -> str:
    """Glue a multi-word name into one token: "James Cameron" -> jamescameron.

    Why: TF-IDF splits on spaces, so "James Cameron" would become the two words
    "james" and "cameron".  Then any movie directed by a different "James" would
    look similar.  Squashing keeps one person (or one keyword phrase, such as
    "time travel") as one identity.
    """
    return clean_text(name).replace(" ", "")


def split_list(value) -> list[str]:
    """Turn a stored list back into items: "Action|Comedy" -> ["Action", "Comedy"]."""
    if not isinstance(value, str) or not value.strip():
        return []
    return [item.strip() for item in value.split(config.LIST_SEP) if item.strip()]


def list_tokens(value) -> str:
    """A stored list as model tokens: "Tom Hanks|Tim Allen" -> "tomhanks timallen"."""
    return " ".join(squash_name(item) for item in split_list(value))


# ---------------------------------------------------------------------------
# STEP 2: missing values
# ---------------------------------------------------------------------------

def fill_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Replace every missing value and report how many were filled.

    Rules used (and why):
      * text columns    -> "" (empty string).  A movie with no tagline simply
        contributes no tagline words; dropping the whole row would throw away
        otherwise good data.
      * numeric columns -> 0.  A rating or runtime of 0 reads as "unknown" and
        is excluded from averages instead of skewing them.
    """
    df = df.copy()
    report: dict[str, int] = {}

    for column in df.columns:
        missing = int(df[column].isna().sum())
        if missing:
            report[column] = missing

    text_columns = [c for c in df.columns if c not in config.NUMERIC_COLUMNS and c != "id"]
    for column in text_columns:
        df[column] = df[column].astype("string").fillna("").astype(str)

    for column in config.NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    return df, report


def drop_bad_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove rows that cannot be used, and report what was removed.

    Removed:
      * rows with a blank title (there would be nothing to show the user)
      * rows duplicated on every single column (true copies)
      * rows sharing an id (the id is our primary key, so it must be unique)

    NOT removed: movies that merely share a title ("Hamlet", "Les Misérables").
    They are genuinely different films, so deleting one would lose real data.
    """
    report = {"blank_title": 0, "exact_duplicates": 0, "duplicate_ids": 0}

    before = len(df)
    df = df[df["title"].astype(str).str.strip() != ""]
    report["blank_title"] = before - len(df)

    before = len(df)
    df = df.drop_duplicates()
    report["exact_duplicates"] = before - len(df)

    if "id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["id"], keep="first")
        report["duplicate_ids"] = before - len(df)

    return df.reset_index(drop=True), report


# ---------------------------------------------------------------------------
# STEP 3: helper columns and the content soup
# ---------------------------------------------------------------------------

def add_helper_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add display columns derived from existing ones.

      year          -> the 4-digit release year pulled out of release_date
      genre_list    -> "Action, Adventure, Science Fiction"
      cast_list     -> "Tom Hanks, Tim Allen, Don Rickles"
      keyword_list  -> "jealousy, toy, boy next door"
      director_list -> "Joel Coen, Ethan Coen"
    """
    df = df.copy()

    if "release_date" in df.columns:
        years = pd.to_datetime(df["release_date"], errors="coerce", format="mixed")
        df["year"] = years.dt.year.fillna(0).astype(int)
    else:
        df["year"] = 0

    for column, target in [("genres", "genre_list"), ("cast", "cast_list"),
                           ("keywords", "keyword_list"), ("director", "director_list")]:
        source = df[column] if column in df.columns else pd.Series("", index=df.index)
        df[target] = source.apply(lambda value: ", ".join(split_list(value)))

    return df


def build_soup_for_row(row: pd.Series) -> str:
    """Join one movie's chosen fields into a single bag of words.

    This is the heart of a content-based recommender.  Each field is cleaned,
    then repeated according to config.FEATURE_WEIGHTS so the fields that best
    describe the *kind* of movie (genres, keywords, collection) count more than
    the long overview text.

    Shortened example for Toy Story:
        "animation comedy family                  (x3)
         jealousy toy boy friendship boynextdoor  (x3)
         toystorycollection                       (x3)
         tomhanks timallen donrickles             (x2)
         johnlasseter                             (x2)
         led by woody andys toys live happily     (x1)"
    """
    parts: list[str] = []

    for column, weight in config.FEATURE_WEIGHTS.items():
        value = row.get(column, "")
        if column in ("overview", "tagline"):
            text = clean_text(value)
        else:
            # Lists (and the collection name): one token per person or phrase.
            text = list_tokens(value)
        if text:
            parts.extend([text] * weight)

    return " ".join(parts)


def build_soup(df: pd.DataFrame) -> pd.DataFrame:
    """Add the "soup" column that the TF-IDF vectorizer reads."""
    df = df.copy()
    df["soup"] = df.apply(build_soup_for_row, axis=1) if len(df) else ""
    return df


def prepare(df: pd.DataFrame, verbose: bool = False) -> tuple[pd.DataFrame, dict]:
    """Run the whole cleaning pipeline and return (clean_df, report)."""
    rows_in = len(df)

    df, missing_report = fill_missing(df)
    df, dropped_report = drop_bad_rows(df)
    df = add_helper_columns(df)
    df = build_soup(df)

    report = {
        "rows_in": rows_in,
        "rows_out": len(df),
        "missing_values_filled": missing_report,
        "rows_removed": dropped_report,
        "empty_soup": int((df["soup"].str.strip() == "").sum()),
    }

    if verbose:
        print(f"Prepared {report['rows_out']} movies "
              f"({sum(missing_report.values())} missing values filled)")

    return df, report


# ---------------------------------------------------------------------------
# STEP 4: numbers for the pages
# ---------------------------------------------------------------------------

def all_genres(df: pd.DataFrame) -> list[str]:
    """Every distinct genre in the catalogue, most common first."""
    return list(genre_counts(df).index)


def genre_counts(df: pd.DataFrame) -> pd.Series:
    genres = [g for raw in df.get("genres", pd.Series(dtype=str)) for g in split_list(raw)]
    return pd.Series(genres).value_counts() if genres else pd.Series(dtype=int)


def dataset_summary(df: pd.DataFrame) -> dict:
    """A few catalogue statistics."""
    rated = df.loc[df["vote_average"] > 0, "vote_average"] if "vote_average" in df else pd.Series(dtype=float)
    years = df.loc[df["year"] > 0, "year"] if "year" in df else pd.Series(dtype=int)
    counts = genre_counts(df)
    return {
        "total_movies": int(len(df)),
        "average_rating": round(float(rated.mean()), 2) if len(rated) else 0.0,
        "rated_movies": int(len(rated)),
        "genre_count": int(len(counts)),
        "genre_counts": counts,
        "year_min": int(years.min()) if len(years) else 0,
        "year_max": int(years.max()) if len(years) else 0,
    }
