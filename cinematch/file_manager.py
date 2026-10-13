"""
file_manager.py -- ALL file input/output for CineMatch lives here.

This is the "file handling" module.  Every read and write goes through a
function in this file, which makes the file behaviour easy to find, easy to
explain and easy to test.

Two safety rules are enforced by the code itself:

  RULE 1  The source dataset folder ("movies data/") is READ ONLY.
          _guard_source() raises an error if any code ever tries to write
          inside it, so the downloaded data cannot be damaged even by mistake.

  RULE 2  Every write is ATOMIC.  The new content is written to a temporary
          file first, then swapped into place with os.replace().  If the program
          crashes halfway through, the old file is still intact instead of being
          left half-written.

Files this module manages:

    movies data/movies_metadata.csv   (read-only source)
    movies data/credits.csv           (read-only source)
    movies data/keywords.csv          (read-only source)
    data/catalogue.csv                (built once from the three sources)
    data/users/<name>/my_list.csv     (each user's list -- create/read/update/delete)
    data/users/<name>/activity.csv    (what each user searches / opens -- activity.py)
    data/activity_log.csv             (the log; accounts are in data/cinematch.db,
                                       a SQLite database -- see accounts.py)
"""

from __future__ import annotations

import csv
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import config, preprocessing


class FileManagerError(Exception):
    """Raised when a file operation is refused or cannot be completed."""


# ---------------------------------------------------------------------------
# SAFETY HELPERS
# ---------------------------------------------------------------------------

def _guard_source(path: Path) -> None:
    """Refuse to write inside the source dataset folder.  RULE 1 in code form."""
    if config.SOURCE_DIR.resolve() in Path(path).resolve().parents:
        raise FileManagerError(
            f'Refused: "{config.SOURCE_DIR.name}" holds the original dataset and '
            "is read-only."
        )


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _atomic_write(df: pd.DataFrame, path: Path) -> Path:
    """Write a DataFrame to CSV safely.  This is RULE 2 in code form.

    Steps: write "file.csv.tmp"  ->  os.replace() it onto "file.csv".
    os.replace is atomic on Windows and Linux, so readers never see a
    half-written file.
    """
    path = Path(path)
    _guard_source(path)
    config.ensure_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp_path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    os.replace(temp_path, path)
    return path


def write_csv_file(df: pd.DataFrame, path: Path) -> Path:
    """WRITE a table to CSV, atomically (the public name for _atomic_write)."""
    return _atomic_write(df, path)


def write_text_file(text: str, path: Path) -> Path:
    """WRITE plain text atomically."""
    path = Path(path)
    _guard_source(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(text, encoding="utf-8")
    os.replace(temp_path, path)
    return path


def append_csv_row(row: dict, path: Path, columns: list[str]) -> None:
    """APPEND one row to a CSV, writing the header first if the file is new.

    Appending (mode "a") adds to the end without rewriting what is already
    there, which suits a log that only ever grows.
    """
    path = Path(path)
    _guard_source(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def delete_user_folder(username: str) -> None:
    """DELETE data/users/<username>/ and everything in it.

    The resolved path must sit directly inside data/users/, so a crafted name
    such as "../.." can never point the delete anywhere else.
    """
    folder = config.user_dir(username).resolve()
    if folder.parent != config.USERS_DIR.resolve():
        raise FileManagerError(f'Refused to delete "{folder}".')
    if folder.exists():
        shutil.rmtree(folder)


# ---------------------------------------------------------------------------
# READ
# ---------------------------------------------------------------------------

def read_csv_file(path: Path | str, **options) -> pd.DataFrame:
    """Read any CSV file into a DataFrame, with clear errors.

    This is the single low-level read used by the whole program.
    """
    path = Path(path)
    if not path.exists():
        raise FileManagerError(f"File not found: {path}")
    try:
        df = pd.read_csv(path, encoding="utf-8", low_memory=False, **options)
    except UnicodeDecodeError:
        # Some spreadsheet programs save CSVs in the Windows Latin-1 encoding.
        df = pd.read_csv(path, encoding="latin-1", low_memory=False, **options)
    except pd.errors.EmptyDataError as exc:
        raise FileManagerError(f"{path.name} is empty.") from exc
    except pd.errors.ParserError as exc:
        raise FileManagerError(f"{path.name} is not a valid CSV file: {exc}") from exc
    return df


# ---------------------------------------------------------------------------
# BUILDING THE CATALOGUE  (source folder -> data/catalogue.csv)
# ---------------------------------------------------------------------------
# The source files store lists as Python-style text, for example
#     [{'id': 16, 'name': 'Animation'}, {'id': 35, 'name': 'Comedy'}]
# A regular expression pulls the names out far faster than evaluating 45,000
# of these strings (credits.csv alone is ~190 MB).  A name holding an
# apostrophe is written in double quotes ("Peter O'Toole"), so both are handled.
_NAME = r"""'name': (?:'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)")"""
_NAME_PATTERN = re.compile(_NAME)
# In crew entries "job" comes just before "name", which isolates directors.
_DIRECTOR_PATTERN = re.compile(r"'job': 'Director', " + _NAME)


def _names(text, pattern=_NAME_PATTERN, limit: int | None = None) -> list[str]:
    """Every name matched in one source cell, in order."""
    if not isinstance(text, str):
        return []
    names = []
    for match in pattern.finditer(text):
        name = (match.group(1) if match.group(1) is not None else match.group(2))
        name = name.replace("\\'", "'").replace('\\"', '"').strip()
        if name and name not in names:
            names.append(name)
            if limit and len(names) >= limit:
                break
    return names


def _joined(names: list[str]) -> str:
    return config.LIST_SEP.join(n.replace(config.LIST_SEP, "/") for n in names)


def _numeric_ids(series: pd.Series) -> pd.Series:
    """Source ids as integers; the few malformed ones (dates in the id column) become NaN."""
    return pd.to_numeric(series, errors="coerce")


def build_catalogue(progress=None) -> Path:
    """READ the three source files, merge them, and WRITE data/catalogue.csv.

    progress is an optional callback(message) so the page can say what is
    happening during the ~30 seconds this takes on the first run.
    """
    say = progress or (lambda message: None)
    for source in (config.METADATA_CSV, config.CREDITS_CSV, config.KEYWORDS_CSV):
        if not source.exists():
            raise FileManagerError(f"Source file not found: {source}")

    say("Reading movies_metadata.csv ...")
    meta = read_csv_file(config.METADATA_CSV, dtype=str)
    meta["id"] = _numeric_ids(meta["id"])
    for column in config.NUMERIC_COLUMNS:
        meta[column] = pd.to_numeric(meta[column], errors="coerce").fillna(0)

    # Keep released, non-adult films that enough people have rated, and that
    # have something to show: a plot summary and a poster.
    keep = (
        meta["id"].notna()
        & (meta["status"].fillna("Released") == "Released")
        & (meta["adult"].fillna("False") == "False")
        & (meta["vote_count"] >= config.MIN_VOTES)
        & meta["overview"].fillna("").str.strip().ne("")
        & meta["poster_path"].fillna("").str.startswith("/")
        & meta["title"].fillna("").str.strip().ne("")
    )
    meta = meta[keep].drop_duplicates(subset="id").copy()
    meta["id"] = meta["id"].astype(int)
    wanted = set(meta["id"])

    meta["genres"] = meta["genres"].apply(lambda t: _joined(_names(t)))
    meta["collection"] = meta["belongs_to_collection"].apply(
        lambda t: _joined(_names(t, limit=1)))

    say("Reading keywords.csv ...")
    keywords = read_csv_file(config.KEYWORDS_CSV, dtype=str)
    keywords["id"] = _numeric_ids(keywords["id"])
    keywords = keywords[keywords["id"].isin(wanted)].drop_duplicates(subset="id")
    keywords["keywords"] = keywords["keywords"].apply(lambda t: _joined(_names(t)))

    say("Reading credits.csv (the big one) ...")
    credits = read_csv_file(config.CREDITS_CSV, dtype=str)
    credits["id"] = _numeric_ids(credits["id"])
    credits = credits[credits["id"].isin(wanted)].drop_duplicates(subset="id")
    # The cast list is already in billing order, so the first names are the stars.
    credits["cast"] = credits["cast"].apply(
        lambda t: _joined(_names(t, limit=config.CAST_SIZE)))
    credits["director"] = credits["crew"].apply(
        lambda t: _joined(_names(t, pattern=_DIRECTOR_PATTERN)))

    say("Merging and saving the catalogue ...")
    for table in (keywords, credits):
        table["id"] = table["id"].astype(int)
    catalogue = (meta
                 .merge(keywords[["id", "keywords"]], on="id", how="left")
                 .merge(credits[["id", "cast", "director"]], on="id", how="left"))
    catalogue = catalogue[[c for c in config.CATALOGUE_COLUMNS if c in catalogue.columns]]
    catalogue = catalogue.sort_values("popularity", ascending=False)
    return _atomic_write(catalogue, config.CATALOGUE_CSV)


def catalogue_is_current() -> bool:
    """True when the catalogue exists and is newer than every source file."""
    if not config.CATALOGUE_CSV.exists():
        return False
    built = config.CATALOGUE_CSV.stat().st_mtime
    sources = (config.METADATA_CSV, config.CREDITS_CSV, config.KEYWORDS_CSV)
    return all(not s.exists() or s.stat().st_mtime <= built for s in sources)


def read_catalogue(progress=None) -> pd.DataFrame:
    """READ data/catalogue.csv, building it from the source files first if needed."""
    if not catalogue_is_current():
        build_catalogue(progress)
    df = read_csv_file(config.CATALOGUE_CSV)
    if df.empty:
        raise FileManagerError("The catalogue is empty -- check the source files.")
    return df


def load_prepared_movies(progress=None) -> tuple[pd.DataFrame, dict]:
    """READ + PROCESS: load the catalogue and run the cleaning pipeline."""
    return preprocessing.prepare(read_catalogue(progress))


# ---------------------------------------------------------------------------
# MY LIST  (the user's CRUD file)
# ---------------------------------------------------------------------------
# One row per movie the user has saved:
#     movie_id, title, status, my_rating, notes, added_on, updated_on
# Every function below reads the file, makes one change, and writes it back.
# Each takes the signed-in username, so every account has its own list in
# data/users/<username>/my_list.csv.

def read_my_list(username: str | None = None) -> pd.DataFrame:
    """READ the user's list (an empty table if nothing has been saved yet)."""
    path = config.my_list_path(username)
    if not path.exists():
        return pd.DataFrame(columns=config.MY_LIST_COLUMNS)
    df = read_csv_file(path)
    for column in config.MY_LIST_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    df["movie_id"] = pd.to_numeric(df["movie_id"], errors="coerce")
    df = df.dropna(subset=["movie_id"])
    df["movie_id"] = df["movie_id"].astype(int)
    df["my_rating"] = pd.to_numeric(df["my_rating"], errors="coerce")
    df["notes"] = df["notes"].fillna("").astype(str)
    return df[config.MY_LIST_COLUMNS].reset_index(drop=True)


def _save_my_list(df: pd.DataFrame, username: str | None = None) -> Path:
    return _atomic_write(df[config.MY_LIST_COLUMNS], config.my_list_path(username))


def get_list_entry(movie_id: int, username: str | None = None) -> dict | None:
    """READ one entry of the list, or None if the movie is not on it."""
    df = read_my_list(username)
    rows = df[df["movie_id"] == int(movie_id)]
    return rows.iloc[0].to_dict() if len(rows) else None


def _check_status(status: str) -> str:
    if status not in config.LIST_STATUSES:
        raise FileManagerError(
            f'Unknown status "{status}". Use one of: {", ".join(config.LIST_STATUSES)}.')
    return status


def _check_rating(rating):
    """A rating is blank (not rated) or a number from 1 to 10."""
    if rating is None or (isinstance(rating, float) and pd.isna(rating)) or rating == "":
        return None
    value = pd.to_numeric(rating, errors="coerce")
    if pd.isna(value) or not 1 <= float(value) <= 10:
        raise FileManagerError("A rating must be a number from 1 to 10.")
    return float(value)


def add_to_list(movie_id: int, title: str, status: str = "Want to watch",
                my_rating=None, notes: str = "", username: str | None = None) -> str:
    """CREATE: put a movie on the user's list and save the file."""
    df = read_my_list(username)
    if (df["movie_id"] == int(movie_id)).any():
        raise FileManagerError(f'"{title}" is already on your list.')

    stamp = _now()
    entry = {
        "movie_id": int(movie_id),
        "title": str(title),
        "status": _check_status(status),
        "my_rating": _check_rating(my_rating),
        "notes": str(notes or "").strip(),
        "added_on": stamp,
        "updated_on": stamp,
    }
    frames = [df, pd.DataFrame([entry])] if len(df) else [pd.DataFrame([entry])]
    _save_my_list(pd.concat(frames, ignore_index=True), username)
    return f'Added "{title}" to your list as "{status}".'


def update_list_entry(movie_id: int, username: str | None = None, **changes) -> str:
    """UPDATE: change the status, rating or notes of a movie on the list."""
    df = read_my_list(username)
    mask = df["movie_id"] == int(movie_id)
    if not mask.any():
        raise FileManagerError("That movie is not on your list.")

    row = df.index[mask][0]
    if "status" in changes:
        df.loc[row, "status"] = _check_status(changes["status"])
    if "my_rating" in changes:
        df.loc[row, "my_rating"] = _check_rating(changes["my_rating"])
    if "notes" in changes:
        df.loc[row, "notes"] = str(changes["notes"] or "").strip()
    df.loc[row, "updated_on"] = _now()

    _save_my_list(df, username)
    return f'Updated "{df.loc[row, "title"]}".'


def remove_from_list(movie_id: int, username: str | None = None) -> str:
    """DELETE: take a movie off the list and save the file."""
    df = read_my_list(username)
    mask = df["movie_id"] == int(movie_id)
    if not mask.any():
        raise FileManagerError("That movie is not on your list.")
    title = df.loc[mask, "title"].iloc[0]
    _save_my_list(df[~mask], username)
    return f'Removed "{title}" from your list.'


def replace_my_list(df: pd.DataFrame, username: str | None = None) -> str:
    """UPDATE many entries at once (the editable table on the My List page).

    Every row is validated before anything is written, so one bad value
    cannot leave the file half-changed.
    """
    current = read_my_list(username).set_index("movie_id")
    edited = df.copy()
    edited["movie_id"] = edited["movie_id"].astype(int)
    edited["status"] = edited["status"].apply(_check_status)
    edited["my_rating"] = edited["my_rating"].apply(_check_rating)
    edited["notes"] = edited["notes"].fillna("").astype(str).str.strip()

    changed = 0
    for _, row in edited.iterrows():
        if row["movie_id"] not in current.index:
            continue
        old = current.loc[row["movie_id"]]
        # pandas turns a blank rating into NaN, and NaN never equals itself,
        # so both sides are normalised to None before comparing.
        new_rating = None if pd.isna(row["my_rating"]) else float(row["my_rating"])
        old_rating = None if pd.isna(old["my_rating"]) else float(old["my_rating"])
        if (row["status"], new_rating, row["notes"]) != (old["status"], old_rating,
                                                         old["notes"]):
            current.loc[row["movie_id"], ["status", "my_rating", "notes"]] = [
                row["status"], new_rating, row["notes"]]
            current.loc[row["movie_id"], "updated_on"] = _now()
            changed += 1

    _save_my_list(current.reset_index(), username)
    return f"Saved {changed} change(s) to your list."


# ---------------------------------------------------------------------------
# EXPORT
# ---------------------------------------------------------------------------

def list_as_csv(df: pd.DataFrame) -> bytes:
    """The user's list as CSV bytes, for the browser's download button."""
    return df.to_csv(index=False).encode("utf-8")
