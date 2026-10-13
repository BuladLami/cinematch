"""
inspect_dataset.py -- Step 1 of the project: look at the data before coding.

Run it with:    python inspect_dataset.py

It prints the size, columns, missing values and duplicates of the three
source files in "movies data/", and shows how many movies survive each of
the catalogue filters.  The decisions in cinematch/config.py (which movies
and columns to keep) and cinematch/preprocessing.py (how to fill the gaps)
all come from this report.

The files are opened read-only, so running this can never change the dataset.
"""

import pandas as pd

from cinematch import config

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


def line(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def describe(path) -> pd.DataFrame:
    line(f"READING {path.name}")
    df = pd.read_csv(path, low_memory=False, dtype=str)
    print(f"File size : {path.stat().st_size / (1024 * 1024):.1f} MB")
    print(f"Rows      : {len(df):,}")
    print(f"Columns   : {len(df.columns)}  ({', '.join(df.columns)})")
    ids = pd.to_numeric(df["id"], errors="coerce")
    print(f"Malformed ids   : {int(ids.isna().sum())}")
    print(f"Duplicate ids   : {int(ids.dropna().duplicated().sum())}")
    return df


def main() -> None:
    meta = describe(config.METADATA_CSV)

    line("MISSING VALUES IN movies_metadata.csv")
    missing = meta.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print((missing.to_frame("missing")
           .assign(percent=(missing / len(meta) * 100).round(1))).to_string())

    describe(config.KEYWORDS_CSV)
    describe(config.CREDITS_CSV)

    line("HOW MANY MOVIES SURVIVE EACH CATALOGUE FILTER")
    votes = pd.to_numeric(meta["vote_count"], errors="coerce").fillna(0)
    steps = [
        ("all rows", pd.Series(True, index=meta.index)),
        ("valid numeric id", pd.to_numeric(meta["id"], errors="coerce").notna()),
        ("status Released", meta["status"].fillna("Released") == "Released"),
        ("not adult", meta["adult"].fillna("False") == "False"),
        (f"at least {config.MIN_VOTES} votes", votes >= config.MIN_VOTES),
        ("has an overview", meta["overview"].fillna("").str.strip() != ""),
        ("has a poster path", meta["poster_path"].fillna("").str.startswith("/")),
    ]
    keep = pd.Series(True, index=meta.index)
    for label, condition in steps:
        keep &= condition
        print(f"  {label:<24} {int(keep.sum()):>7,}")
    print("  (duplicate ids are then dropped, leaving the final catalogue)")

    line("VOTE COUNTS")
    print(votes.describe().round(1).to_string())


if __name__ == "__main__":
    main()
