"""
patterns.py -- Finding the PATTERN in what a user watches, then recommending from it.

The input is the PROFILE that activity.py learns from what the user does on
the site: every movie they searched, opened, picked or put on My List, each
with an interest score (recent and deliberate actions score higher).

PART 1: analyze()  looks at every movie the user has shown interest in and
counts what keeps coming back, each movie counting as much as its score:

    genres          which genres, and which genre COMBINATIONS (Action + Sci-Fi)
    people          directors and actors seen in two or more of the movies
    keywords        recurring themes ("dystopia", "time travel")
    series          collections started but not finished
    era             the decade most of the movies come from
    style           typical length, how highly rated, how mainstream, language
    lately          what the movies touched most recently lean towards,
                    compared with everything as a whole
    own ratings     which genres the user rates highest on My List

Each finding becomes a plain sentence ("insights") shown to the user.

PART 2: recommend()  scores every movie in the catalogue by mixing

    35%  content similarity  TF-IDF taste profile of the movies, each weighted
                             by its interest score (what the films are ABOUT)
    20%  people like you     collaborative filtering: how much viewers who
                             liked the same movies liked this one
                             (collaborative.py; without it, content gets 55%)
    15%  genre fit           how much of the user's interest shares its genres
    10%  era fit             how often the user goes for films from its decade
     5%  rating fit          how close its rating is to what the user watches
    15%  people / series     a recurring director or actor, or the next film
                             in a series the user has started

and returns the best ones with the reasons that put them there, so every
recommendation can be traced back to the pattern.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import pandas as pd

from . import config, preprocessing

WEIGHTS = {"similarity": 0.35, "behaviour": 0.20, "genre": 0.15, "era": 0.10,
           "rating": 0.05, "bonus": 0.15}

# A movie whose "people like you" score is at least this (0-1) gets that as a reason.
BEHAVIOUR_REASON = 0.5

# Fewer movies than this still get recommendations, but the "lately"
# comparison needs enough of them to split into an earlier and a recent part.
MIN_FOR_TREND = 6

PROFILE_FIELDS = ["movie_id", "score", "my_rating", "disliked", "last_seen"]


def _decade(year) -> str:
    year = int(year or 0)
    return f"{year // 10 * 10}s" if year else ""


def engaged_movies(profile: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """The catalogue rows of the profile's movies, least recent first, with their score."""
    joined = profile[PROFILE_FIELDS].merge(movies.reset_index(), left_on="movie_id",
                                           right_on="id", how="inner")
    return joined.set_index("index")


def _liked(profile: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    engaged = engaged_movies(profile, movies)
    return engaged[engaged["score"] > 0]


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    keep = values > 0
    if not keep.any():
        return 0.0
    return float((values[keep] * weights[keep]).sum() / weights[keep].sum())


def _repeats(counter: Counter, limit: int) -> list[tuple[str, int]]:
    return [(name, n) for name, n in counter.most_common(limit) if n >= config.MIN_REPEAT]


# ---------------------------------------------------------------------------
# PART 1: THE PATTERN
# ---------------------------------------------------------------------------

def analyze(profile: pd.DataFrame, movies: pd.DataFrame,
            genre_interest: dict[str, float] | None = None) -> dict:
    """Describe the user's watching pattern.  Returns {} until there is something to go on.

    Genres, decades, length, rating, popularity and language are weighted by
    each movie's interest score.  Directors, actors, themes and series are
    counted in whole movies, because "a director you keep coming back to"
    means one found in two or more different movies.  genre_interest (the
    genres the user browsed) adds to the genre weights.
    """
    watched = _liked(profile, movies)
    n = len(watched)
    if n == 0:
        return {}
    weight = watched["score"]
    total = float(weight.sum())

    genre_lists = watched["genres"].apply(preprocessing.split_list)
    genre_movies = Counter(g for genres in genre_lists for g in genres)
    movie_genre_weight: Counter = Counter()
    for genres, w in zip(genre_lists, weight):
        for g in genres:
            movie_genre_weight[g] += w
    genre_weight = Counter(movie_genre_weight)
    browsed = genre_interest or {}
    for g, w in browsed.items():
        genre_weight[g] += w
    genre_share = {g: w / (total + sum(browsed.values())) for g, w in genre_weight.items()}
    pair_counts = Counter(pair for genres in genre_lists
                          for pair in combinations(sorted(genres), 2))

    directors = Counter(d for v in watched["director"] for d in preprocessing.split_list(v))
    actors = Counter(a for v in watched["cast"] for a in preprocessing.split_list(v))
    keywords = Counter(k for v in watched["keywords"] for k in preprocessing.split_list(v))

    # Series: how many films of each collection the catalogue holds vs. seen.
    # Anything the user has touched (even disliked) is not "missing".
    series = []
    seen_ids = set(profile["movie_id"])
    for name, group in watched[watched["collection"].str.strip() != ""].groupby("collection"):
        in_catalogue = movies[movies["collection"] == name].sort_values("year")
        missing = in_catalogue[~in_catalogue["id"].isin(seen_ids)]
        series.append({"name": name, "watched": len(group), "total": len(in_catalogue),
                       "missing_ids": missing["id"].tolist()})
    series.sort(key=lambda s: (-s["watched"], s["name"]))

    decades: Counter = Counter()
    for d, w in zip(watched["year"].apply(_decade), weight):
        if d:
            decades[d] += w
    decade_share = {d: w / total for d, w in decades.items()}

    # Popularity as a percentile of the catalogue: 0.9 = more popular than 90%.
    popularity_pct = movies["popularity"].rank(pct=True).loc[watched.index]
    languages: Counter = Counter()
    for language, w in zip(watched["original_language"].astype(str).str.lower(), weight):
        languages[language] += w
    foreign_share = 1 - languages.get("en", 0) / total

    pattern = {
        "count": n,
        "disliked": int(profile["disliked"].sum()),
        "genre_counts": pd.Series(genre_weight, dtype=float).round(1)
                        .sort_values(ascending=False),
        "genre_movies": genre_movies,
        "genre_share": genre_share,
        "top_genres": [g for g, _ in genre_weight.most_common(3)],
        "genre_pairs": _repeats(pair_counts, 3),
        "directors": _repeats(directors, 5),
        "actors": _repeats(actors, 5),
        "keywords": _repeats(keywords, 8),
        "series": series,
        "decade_counts": pd.Series(decades, dtype=float).round(1).sort_index(),
        "decade_share": decade_share,
        "top_decade": decades.most_common(1)[0][0] if decades else "",
        "avg_runtime": _weighted_mean(watched["runtime"], weight),
        "avg_vote": _weighted_mean(watched["vote_average"], weight),
        "popularity_pct": float((popularity_pct * weight).sum() / total),
        "languages": languages,
        "foreign_share": foreign_share,
        "recent": _recent_trend(watched, {g: w / total for g, w in movie_genre_weight.items()}),
        "rated_genres": _rated_genres(watched, genre_lists),
    }
    pattern["insights"] = _insights(pattern)
    return pattern


def _recent_trend(watched: pd.DataFrame, overall_share: dict) -> dict:
    """Compare the most recently touched third of the movies with all of them."""
    if len(watched) < MIN_FOR_TREND:
        return {}
    recent = watched.tail(max(3, len(watched) // 3))
    counts = Counter(g for v in recent["genres"] for g in preprocessing.split_list(v))
    share = {g: c / len(recent) for g, c in counts.items()}
    # A genre is "rising" when it is in at least half of the recent films and
    # clearly more common lately than across everything.
    rising = sorted((g for g, s in share.items()
                     if s >= 0.5 and s - overall_share.get(g, 0) >= 0.15),
                    key=lambda g: -(share[g] - overall_share.get(g, 0)))
    return {"size": len(recent), "top": [g for g, _ in counts.most_common(3)],
            "rising": rising[:2], "ids": recent["id"].tolist()}


def _rated_genres(watched: pd.DataFrame, genre_lists: pd.Series) -> list[tuple[str, float]]:
    """The user's average own rating per genre (genres rated 2+ times)."""
    scores: dict[str, list[float]] = {}
    for genres, rating in zip(genre_lists, watched["my_rating"]):
        if pd.notna(rating):
            for g in genres:
                scores.setdefault(g, []).append(float(rating))
    averages = [(g, sum(v) / len(v)) for g, v in scores.items() if len(v) >= config.MIN_REPEAT]
    return sorted(averages, key=lambda item: -item[1])


def _insights(p: dict) -> list[str]:
    """Turn the numbers into sentences a person would say."""
    n = p["count"]
    lines = []
    if p["top_genres"]:
        g = p["top_genres"][0]
        lines.append(f"**{g}** is your main genre: it is in {p['genre_movies'].get(g, 0)} of "
                     f"the {n} movies you showed interest in "
                     f"({p['genre_share'][g]:.0%} of your interest).")
    if p["genre_pairs"]:
        (a, b), count = p["genre_pairs"][0]
        lines.append(f"Your favourite mix is **{a} + {b}** ({count} movies).")
    if p["directors"]:
        names = ", ".join(f"{d} ({c})" for d, c in p["directors"][:3])
        lines.append(f"You keep coming back to the same directors: **{names}**.")
    if p["actors"]:
        names = ", ".join(f"{a} ({c})" for a, c in p["actors"][:3])
        lines.append(f"Actors who show up again and again: **{names}**.")
    if p["keywords"]:
        themes = ", ".join(k for k, _ in p["keywords"][:5])
        lines.append(f"Recurring themes: **{themes}**.")
    unfinished = [s for s in p["series"] if s["missing_ids"]]
    if unfinished:
        s = unfinished[0]
        lines.append(f"You have been looking at **{s['name']}** "
                     f"({s['watched']} of {s['total']} films).")
    if p["top_decade"]:
        lines.append(f"Most of what you go for is from the **{p['top_decade']}** "
                     f"({p['decade_share'][p['top_decade']]:.0%}).")
    if p["avg_runtime"]:
        style = ("short and snappy" if p["avg_runtime"] < 100 else
                 "long and epic" if p["avg_runtime"] > 130 else "standard length")
        lines.append(f"Your movies are usually **{style}** "
                     f"(about {p['avg_runtime']:.0f} minutes).")
    if p["avg_vote"]:
        taste = ("critically loved films" if p["avg_vote"] >= 7.3 else
                 "easy-watching crowd-pleasers" if p["avg_vote"] < 6.2 else
                 "well-reviewed films")
        lines.append(f"You lean towards **{taste}** (average rating {p['avg_vote']:.1f}/10).")
    pct = p["popularity_pct"]
    reach = ("big, popular titles" if pct >= 0.85 else
             "lesser-known films" if pct < 0.5 else "a mix of hits and lesser-known films")
    lines.append(f"You mostly pick **{reach}**.")
    if p["foreign_share"] >= 0.25:
        lines.append(f"**{p['foreign_share']:.0%}** of your interest is in films not in English.")
    if p["recent"].get("rising"):
        lines.append(f"Lately you have been looking at more "
                     f"**{' and '.join(p['recent']['rising'])}**.")
    if p["rated_genres"]:
        g, score = p["rated_genres"][0]
        lines.append(f"You rate **{g}** highest ({score:.1f}/10 on average).")
    if p["disliked"]:
        lines.append(f"{p['disliked']} movie(s) you rated low are kept out of your pattern.")
    if n < config.MIN_MOVIES_FOR_PATTERN:
        lines.append("Keep browsing: the pattern gets sharper with every movie you "
                     "search, open or save.")
    return lines


# ---------------------------------------------------------------------------
# PART 2: RECOMMENDATIONS FROM THE PATTERN
# ---------------------------------------------------------------------------

def recommend(engine, movies: pd.DataFrame, profile: pd.DataFrame, pattern: dict,
              top_n: int = 12, exclude_ids=(), behaviour=None) -> pd.DataFrame:
    """The movies that best fit the pattern, best first, each with its reasons.

    The taste profile is the average of the movies the user showed interest
    in, each weighted by its score; disliked movies are left out of it.
    behaviour is a collaborative.BehaviourModel; without one (or when it knows
    none of the user's movies) the content similarity takes its share.
    Nothing the user has already searched, opened, picked or listed comes back.
    Returns catalogue rows (indexed by catalogue row) plus the columns
    pattern_score, similarity_pct (the blended match, 0-100) and reasons.
    """
    if not pattern:
        return pd.DataFrame()
    watched = _liked(profile, movies)
    if watched.empty:
        return pd.DataFrame()
    rows, weights = list(watched.index), list(watched["score"])
    scores, _ = engine.profile_scores(rows, weights)
    if scores is None:
        return pd.DataFrame()

    m = movies
    similarity = pd.Series(scores, index=m.index)
    similarity = similarity / similarity.max() if similarity.max() > 0 else similarity

    learned = behaviour.scores(rows, weights) if behaviour is not None else None
    if learned is None:
        people_like_you = pd.Series(0.0, index=m.index)
        w_similarity, w_behaviour = WEIGHTS["similarity"] + WEIGHTS["behaviour"], 0.0
    else:
        people_like_you = pd.Series(learned, index=m.index).clip(lower=0)
        if people_like_you.max() > 0:
            people_like_you = people_like_you / people_like_you.max()
        w_similarity, w_behaviour = WEIGHTS["similarity"], WEIGHTS["behaviour"]

    # Genre fit: the shares of the candidate's genres, against the best possible.
    best = sum(sorted(pattern["genre_share"].values(), reverse=True)[:3]) or 1
    genre_fit = m["genres"].apply(
        lambda v: min(1.0, sum(pattern["genre_share"].get(g, 0)
                               for g in preprocessing.split_list(v)) / best))

    top_share = max(pattern["decade_share"].values(), default=0) or 1
    era_fit = m["year"].apply(lambda y: pattern["decade_share"].get(_decade(y), 0) / top_share)

    avg_vote = pattern["avg_vote"] or 6.5
    rating_fit = (1 - (m["vote_average"] - avg_vote).abs() / 3).clip(0, 1)

    # Bonus: the next film in a started series, or a recurring director / actor.
    series_next = {mid: s for s in pattern["series"] for mid in s["missing_ids"]}
    directors = dict(pattern["directors"])
    actors = dict(pattern["actors"])
    split = preprocessing.split_list

    def bonus(movie_id, director, cast) -> float:
        if movie_id in series_next or any(d in directors for d in split(director)):
            return 1.0
        return 0.6 if any(a in actors for a in split(cast)) else 0.0

    bonuses = pd.Series([bonus(i, d, c) for i, d, c in zip(m["id"], m["director"], m["cast"])],
                        index=m.index)

    total = (w_similarity * similarity + w_behaviour * people_like_you
             + WEIGHTS["genre"] * genre_fit + WEIGHTS["era"] * era_fit
             + WEIGHTS["rating"] * rating_fit + WEIGHTS["bonus"] * bonuses)

    skip = set(profile["movie_id"]) | {int(i) for i in exclude_ids}
    total = total[~m["id"].isin(skip)].sort_values(ascending=False).head(top_n)

    result = m.loc[total.index].copy()
    result["pattern_score"] = total.values
    result["similarity_pct"] = (total.values * 100).round(1)
    result["people_like_you"] = people_like_you.loc[total.index].values
    result["reasons"] = [_reasons(movie, pattern, series_next, directors, actors)
                         for _, movie in result.iterrows()]
    return result


def _reasons(movie: pd.Series, pattern: dict, series_next: dict,
             directors: dict, actors: dict) -> list[str]:
    """Why this movie fits the pattern, most specific reason first."""
    reasons = []
    split = preprocessing.split_list
    if movie["id"] in series_next:
        s = series_next[movie["id"]]
        reasons.append(f"Continues {s['name']} (you looked at {s['watched']} of {s['total']})")
    for d in split(movie["director"]):
        if d in directors:
            reasons.append(f"Directed by {d} ({directors[d]} of your movies)")
            break
    for a in split(movie["cast"]):
        if a in actors:
            reasons.append(f"Stars {a} ({actors[a]} of your movies)")
            break
    if movie.get("people_like_you", 0) >= BEHAVIOUR_REASON:
        reasons.append("Viewers who like your movies like this one")
    shared = [g for g in split(movie["genres"]) if g in pattern["top_genres"]]
    if shared:
        reasons.append(f"{' + '.join(shared[:2])}: your top genre{'s' if len(shared) > 1 else ''}")
    if pattern["recent"].get("rising") and any(
            g in pattern["recent"]["rising"] for g in split(movie["genres"])):
        reasons.append("Fits what you have been looking at lately")
    if _decade(movie["year"]) == pattern["top_decade"]:
        reasons.append(f"From the {pattern['top_decade']}, your favourite decade")
    themes = [k for k in split(movie["keywords"]) if k in dict(pattern["keywords"])]
    if themes:
        reasons.append(f"Themes you like: {', '.join(themes[:3])}")
    return reasons[:3] or ["Close in content to the movies you go for"]


def because_you_liked(engine, movies: pd.DataFrame, profile: pd.DataFrame, seeds: int = 3,
                      per_seed: int = 6, exclude_ids=()) -> list[tuple[pd.Series, pd.DataFrame]]:
    """Rows of "Because you liked X": the films most similar to the user's favourites.

    The seeds are the liked movies with the highest interest score (the most
    recent one wins a tie).  Each row leaves out everything the user has
    touched, the ids in exclude_ids, and whatever an earlier row already shows,
    so no film appears twice on the page.  Returns [(seed movie, matches)],
    where matches are catalogue rows with similarity_pct.
    """
    liked = _liked(profile, movies)
    if liked.empty:
        return []
    order = liked.assign(_recent=range(len(liked))).sort_values(
        ["score", "_recent"], ascending=False)
    skip = set(profile["movie_id"]) | {int(i) for i in exclude_ids}
    rows = []
    for index, seed in order.head(seeds).iterrows():
        found = engine.recommend(int(index), top_n=per_seed + len(skip) + 10)
        found = found[~found["id"].isin(skip)].head(per_seed)
        if found.empty:
            continue
        # recommend() re-indexes its results; map them back to catalogue rows.
        positions = [engine.index_of_id(i) for i in found["id"]]
        matches = movies.loc[positions].assign(similarity_pct=found["similarity_pct"].values)
        skip |= set(matches["id"])
        rows.append((movies.loc[index], matches))
    return rows


def finish_series(pattern: dict, movies: pd.DataFrame, limit: int = 6) -> pd.DataFrame:
    """The next unseen film of every series the user has started, oldest first."""
    ids = [i for s in pattern.get("series", []) for i in s["missing_ids"][:2]]
    order = {movie_id: n for n, movie_id in enumerate(ids)}
    found = movies[movies["id"].isin(ids)]
    return found.iloc[found["id"].map(order).argsort()].head(limit)


def site_summary(profiles: dict[str, pd.DataFrame], movies: pd.DataFrame) -> pd.Series:
    """How many liked movies of each genre, across every user (for the admin dashboard)."""
    ids = [i for p in profiles.values() for i in p.loc[p["score"] > 0, "movie_id"]]
    if not ids:
        return pd.Series(dtype=int)
    rows = movies.set_index("id").loc[[i for i in ids if i in set(movies["id"])], "genres"]
    counts = Counter(g for v in rows for g in preprocessing.split_list(v))
    return pd.Series(counts).sort_values(ascending=False)


def profile_with_details(profile: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """The learned movies as a readable table, strongest interest first."""
    if profile.empty:
        return profile
    details = movies[["id", "year", "genre_list"]]
    table = profile.merge(details, left_on="movie_id", right_on="id", how="left")
    table["year"] = table["year"].fillna(0).astype(int)
    table["score"] = table["score"].round(1)
    return table.drop(columns="id").sort_values("score", ascending=False)
