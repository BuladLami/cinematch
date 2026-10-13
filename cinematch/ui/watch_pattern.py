"""
ui/watch_pattern.py -- The "Recommended" page.

Nothing is uploaded.  CineMatch learns the user's watch pattern from what they
do on the site (see activity.py): the titles they search, the movies they
open, the ones they ask "Find movies like this" for, the genres they browse,
and everything on My List.  This page turns that pattern into movies:

    Top picks for you         the best fits for the pattern, each with its reasons
    Because you liked X       the films most similar to each of their favourites
    Finish the series         the next films of series they started

Below the picks, two sections are folded away:

    What CineMatch has noticed       the pattern in plain sentences, with two charts
    Manage what CineMatch learned    every movie that counts (forget some, or clear
                                     everything) and the recent actions

Until CineMatch has learned something, the page shows what is popular and
explains how to get personal picks.

show_for() draws everything for one username, so the admin's "User patterns"
page reuses it to look at any user's data.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import accounts, activity, config, file_manager, patterns
from . import components, state

EVENT_LABELS = {"search": "Searched", "view": "Opened", "pick": "Found movies like",
                "genre": "Browsed genre"}


def _grid(shelf: pd.DataFrame, movies: pd.DataFrame, key: str, **options) -> None:
    """A shelf of posters; a click opens the movie's details pop-up."""
    clicked = components.movie_grid(shelf, per_row=6, key_prefix=key, **options)
    if clicked is not None:
        components.movie_dialog(movies.loc[clicked])


# ---------------------------------------------------------------------------
# THE PICKS
# ---------------------------------------------------------------------------
def _picks(username: str, movies: pd.DataFrame, engine, profile: pd.DataFrame,
           pattern: dict) -> None:
    behaviour = state.behaviour_model(movies, exclude=username)
    recs = patterns.recommend(engine, movies, profile, pattern, top_n=12,
                              behaviour=behaviour)
    shown = set(recs["id"]) if len(recs) else set()

    if len(recs):
        top = " and ".join(pattern["top_genres"][:2])
        components.row_header("Top picks for you",
                              f"you go for {top}" if top else "picked from what you browse")
        _grid(recs, movies, f"pattern_recs_{username}", ranked=True,
              why_lookup=dict(zip(recs.index, recs["reasons"])), why_prefix="Why: ")

    for seed, matches in patterns.because_you_liked(engine, movies, profile,
                                                    exclude_ids=shown):
        components.row_header(f"Because you liked {seed['title']}", "similar films")
        _grid(matches, movies, f"because_{username}_{int(seed['id'])}", minimal=True)
        shown |= set(matches["id"])

    series = patterns.finish_series(pattern, movies)
    series = series[~series["id"].isin(shown)]
    if not series.empty:
        components.row_header("Finish the series", "the next films in series you started")
        _grid(series, movies, f"pattern_series_{username}", minimal=True)

    info = behaviour.info
    sources = [f"{info['app_viewers']} other CineMatch account(s)"]
    if info["movielens_viewers"]:
        sources.insert(0, f"{info['movielens_viewers']:,} MovieLens viewers")
    st.caption("Top picks blend similar content (35%), what people with the same taste "
               "went for (20%), your genres (15%), your decade (10%), the ratings you "
               "usually go for (5%), and your directors, actors and unfinished series "
               f"(15%). The “same taste” part learns from {' and '.join(sources)}.")


# ---------------------------------------------------------------------------
# WHAT CINEMATCH HAS NOTICED  (the pattern, folded away under the picks)
# ---------------------------------------------------------------------------
def _taste_summary(pattern: dict) -> None:
    st.markdown("\n".join(f"- {line}" for line in pattern["insights"]))

    genre_column, decade_column = st.columns(2, gap="large")
    with genre_column:
        st.markdown("**Genres you go for**")
        genres = pattern["genre_counts"].head(8).rename_axis("genre").reset_index(name="interest")
        st.bar_chart(genres, x="genre", y="interest", horizontal=True, sort="-interest",
                     color=components.ACCENT, x_label="", y_label="", height=260)
    with decade_column:
        st.markdown("**When your movies are from**")
        decades = pattern["decade_counts"].rename_axis("decade").reset_index(name="interest")
        st.bar_chart(decades, x="decade", y="interest", sort=False,
                     color=components.ACCENT, x_label="", y_label="", height=260)


# ---------------------------------------------------------------------------
# MANAGE WHAT CINEMATCH LEARNED
# ---------------------------------------------------------------------------
def _learned_tab(username: str, movies: pd.DataFrame, profile: pd.DataFrame,
                 acting_user: str) -> None:
    st.caption("Every movie that shapes your recommendations. Searching, opening and saving a "
               "movie raise its score; old actions fade (half the weight every "
               f"{config.HALF_LIFE_DAYS} days). A movie you rate below "
               f"{config.LIKED_RATING} on My List scores 0 and is never recommended.")
    table = patterns.profile_with_details(profile, movies)
    table["forget"] = False
    edited = st.data_editor(
        table[["title", "year", "genre_list", "score", "views", "searches", "picks",
               "list_status", "my_rating", "forget", "movie_id"]],
        hide_index=True, width="stretch",
        disabled=["title", "year", "genre_list", "score", "views", "searches", "picks",
                  "list_status", "my_rating"],
        column_order=["title", "year", "genre_list", "score", "views", "searches", "picks",
                      "list_status", "my_rating", "forget"],
        column_config={
            "title": "Movie", "genre_list": "Genres",
            "year": st.column_config.NumberColumn("Year", format="%d"),
            "score": st.column_config.ProgressColumn(
                "Interest", format="%.1f", min_value=0,
                max_value=float(max(table["score"].max(), 1))),
            "views": st.column_config.NumberColumn("Opened", format="%d"),
            "searches": st.column_config.NumberColumn("Searched", format="%d"),
            "picks": st.column_config.NumberColumn("Like this", format="%d"),
            "list_status": "On My List",
            "my_rating": st.column_config.NumberColumn("Rating", format="%.0f / 10"),
            "forget": st.column_config.CheckboxColumn("Forget"),
        },
        key=f"learned_editor_{username}",
    )
    forget_column, clear_column, download_column = st.columns(3)
    with forget_column:
        if st.button("Forget ticked movies", key=f"learned_forget_{username}"):
            ids = edited.loc[edited["forget"], "movie_id"].tolist()
            if ids:
                components.flash(activity.forget_movies(username, ids))
                accounts.log(acting_user, "edited activity", f"{username}: forgot {len(ids)}")
                st.rerun()
    with clear_column:
        confirm = st.checkbox("Yes, clear all activity", key=f"learned_confirm_{username}")
        if st.button("Clear activity", disabled=not confirm, key=f"learned_clear_{username}"):
            components.flash(activity.clear_activity(username))
            accounts.log(acting_user, "cleared activity", username)
            st.rerun()
    with download_column:
        st.download_button("Download (CSV)", file_manager.list_as_csv(table.drop(columns="forget")),
                           file_name=f"{username}_learned_movies.csv", mime="text/csv",
                           on_click="ignore", key=f"learned_download_{username}")


def _activity_tab(log: pd.DataFrame) -> None:
    recent = log.iloc[::-1].head(200).copy()
    recent["what"] = recent["event"].map(EVENT_LABELS).fillna(recent["event"])
    recent["about"] = recent["title"].where(recent["title"].ne(""), recent["detail"])
    st.dataframe(recent[["timestamp", "what", "about"]], hide_index=True, width="stretch",
                 height=420, column_config={"timestamp": "When", "what": "Action",
                                            "about": "Movie, search or genre"})
    searches = log.loc[(log["event"] == "search") & log["detail"].ne(""), "detail"]
    if len(searches):
        st.markdown("**Your most frequent searches**")
        st.markdown(components.genre_chips(", ".join(
            f"{q} ×{n}" for q, n in searches.str.lower().value_counts().head(10).items())),
            unsafe_allow_html=True)



# ---------------------------------------------------------------------------
# PAGE
# ---------------------------------------------------------------------------
def _getting_started(movies: pd.DataFrame, profile: pd.DataFrame, username: str) -> None:
    """Nothing learned yet: say how to get personal picks, and show what is popular."""
    st.info(
        "Your recommendations appear here as you use CineMatch. **Search** for titles on "
        "Discover, **open** movies you are curious about, press **Find movies like "
        "this**, and add films to **My List**: each one teaches CineMatch your taste."
    )
    if len(profile):
        st.caption(f"Everything so far is rated below {config.LIKED_RATING}, so there is "
                   "nothing you liked to learn from yet.")
    components.row_header("Popular right now", "a place to start")
    _grid(movies.nlargest(12, "popularity"), movies, f"start_{username}", minimal=True)


def show_for(username: str, movies: pd.DataFrame, engine, acting_user: str) -> None:
    """One user's recommendations, and what they were learned from (users and the admin)."""
    log = activity.read_activity(username)
    profile, genre_interest = activity.build_profile(
        log, file_manager.read_my_list(username), movies)
    pattern = patterns.analyze(profile, movies, genre_interest)

    if not pattern:
        _getting_started(movies, profile, username)
        return

    _picks(username, movies, engine, profile, pattern)

    st.write("")
    with st.expander("What CineMatch has noticed about your taste"):
        _taste_summary(pattern)
    with st.expander("Manage what CineMatch learned"):
        learned_tab, activity_tab = st.tabs(["Movies that count", "Recent activity"])
        with learned_tab:
            _learned_tab(username, movies, profile, acting_user)
        with activity_tab:
            _activity_tab(log)


def render(movies: pd.DataFrame, report: dict, engine) -> None:
    components.page_header(
        "Recommended",
        "Movies picked for you from what you search, open and save. The more you "
        "browse, the better they get.",
        eyebrow="For you",
    )
    components.show_flash()
    show_for(state.username(), movies, engine, acting_user=state.username())
