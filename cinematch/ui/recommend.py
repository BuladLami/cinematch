"""
ui/recommend.py -- The Discover page (browse, then get recommendations).

The flow the user follows:
    1. search a title, or filter by genre / rating / decade
    2. click a poster from the results wall          -> details pop-up
    3. press "Find movies like this"                 -> recommendations pop-up
    4. choose 5 or 10                                -> matches with a score
    5. add any of them to My List                    -> saved to my_list.csv

Searches, the genre filter and "Find movies like this" are recorded in the
user's activity log, which the watch pattern is learned from (activity.py).
"""

from __future__ import annotations

import random

import pandas as pd
import streamlit as st

from .. import activity, config
from . import components, state

PAGE_SIZE = 15          # posters shown per page of browse results
CAROUSEL_SIZE = 7       # page-number buttons shown at once in the pager
DECADES = ["Any decade", "2010s", "2000s", "1990s", "1980s", "1970s", "Before 1970"]


# ---------------------------------------------------------------------------
# BROWSING
# ---------------------------------------------------------------------------
def _decade_filter(movies: pd.DataFrame, choice: str) -> pd.DataFrame:
    """Keep only the movies released inside the chosen decade."""
    if choice == "Any decade":
        return movies
    if choice == "Before 1970":
        return movies[(movies["year"] > 0) & (movies["year"] < 1970)]
    start = int(choice[:4])
    return movies[(movies["year"] >= start) & (movies["year"] < start + 10)]


def _record_search(movies: pd.DataFrame) -> None:
    activity.record_search(state.username(), st.session_state.get("search_query"), movies)


def _record_genre() -> None:
    genre = st.session_state.get("browse_genre")
    if genre and genre != "All genres":
        activity.record(state.username(), "genre", detail=genre)


def _browse(movies: pd.DataFrame, engine) -> pd.DataFrame:
    """The search box plus the three filters, returning the matching movies."""
    # Every control carries a visible label and the row is bottom-aligned, so
    # the four input boxes line up on one baseline.
    search_column, genre_column, rating_column, decade_column = st.columns(
        [3, 1.4, 1.4, 1.4], gap="medium", vertical_alignment="bottom",
    )

    with search_column:
        # A selectbox filters its list in the browser on every keystroke, so
        # matching titles appear while typing; accept_new_options still lets
        # a partial word ("dark") be searched as free text.
        titles = sorted(set(movies["title"].astype(str)), key=str.lower)
        query = st.selectbox(
            "Search",
            titles,
            index=None,
            placeholder="Search a title...  e.g. Dark Knight, Toy Story, Avatar",
            accept_new_options=True,
            key="search_query",
            on_change=_record_search, args=(movies,),
        ) or ""
    with genre_column:
        # genre_list is already "Action, Adventure, Science Fiction", so the
        # distinct genres are one comma-split away.
        genres = sorted({g.strip() for row in movies["genre_list"]
                         for g in str(row).split(",") if g.strip()})
        genre = st.selectbox("Genre", ["All genres"] + genres, key="browse_genre",
                             on_change=_record_genre)
    with rating_column:
        min_rating = st.selectbox("Minimum rating", ["Any rating", "6+", "7+", "8+"])
    with decade_column:
        decade = st.selectbox("Decade", DECADES)

    # Search first (it is the strongest signal), then narrow with the filters.
    results = engine.search(query, limit=400) if query else movies

    if genre != "All genres":
        results = results[results["genre_list"].astype(str).str.contains(genre, regex=False)]
    if min_rating != "Any rating":
        results = results[results["vote_average"] >= float(min_rating[0])]
    results = _decade_filter(results, decade)

    # With no search term, the wall should still look like a front page, so the
    # most popular titles lead.
    if not query and "popularity" in results.columns:
        results = results.sort_values("popularity", ascending=False)

    return results


def _choose(index) -> None:
    """Make a movie the pick that recommendations are built from."""
    st.session_state["chosen_index"] = index
    st.session_state["show_recommendations"] = True


def _pick(movie: pd.Series, index) -> None:
    """"Find movies like this": a strong sign of taste, so it is recorded."""
    activity.record_movie(state.username(), "pick", movie)
    _choose(index)


def _show_details(movie: pd.Series, index) -> None:
    """Open the detail pop-up, with a button to base recommendations on the movie."""
    components.movie_dialog(movie, action_label="Find movies like this",
                            on_action=lambda: _pick(movie, index))


def _page_bounds(results: pd.DataFrame) -> tuple[int, int]:
    """The current browse page and the last page, both zero-based."""
    last_page = max(0, (len(results) - 1) // PAGE_SIZE)
    return min(st.session_state.get("browse_page", 0), last_page), last_page


def _page_carousel(page: int, last_page: int, key: str, padding: float = 2) -> None:
    """A sliding row of page-number buttons, with the current page highlighted."""
    # Keep the current page in the middle of the window where possible, and
    # slide the window instead of running past either end.
    start = max(0, min(page - CAROUSEL_SIZE // 2, last_page - CAROUSEL_SIZE + 1))
    numbers = range(start, min(last_page, start + CAROUSEL_SIZE - 1) + 1)

    # Equal spacer columns on both sides centre the buttons; a wider padding
    # squeezes them closer together.
    columns = st.columns([padding] + [1] * len(numbers) + [padding], gap="small")[1:-1]

    for column, number in zip(columns, numbers):
        with column:
            if st.button(str(number + 1), key=f"{key}_page_{number}", width="stretch",
                         type="primary" if number == page else "secondary"):
                st.session_state["browse_page"] = number
                st.rerun()


def _browse_wall(results: pd.DataFrame) -> None:
    """Draw the current page of posters and handle clicks and paging."""
    if results.empty:
        st.warning("Nothing matches those filters. Try widening them.")
        return

    page, last_page = _page_bounds(results)
    window = results.iloc[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]

    components.row_header(
        "Browse results",
        f"{len(results):,} titles  -  showing {page * PAGE_SIZE + 1}"
        f"-{page * PAGE_SIZE + len(window)}",
    )

    clicked = components.movie_grid(window, per_row=5, key_prefix="browse", minimal=True)
    if clicked is not None:
        _show_details(results.loc[clicked], clicked)

    if last_page:
        previous, indicator, following = st.columns([1, 3, 1])
        with previous:
            if st.button("Previous", disabled=page == 0, width="stretch"):
                st.session_state["browse_page"] = page - 1
                st.rerun()
        with indicator:
            _page_carousel(page, last_page, key="bottom")
            st.markdown(
                f"<p style='text-align:center;color:#97A0B5;padding-top:.45rem'>"
                f"Page {page + 1} of {last_page + 1}</p>",
                unsafe_allow_html=True,
            )
        with following:
            if st.button("Next", disabled=page >= last_page, width="stretch"):
                st.session_state["browse_page"] = page + 1
                st.rerun()


# ---------------------------------------------------------------------------
# THE RECOMMENDATIONS POP-UP
# ---------------------------------------------------------------------------
def _clear_pick() -> None:
    st.session_state.pop("chosen_index", None)


@st.dialog("Movies like your pick", width="large", on_dismiss=_clear_pick)
def _recommendations_dialog(movies: pd.DataFrame, engine) -> None:
    """The chosen movie, its closest matches, and the save/export actions.

    Everything in here reruns as a fragment, so changing "How many?" or
    clicking a match updates the pop-up without closing it.
    """
    chosen_index = st.session_state.get("chosen_index")
    if chosen_index is None or chosen_index not in movies.index:
        st.info("That movie is no longer in the catalogue.")
        return
    chosen = movies.loc[chosen_index]

    # ---------------- the selected movie ----------------
    components.row_header("Your pick", "recommendations are built from this film")
    components.movie_card(chosen)
    components.list_controls(chosen, key=f"pick_{int(chosen['id'])}")
    st.write("")

    count_column, explain_column, cancel_column = st.columns(
        [1, 1.6, 1], vertical_alignment="bottom",
    )
    with count_column:
        top_n = st.selectbox("How many?", config.TOP_N_CHOICES, index=0)
    with explain_column:
        explain = st.checkbox("Explain each match", value=True)
    with cancel_column:
        if st.button("Cancel", width="stretch", help="Clear this pick and close"):
            _clear_pick()
            st.rerun()

    # ---------------- the matches ----------------
    found = engine.recommend(chosen_index, top_n=top_n)
    if found.empty:
        st.warning(
            "This movie has no words in common with any other record, so no "
            "recommendation can be made. It is probably missing its genres, "
            "keywords and overview."
        )
        return

    components.row_header(
        f'Because you picked "{chosen["title"]}"',
        f"{len(found)} closest matches  -  click one to explore from it",
    )

    # recommend() hands back a re-indexed table, so each result is looked up
    # again by its id (titles are not unique) to explain the match.
    why_lookup = {}
    if explain:
        for index, movie in found.iterrows():
            original = engine.index_of_id(movie["id"])
            if original is not None:
                terms = engine.shared_terms(chosen_index, original, limit=4)
                why_lookup[index] = engine.readable(terms)

    clicked = components.movie_grid(found, per_row=5, ranked=True,
                                    why_lookup=why_lookup, key_prefix="match")
    if clicked is not None:
        # A pop-up cannot open another pop-up, so a clicked match becomes the
        # new pick and the pop-up redraws around it.
        st.session_state["chosen_index"] = engine.index_of_id(found.loc[clicked, "id"])
        st.rerun(scope="fragment")

    with st.expander("See the same results as a table"):
        components.movie_table(found, extra_columns=["similarity_pct"])


# ---------------------------------------------------------------------------
# THE PAGE
# ---------------------------------------------------------------------------
def render(movies: pd.DataFrame, report: dict, engine) -> None:
    components.page_header(
        "Find your next movie",
        "Search or filter the catalogue, pick a film you already like, and "
        "CineMatch will find the ones most like it.",
        eyebrow="Discover",
    )

    # ---------------- browse ----------------
    results = _browse(movies, engine)

    surprise_column, carousel_column = st.columns([1, 4])
    with surprise_column:
        if st.button("Surprise me", width="stretch",
                     help="Jump to a random well-rated movie"):
            pool = movies[movies["vote_count"] >= 500]
            pool = pool if not pool.empty else movies
            _choose(random.choice(list(pool.index)))
            st.rerun()
    with carousel_column:
        page, last_page = _page_bounds(results)
        if last_page:
            _page_carousel(page, last_page, key="top", padding=4)

    _browse_wall(results)

    # ---------------- recommend ----------------
    # _choose() raises the flag; popping it means the pop-up opens once and a
    # later rerun of the page does not keep reopening it.
    if st.session_state.pop("show_recommendations", False):
        _recommendations_dialog(movies, engine)
