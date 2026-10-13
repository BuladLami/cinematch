"""
ui/dashboard.py -- The Home page.

It opens the way a streaming site opens: shelves of posters.  Once CineMatch
has learned something from what the user searches, opens and saves (see
activity.py), the first shelf is picked from that pattern, followed by what
they opened lately and "Continue watching" from My List; below them sit what
is popular, what is best rated, and a shelf for whichever genre the user picks
(browsing a genre is itself recorded as a signal).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import activity, file_manager, patterns, preprocessing
from . import components, state


def _shelf(title: str, hint: str, shelf: pd.DataFrame, key: str, movies: pd.DataFrame) -> None:
    """One row of clickable posters; a click opens the movie's details pop-up."""
    if shelf.empty:
        return
    components.row_header(title, hint)
    clicked = components.movie_grid(shelf, per_row=6, key_prefix=key, minimal=True)
    if clicked is not None:
        components.movie_dialog(movies.loc[clicked])


def _record_genre(username: str) -> None:
    """Choosing a genre pill is a signal of taste (the default choice is not)."""
    genre = st.session_state.get("home_genre")
    if genre:
        activity.record(username, "genre", detail=genre)


def render(movies: pd.DataFrame, report: dict, engine) -> None:
    components.page_header(
        "Tonight's shortlist starts here",
        f"{len(movies):,} films to explore. Save the ones you like to My List and "
        "CineMatch will keep suggesting more like them.",
        eyebrow="CineMatch",
    )

    username = state.username()
    my_list = file_manager.read_my_list(username)

    # ---------------------------------------------------------- personal
    log = activity.read_activity(username)
    profile, genre_interest = activity.build_profile(log, my_list, movies)
    pattern = patterns.analyze(profile, movies, genre_interest)
    if pattern:
        picks = patterns.recommend(engine, movies, profile, pattern, top_n=6,
                                   behaviour=state.behaviour_model(movies, exclude=username))
        top = " and ".join(pattern["top_genres"][:2])
        _shelf("Picked for you", f"learned from what you browse: lots of {top}"
               if top else "learned from what you browse", picks, "pattern", movies)

    opened = log.loc[log["event"] == "view", "movie_id"].iloc[::-1]
    opened = pd.to_numeric(opened, errors="coerce").dropna().astype(int).drop_duplicates()
    order = {movie_id: n for n, movie_id in enumerate(opened.head(6))}
    recent = movies[movies["id"].isin(order)]
    recent = recent.iloc[recent["id"].map(order).argsort()]
    _shelf("Recently viewed", "pick up where you left off", recent, "recent", movies)

    watching_ids = my_list.loc[my_list["status"] == "Watching", "movie_id"].tolist()
    watching = movies[movies["id"].isin(watching_ids)].head(6)
    _shelf("Continue watching", "from My List", watching, "watching", movies)

    # ----------------------------------------------------------- general
    _shelf("Trending now", "", movies.nlargest(6, "popularity"), "trending", movies)

    # A vote-count floor keeps a single enthusiastic 10/10 out of the top row.
    credible = movies[movies["vote_count"] >= 1000]
    _shelf("Highest rated", "loved by thousands of viewers",
           credible.nlargest(6, "vote_average"), "top_rated", movies)

    # --------------------------------------------------------- by genre
    genres = preprocessing.all_genres(movies)
    components.row_header("Browse by genre")
    genre = st.pills("Genre", genres[:14], default=genres[0] if genres else None,
                     label_visibility="collapsed", key="home_genre",
                     on_change=_record_genre, args=(username,))
    if genre:
        in_genre = movies[movies["genre_list"].str.contains(genre, regex=False)]
        clicked = components.movie_grid(in_genre.nlargest(12, "popularity"), per_row=6,
                                        key_prefix="genre", minimal=True)
        if clicked is not None:
            components.movie_dialog(movies.loc[clicked])

    st.divider()
    with st.expander("How does CineMatch choose movies?"):
        st.markdown(
            """
CineMatch looks at what each film **is about**: its genres, plot keywords,
cast, director, the series it belongs to, and its plot summary. Two films that
share a lot of those are a close match.

- **Movies like this** (on Discover) compares every film with the one you pick.
- **Picked for you** is learned from how you use CineMatch. The titles you
  search, the movies you open, the ones you ask "movies like this" for, the
  genres you browse and everything on My List each count, recent actions
  more than old ones. CineMatch blends what those movies are about with what
  other people who liked the same films went for. Movies you rated below 6
  are left out.
- **Recommended** has your top picks, "Because you liked..." rows of similar
  films and the next films of series you started. It also shows what
  CineMatch has noticed about your taste, and lets you forget movies or clear
  it all.

Nothing is sent anywhere else: your activity stays inside CineMatch, in
`data/users/`.
            """
        )
