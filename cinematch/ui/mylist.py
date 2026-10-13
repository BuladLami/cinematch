"""
ui/mylist.py -- The My List page: the user's own movies.

This is where CRUD lives in CineMatch, from the user's side rather than a
database administrator's.  Each account has its own list, in
data/users/<username>/my_list.csv:

    CREATE  "+ Add to My List" in any movie's pop-up        -> add_to_list()
    READ    this page: posters, counts, filters, the table   -> read_my_list()
    UPDATE  status / rating / notes, in the pop-up or table  -> update_list_entry(),
                                                               replace_my_list()
    DELETE  "Remove from My List", or the table's Remove box -> remove_from_list()

Every action writes the user's my_list.csv straight away, and the list in turn steers
"Recommended for you".
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import config, file_manager
from . import components, state

SORTS = {
    "Recently added": ("added_on", False),
    "Recently updated": ("updated_on", False),
    "Title (A-Z)": ("title", True),
    "My rating": ("my_rating", False),
}


def recommended_for_you(movies: pd.DataFrame, engine, my_list: pd.DataFrame,
                        top_n: int = 6) -> tuple[pd.DataFrame, int]:
    """Movies matching the taste of the user's list, and how many list movies were used.

    Every listed movie counts except the ones rated below config.LIKED_RATING.
    A movie's rating is its weight (unrated counts as 7), so a 10/10 pulls the
    suggestions towards itself harder than a 6/10.  Anything already on the
    list is left out of the results.
    """
    if my_list.empty:
        return pd.DataFrame(), 0

    liked = my_list[my_list["my_rating"].isna() | (my_list["my_rating"] >= config.LIKED_RATING)]
    rows, weights = [], []
    for _, entry in liked.iterrows():
        row = engine.index_of_id(entry["movie_id"])
        if row is not None:
            rows.append(row)
            weights.append(7.0 if pd.isna(entry["my_rating"]) else float(entry["my_rating"]))
    if not rows:
        return pd.DataFrame(), 0

    found = engine.recommend_for_profile(rows, weights, top_n=top_n + len(my_list))
    found = found[~found["id"].isin(my_list["movie_id"])].head(top_n)
    return found, len(rows)


def _editor(my_list: pd.DataFrame) -> None:
    """Edit many entries at once (UPDATE), or tick Remove to delete them (DELETE)."""
    table = my_list[["movie_id", "title", "status", "my_rating", "notes"]].copy()
    table["remove"] = False

    edited = st.data_editor(
        table,
        hide_index=True,
        width="stretch",
        disabled=["movie_id", "title"],
        column_order=["title", "status", "my_rating", "notes", "remove"],
        column_config={
            "title": st.column_config.TextColumn("Movie"),
            "status": st.column_config.SelectboxColumn(
                "Status", options=config.LIST_STATUSES, required=True),
            "my_rating": st.column_config.NumberColumn(
                "My rating", min_value=1, max_value=10, step=1, format="%d / 10",
                help="Leave blank if you have not rated it"),
            "notes": st.column_config.TextColumn("Notes", width="large"),
            "remove": st.column_config.CheckboxColumn(
                "Remove", help="Tick, then press Save changes"),
        },
        key="my_list_editor",
    )

    if st.button("Save changes", type="primary", key="my_list_save"):
        try:
            to_remove = edited.loc[edited["remove"], "movie_id"].tolist()
            message = file_manager.replace_my_list(edited[~edited["remove"]],
                                                   username=state.username())
            for movie_id in to_remove:
                file_manager.remove_from_list(movie_id, username=state.username())
            if to_remove:
                message += f" Removed {len(to_remove)} movie(s)."
        except file_manager.FileManagerError as error:
            st.error(str(error))
            return
        components.flash(message)
        st.rerun()


def render(movies: pd.DataFrame, report: dict, engine) -> None:
    components.page_header(
        "My List",
        "Movies you want to watch, are watching, or have seen -- with your own "
        "ratings and notes.",
        eyebrow="Your movies",
    )
    components.show_flash()

    my_list = file_manager.read_my_list(state.username())
    if my_list.empty:
        st.info("Your list is empty. Open any movie on Home or Discover and press "
                "**+ Add to My List**.")
        return

    # ------------------------------------------------------------- READ
    rated = my_list["my_rating"].dropna()
    counts = my_list["status"].value_counts()
    components.metric_tiles(
        [("On my list", f"{len(my_list)}", "movies saved")]
        + [(status, f"{int(counts.get(status, 0))}", "movies") for status in config.LIST_STATUSES]
        + [("My average rating", f"{rated.mean():.1f}" if len(rated) else "-",
            f"from {len(rated)} rated")]
    )
    st.write("")

    filter_column, sort_column = st.columns([3, 1], vertical_alignment="bottom")
    with filter_column:
        shown_status = st.segmented_control(
            "Show", ["All"] + config.LIST_STATUSES, default="All", key="my_list_filter")
    with sort_column:
        sort_by = st.selectbox("Sort by", list(SORTS), key="my_list_sort")

    column, ascending = SORTS[sort_by]
    shown = my_list if shown_status in (None, "All") else my_list[my_list["status"] == shown_status]
    shown = shown.sort_values(column, ascending=ascending, na_position="last")

    # Keep the list's order while pulling each movie's full record.
    catalogue_rows = movies.reset_index().merge(
        shown[["movie_id"]], left_on="id", right_on="movie_id", how="inner", sort=False)
    order = {movie_id: n for n, movie_id in enumerate(shown["movie_id"])}
    catalogue_rows = (catalogue_rows.assign(_order=catalogue_rows["id"].map(order))
                      .sort_values("_order").set_index("index"))

    if catalogue_rows.empty:
        st.caption("No movies with this status yet.")
    else:
        clicked = components.movie_grid(catalogue_rows, per_row=6, key_prefix="mylist",
                                        minimal=True)
        if clicked is not None:
            components.movie_dialog(movies.loc[clicked])
        st.caption("Click a poster to change its status, rating or notes, or to remove it.")

    # ------------------------------------------------ UPDATE / DELETE in bulk
    with st.expander("Edit several movies at once"):
        _editor(my_list)

    st.download_button(
        "Download my list (CSV)",
        data=file_manager.list_as_csv(my_list),
        file_name="cinematch_my_list.csv",
        mime="text/csv",
        on_click="ignore",
    )

    # ----------------------------------------------------- recommendations
    for_you, based_on = recommended_for_you(movies, engine, my_list, top_n=6)
    if not for_you.empty:
        st.divider()
        components.row_header("Recommended for you",
                              f"based on {based_on} movie(s) on your list")
        rows = [engine.index_of_id(i) for i in for_you["id"]]
        clicked = components.movie_grid(movies.loc[[r for r in rows if r is not None]],
                                        per_row=6, key_prefix="mylist_for_you", minimal=True)
        if clicked is not None:
            components.movie_dialog(movies.loc[clicked])
