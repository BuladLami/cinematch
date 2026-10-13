"""
app.py -- CineMatch entry point.

Run it with:      streamlit run app.py

This file deliberately does almost nothing.  Its only jobs are to set up the
page, make sure someone is signed in, load the data once, and hand control to
whichever page the user picked in the sidebar.  All the real work lives in the
cinematch package:

    cinematch/config.py         paths and settings
    cinematch/accounts.py       sign-in, the admin / user roles, the activity log
    cinematch/file_manager.py   reading and writing files (catalogue, My List)
    cinematch/activity.py       recording what each user does, and learning from it
    cinematch/collaborative.py  "people who liked this also liked" (item-item CF)
    cinematch/patterns.py       finding the watch pattern + recommending from it
    cinematch/preprocessing.py  cleaning the data
    cinematch/recommender.py    the TF-IDF + cosine similarity model
    cinematch/posters.py        poster images
    cinematch/ui/               the pages
"""

import streamlit as st

from cinematch import accounts, config, file_manager
from cinematch.ui import (admin, components, dashboard, login, mylist, recommend, state,
                          watch_pattern)

# The label in the sidebar -> the function that draws that page.
USER_PAGES = {
    "Home": dashboard.render,
    "Discover": recommend.render,
    "Recommended": watch_pattern.render,
    "My List": mylist.render,
}
# An admin gets these on top of every user page.
ADMIN_PAGES = {
    "Admin dashboard": admin.render_dashboard,
    "Manage users": admin.render_users,
    "User patterns": admin.render_patterns,
    "Activity log": admin.render_log,
}


def account_box() -> None:
    """Who is signed in, a password change, and the sign-out button."""
    user = state.current_user()
    st.caption(f"Signed in as **{user['username']}** ({user['role']})")
    if user["role"] == "admin" and accounts.using_default_password():
        st.warning("You are still using the default admin password. Change it below.")
    with st.expander("Change my password"):
        with st.form("change_password", clear_on_submit=True):
            old = st.text_input("Current password", type="password")
            new = st.text_input("New password", type="password")
            if st.form_submit_button("Change password"):
                try:
                    st.success(accounts.change_password(user["username"], new,
                                                        old_password=old))
                except accounts.AccountError as error:
                    st.error(str(error))
    if st.button("Sign out", width="stretch"):
        accounts.log(user["username"], "signed out")
        st.session_state.clear()
        st.rerun()


def main() -> None:
    st.set_page_config(page_title=f"{config.APP_NAME} - {config.APP_TAGLINE}",
                       page_icon="🎬", layout="wide")
    components.inject_css()
    config.ensure_dirs()

    # Nobody signed in (or the account was deleted since): show the sign-in screen.
    user = state.current_user()
    if user is None or accounts.get_user(user["username"]) is None:
        st.session_state.pop("user", None)
        login.render()
        return
    # Pick up a role change an admin made while this person was signed in.
    st.session_state["user"] = user = accounts.get_user(user["username"])
    pages = {**USER_PAGES, **(ADMIN_PAGES if user["role"] == "admin" else {})}

    with st.sidebar:
        st.markdown(
            '<div class="cm-brand">'
            '<div class="cm-brand-mark">🎬</div>'
            '<div class="cm-brand-name">Cine<span>Match</span></div></div>',
            unsafe_allow_html=True,
        )
        st.caption(config.APP_TAGLINE)
        st.write("")
        page_name = st.radio("Go to", list(pages), label_visibility="collapsed")
        st.divider()

    # Load the data once; both the table and the model are cached.
    try:
        movies, report, engine = state.get_data()
    except file_manager.FileManagerError as error:
        st.error(str(error))
        st.info(
            f"CineMatch expects the dataset in:\n\n`{config.SOURCE_DIR}`\n\n"
            "with movies_metadata.csv, credits.csv and keywords.csv inside it."
        )
        return

    with st.sidebar:
        st.metric("Movies to explore", f"{len(movies):,}")
        st.metric("On My List", f"{len(file_manager.read_my_list(user['username'])):,}")
        st.divider()
        account_box()

    pages[page_name](movies, report, engine)


if __name__ == "__main__":
    main()
