"""
ui/admin.py -- The pages only an admin sees.

    Admin dashboard   how many accounts, how much activity has been learned
                      from; the genres liked across every user; the latest log
    Manage users      CREATE an account, READ the list, UPDATE a role or a
                      password, DELETE an account (and all of its files) --
                      all in the users table of data/cinematch.db
    User patterns     open any user's learned pattern: what it was learned
                      from, their recommendations, their recent actions
    Activity log      every sign-in, account change and admin action, with a filter

The admin also keeps every user page (Home, Discover, Recommended,
My List) for their own account.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import accounts, activity, config, file_manager, patterns
from . import components, state, watch_pattern


def _all_profiles(movies: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {name: activity.profile_for(name, movies)[0]
            for name in accounts.read_users()["username"]}


# ---------------------------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------------------------
def render_dashboard(movies: pd.DataFrame, report: dict, engine) -> None:
    components.page_header("Admin dashboard", "Accounts, what CineMatch has learned and "
                           "what people go for.", eyebrow="Admin")
    users = accounts.read_users()
    profiles = _all_profiles(movies)
    actions = sum(len(activity.read_activity(name)) for name in users["username"])
    learning = sum(1 for p in profiles.values() if (p["score"] > 0).any())

    components.metric_tiles([
        ("Accounts", f"{len(users)}", f"{int((users['role'] == 'admin').sum())} admin(s)"),
        ("Users with a pattern", f"{learning}", "enough activity to learn from"),
        ("Actions recorded", f"{actions:,}", "searches, views, picks, genres"),
        ("Movies learned from", f"{sum(len(p) for p in profiles.values())}",
         "across every user"),
    ])
    st.write("")

    genre_column, activity_column = st.columns([1, 1.2], gap="large")
    with genre_column:
        st.markdown("#### Genres people go for")
        genres = patterns.site_summary(profiles, movies)
        if genres.empty:
            st.caption("Nobody has used the site enough to learn from yet.")
        else:
            data = genres.head(10).rename_axis("genre").reset_index(name="movies")
            st.bar_chart(data, x="genre", y="movies", horizontal=True, sort="-movies",
                         color=components.ACCENT, x_label="Movies", y_label="")
    with activity_column:
        st.markdown("#### Latest activity")
        st.dataframe(accounts.read_log().head(12), hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# MANAGE USERS
# ---------------------------------------------------------------------------
def _user_table(users: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, user in users.iterrows():
        name = user["username"]
        rows.append({
            "username": name,
            "role": user["role"],
            "created_on": user["created_on"],
            "last_login": user["last_login"] or "never",
            "actions": len(activity.read_activity(name)),
            "my_list": len(file_manager.read_my_list(name)),
        })
    return pd.DataFrame(rows)


def _attempt(action, *args, **kwargs) -> None:
    """Run an account action; flash its message, or show its error."""
    try:
        components.flash(action(*args, **kwargs))
    except accounts.AccountError as error:
        st.error(str(error))
        return
    st.rerun()


def render_users(movies: pd.DataFrame, report: dict, engine) -> None:
    components.page_header("Manage users", "Create accounts, reset passwords, change "
                           "roles and delete accounts.", eyebrow="Admin")
    components.show_flash()
    me = state.username()
    users = accounts.read_users()

    st.dataframe(_user_table(users), hide_index=True, width="stretch", column_config={
        "username": "Username", "role": "Role", "created_on": "Created",
        "last_login": "Last sign-in", "actions": "Actions recorded",
        "my_list": "On My List"})
    db = accounts.database_info()
    st.caption(f"Accounts are stored in the {db['kind']} database `{db['where']}` "
               f"({db['size_kb']:.0f} KB): {db['admins']} admin(s), {db['users']} user(s). "
               + ("Every device signs in against it." if db["online"] else
                  "It is on this computer only; set DATABASE_URL to share accounts "
                  "between devices."))

    create_column, edit_column = st.columns(2, gap="large")

    with create_column:
        st.markdown("#### Create an account")
        with st.form("admin_create_user", clear_on_submit=True):
            name = st.text_input("Username")
            password = st.text_input("Password", type="password")
            role = st.selectbox("Role", config.ROLES)
            if st.form_submit_button("Create account", type="primary"):
                _attempt(accounts.create_user, name, password, role, created_by=me)

    with edit_column:
        st.markdown("#### Change an account")
        chosen = st.selectbox("Account", users["username"], key="admin_pick_user")
        current_role = users.loc[users["username"] == chosen, "role"].iloc[0]

        role_column, role_button = st.columns([2, 1], vertical_alignment="bottom")
        with role_column:
            new_role = st.selectbox("Role", config.ROLES,
                                    index=config.ROLES.index(current_role),
                                    key=f"admin_role_{chosen}")
        with role_button:
            if st.button("Save role", width="stretch", disabled=new_role == current_role):
                _attempt(accounts.set_role, chosen, new_role, changed_by=me)

        password_column, password_button = st.columns([2, 1], vertical_alignment="bottom")
        with password_column:
            new_password = st.text_input("New password", type="password",
                                         key=f"admin_password_{chosen}")
        with password_button:
            if st.button("Reset password", width="stretch"):
                _attempt(accounts.change_password, chosen, new_password, changed_by=me)

        st.write("")
        confirm = st.checkbox(f'Delete "{chosen}" with their list and activity',
                              key=f"admin_confirm_delete_{chosen}",
                              disabled=chosen == me)
        if st.button("Delete account", disabled=not confirm or chosen == me):
            _attempt(accounts.delete_user, chosen, deleted_by=me)
        if chosen == me:
            st.caption("You cannot delete the account you are signed in with.")


# ---------------------------------------------------------------------------
# USER PATTERNS
# ---------------------------------------------------------------------------
def render_patterns(movies: pd.DataFrame, report: dict, engine) -> None:
    components.page_header("User patterns", "Open the pattern CineMatch has learned for "
                           "any user, what it was learned from, and their recommendations.",
                           eyebrow="Admin")
    components.show_flash()
    users = _user_table(accounts.read_users())
    labels = {row["username"]: f"{row['username']}  ({row['actions']} actions, "
                               f"{row['my_list']} on My List)" for _, row in users.iterrows()}
    chosen = st.selectbox("User", list(labels), format_func=labels.get,
                          key="admin_pattern_user")
    st.divider()
    watch_pattern.show_for(chosen, movies, engine, acting_user=state.username())


# ---------------------------------------------------------------------------
# ACTIVITY LOG
# ---------------------------------------------------------------------------
def render_log(movies: pd.DataFrame, report: dict, engine) -> None:
    components.page_header("Activity log", "Every sign-in, account change and admin "
                           "action, newest first.", eyebrow="Admin")
    log = accounts.read_log()
    if log.empty:
        st.caption("Nothing has been logged yet.")
        return

    user_column, action_column = st.columns(2)
    with user_column:
        who = st.multiselect("Users", sorted(log["username"].unique()))
    with action_column:
        what = st.multiselect("Actions", sorted(log["action"].unique()))
    shown = log
    if who:
        shown = shown[shown["username"].isin(who)]
    if what:
        shown = shown[shown["action"].isin(what)]

    st.dataframe(shown, hide_index=True, width="stretch", height=480, column_config={
        "timestamp": "When", "username": "User", "action": "Action", "details": "Details"})
    st.download_button("Download log (CSV)", file_manager.list_as_csv(shown),
                       file_name="cinematch_activity_log.csv", mime="text/csv",
                       on_click="ignore")
