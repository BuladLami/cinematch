"""
ui/login.py -- The sign-in screen, shown until someone signs in.

Two tabs:
    Sign in          any account, user or admin
    Create account   always makes a "user"; only an admin can make another admin

A successful sign-in stores {"username", "role"} in st.session_state["user"],
which app.py reads to decide which pages to show.

After an account is created, an "Account Created!" pop-up asks what to do next:
go straight into CineMatch, or go back to the sign-in tab (with the new
username already filled in).
"""

from __future__ import annotations

import streamlit as st

from .. import accounts, config
from . import components


TAB_SIGN_IN, TAB_SIGN_UP = "Sign in", "Create account"


def _sign_in(user: dict) -> None:
    st.session_state["user"] = user
    st.rerun()


def _proceed(username: str) -> None:
    """The new user chose to go straight in (they have just set the password)."""
    st.session_state.pop("new_account", None)
    accounts.log(username, "signed in", "right after creating the account")
    st.session_state["user"] = accounts.get_user(username)


def _back_to_sign_in(username: str) -> None:
    """The new user chose to sign in themselves: show the sign-in tab, name filled in."""
    st.session_state.pop("new_account", None)
    st.session_state["login_tab"] = TAB_SIGN_IN
    st.session_state["sign_in_username"] = username


# dismissible=False: the pop-up closes only through one of its two buttons.
@st.dialog("Account Created!", dismissible=False)
def _account_created(username: str) -> None:
    st.markdown(f"Welcome to CineMatch, **{username}**! Your account is ready.")
    st.caption("Go straight in, or go back and sign in with your new username and password.")
    proceed_column, back_column = st.columns(2)
    with proceed_column:
        # on_click runs before the rerun, so the choice lands before the page redraws.
        if st.button("Proceed to CineMatch", type="primary", width="stretch",
                     on_click=_proceed, args=(username,)):
            st.rerun()
    with back_column:
        if st.button("Back to sign in", width="stretch",
                     on_click=_back_to_sign_in, args=(username,)):
            st.rerun()


def render() -> None:
    _, middle, _ = st.columns([1, 1.4, 1])
    with middle:
        st.markdown(
            '<div class="cm-brand" style="justify-content:center;margin-top:2rem">'
            '<div class="cm-brand-mark">🎬</div>'
            '<div class="cm-brand-name">Cine<span>Match</span></div></div>',
            unsafe_allow_html=True,
        )
        components.page_header(
            "Welcome", "Just browse: CineMatch learns what you like from what you search, "
            "open and save, and recommends what to watch next.")

        sign_in_tab, sign_up_tab = st.tabs([TAB_SIGN_IN, TAB_SIGN_UP], key="login_tab")

        with sign_in_tab:
            with st.form("sign_in_form"):
                username = st.text_input("Username", key="sign_in_username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign in", type="primary", width="stretch")
            if submitted:
                try:
                    _sign_in(accounts.authenticate(username, password))
                except accounts.AccountError as error:
                    st.error(str(error))

            if accounts.using_default_password():
                name, password = config.DEFAULT_ADMIN
                st.info(f"First run: sign in as **{name}** / **{password}**, then change "
                        "the password from the sidebar.")

        with sign_up_tab:
            # clear_on_submit empties the form, so the password is not left in it.
            with st.form("sign_up_form", clear_on_submit=True):
                new_name = st.text_input("Choose a username",
                                         help="3-20 characters: letters, numbers or _")
                new_password = st.text_input(
                    "Choose a password", type="password",
                    help=f"At least {config.MIN_PASSWORD_LENGTH} characters")
                repeat = st.text_input("Repeat the password", type="password")
                created = st.form_submit_button("Create account", type="primary",
                                                width="stretch")
            if created:
                if new_password != repeat:
                    st.error("The two passwords are different.")
                else:
                    try:
                        accounts.create_user(new_name, new_password, role="user")
                        st.session_state["new_account"] = accounts.clean_username(new_name)
                    except accounts.AccountError as error:
                        st.error(str(error))

    # Shown on every run until one of its buttons is pressed.
    if st.session_state.get("new_account"):
        _account_created(st.session_state["new_account"])
