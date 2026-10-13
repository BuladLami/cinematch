"""
ui/login.py -- The sign-in screen, shown until someone signs in.

Two tabs:
    Sign in          any account, user or admin
    Create account   always makes a "user"; only an admin can make another admin

A successful sign-in stores {"username", "role"} in st.session_state["user"],
which app.py reads to decide which pages to show.
"""

from __future__ import annotations

import streamlit as st

from .. import accounts, config
from . import components


def _sign_in(user: dict) -> None:
    st.session_state["user"] = user
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

        sign_in_tab, sign_up_tab = st.tabs(["Sign in", "Create account"])

        with sign_in_tab:
            with st.form("sign_in_form"):
                username = st.text_input("Username")
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
            with st.form("sign_up_form"):
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
                        _sign_in(accounts.authenticate(new_name, new_password))
                    except accounts.AccountError as error:
                        st.error(str(error))
