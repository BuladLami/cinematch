"""
accounts.py -- Sign-in, the two roles (admin and user), and the activity log.

THE DATABASE
------------
Accounts live in a SQLite database, data/cinematch.db, in one table:

    users
      id             INTEGER  primary key, numbered automatically
      username       TEXT     unique, 3-20 lowercase characters
      password_hash  TEXT     PBKDF2-SHA256 of password + salt
      salt           TEXT     random, different for every account
      role           TEXT     'admin' or 'user' -- nothing else is accepted
      created_on     TEXT
      last_login     TEXT

SQLite is a single file and is built into Python (the sqlite3 module), so
there is no database server to install or start.  The rules are written into
the table itself, so they hold no matter which code touches the database:

    CHECK (role IN ('admin', 'user'))     no third role can be stored
    UNIQUE (username)                     two accounts cannot share a name
    two TRIGGERS                          the last admin can be neither deleted
                                          nor turned into a user

Every query uses "?" placeholders, so whatever a person types into the sign-in
form is treated as data and never as SQL (no SQL injection).

Passwords are NEVER stored as typed.  Each account gets a random "salt", and
the password + salt are run through PBKDF2-SHA256 100,000 times; only the
result is saved.  Signing in repeats the same steps and compares the results.

THE TWO ROLES
-------------
    user    browses, searches and keeps a My List; CineMatch learns their watch
            pattern from that behaviour and recommends from it
    admin   everything a user can do (their own behaviour is learned from too),
            plus: create / delete accounts, reset passwords, change roles, look
            at any user's learned pattern, and read the activity log

The first time the app runs the table is empty, so one admin is created from
config.DEFAULT_ADMIN, with the password from config.default_admin_password()
(a deployment sets CINEMATCH_ADMIN_PASSWORD in its secrets; otherwise it is
admin123).  Every other account is made from the sign-up form
(always a "user") or by an admin.

Every sign-in, account change and admin action is appended to
data/activity_log.csv.  (Each user's browsing is logged separately, in
data/users/<username>/activity.csv -- see activity.py.)
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

from . import config, file_manager
from .file_manager import FileManagerError

# 3-20 letters, digits or underscores.  This also makes every username a safe
# folder name, since it doubles as the name of data/users/<username>/.
_USERNAME_PATTERN = re.compile(r"^[a-z0-9_]{3,20}$")

LAST_ADMIN = "This is the only admin. Make someone else an admin first."

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE
                  CHECK (length(username) BETWEEN 3 AND 20 AND username = lower(username)),
    password_hash TEXT NOT NULL,
    salt          TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('admin', 'user')),
    created_on    TEXT NOT NULL,
    last_login    TEXT NOT NULL DEFAULT ''
);

CREATE TRIGGER IF NOT EXISTS keep_an_admin_on_delete
BEFORE DELETE ON users
WHEN OLD.role = 'admin' AND (SELECT COUNT(*) FROM users WHERE role = 'admin') <= 1
BEGIN
    SELECT RAISE(ABORT, '{LAST_ADMIN}');
END;

CREATE TRIGGER IF NOT EXISTS keep_an_admin_on_update
BEFORE UPDATE OF role ON users
WHEN OLD.role = 'admin' AND NEW.role <> 'admin'
     AND (SELECT COUNT(*) FROM users WHERE role = 'admin') <= 1
BEGIN
    SELECT RAISE(ABORT, '{LAST_ADMIN}');
END;
"""

# The columns the pages see.  The password hash and salt never leave this module.
PUBLIC_COLUMNS = ["username", "role", "created_on", "last_login"]


class AccountError(FileManagerError):
    """Raised when an account action is refused (bad password, duplicate name...)."""


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# CONNECTING
# ---------------------------------------------------------------------------

@contextmanager
def _db():
    """Open the database, make sure the table exists, and close it again.

    "with connection:" commits when the block finishes and rolls everything
    back if it raises, so a half-done change is never saved.  A connection is
    opened per call because Streamlit serves each browser tab on its own thread
    and a SQLite connection must stay on the thread that opened it.
    """
    config.DATABASE.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(config.DATABASE, timeout=10)
    connection.row_factory = sqlite3.Row          # rows behave like dicts
    try:
        with connection:
            connection.executescript(SCHEMA)
            if connection.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
                _insert(connection, config.DEFAULT_ADMIN[0], config.default_admin_password(),
                        role="admin")
                log(config.DEFAULT_ADMIN[0], "account created", "default admin (first run)")
            yield connection
    except sqlite3.IntegrityError as error:
        # A CHECK, UNIQUE or trigger rule refused the change.
        raise AccountError(_explain(error)) from error
    finally:
        connection.close()


def _explain(error: sqlite3.IntegrityError) -> str:
    """Turn a database refusal into a sentence for the page."""
    text = str(error)
    if LAST_ADMIN in text:
        return LAST_ADMIN
    if "UNIQUE" in text:
        return "That username is already taken."
    if "role" in text:
        return f"A role must be one of: {', '.join(config.ROLES)}."
    return f"The database refused the change ({text})."


def _insert(connection: sqlite3.Connection, username: str, password: str, role: str) -> None:
    salt = secrets.token_hex(16)
    connection.execute(
        "INSERT INTO users (username, password_hash, salt, role, created_on) "
        "VALUES (?, ?, ?, ?, ?)",
        (username, hash_password(password, salt), salt, role, _now()),
    )


def _row(connection: sqlite3.Connection, username: str) -> sqlite3.Row | None:
    return connection.execute("SELECT * FROM users WHERE username = ?",
                              (str(username or "").strip().lower(),)).fetchone()


# ---------------------------------------------------------------------------
# PASSWORDS AND NAMES
# ---------------------------------------------------------------------------

def hash_password(password: str, salt: str) -> str:
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                 bytes.fromhex(salt), config.PASSWORD_ITERATIONS)
    return digest.hex()


def _matches(row: sqlite3.Row, password: str) -> bool:
    # compare_digest takes the same time whether the first or last character
    # differs, so the comparison cannot be timed to guess the hash.
    return hmac.compare_digest(hash_password(password or "", row["salt"]),
                               row["password_hash"])


def _check_password_rules(password: str) -> None:
    if len(password or "") < config.MIN_PASSWORD_LENGTH:
        raise AccountError(
            f"A password needs at least {config.MIN_PASSWORD_LENGTH} characters.")


def clean_username(username: str) -> str:
    name = str(username or "").strip().lower()
    if not _USERNAME_PATTERN.match(name):
        raise AccountError("A username is 3-20 characters: letters, numbers or _ only.")
    return name


# ---------------------------------------------------------------------------
# READ
# ---------------------------------------------------------------------------

def read_users() -> pd.DataFrame:
    """Every account (without the password columns), oldest first."""
    with _db() as connection:
        rows = connection.execute(
            f"SELECT {', '.join(PUBLIC_COLUMNS)} FROM users ORDER BY id").fetchall()
    return pd.DataFrame([dict(r) for r in rows], columns=PUBLIC_COLUMNS)


def get_user(username: str) -> dict | None:
    """One account (without its password fields), or None."""
    with _db() as connection:
        row = _row(connection, username)
    return {column: row[column] for column in PUBLIC_COLUMNS} if row else None


def using_default_password() -> bool:
    """True while the first-run admin still has its well-known password."""
    username, password = config.DEFAULT_ADMIN
    with _db() as connection:
        row = _row(connection, username)
    return bool(row) and _matches(row, password)


def database_info() -> dict:
    """Where the database is and how big it is (shown to the admin)."""
    with _db() as connection:
        counts = dict(connection.execute(
            "SELECT role, COUNT(*) FROM users GROUP BY role").fetchall())
    return {"path": config.DATABASE, "size_kb": config.DATABASE.stat().st_size / 1024,
            "admins": counts.get("admin", 0), "users": counts.get("user", 0)}


# ---------------------------------------------------------------------------
# SIGN IN / SIGN UP
# ---------------------------------------------------------------------------

def authenticate(username: str, password: str) -> dict:
    """Check a username + password.  Returns the account, or raises AccountError.

    The same message is given for an unknown name and a wrong password, so the
    form does not reveal which usernames exist.
    """
    name = str(username or "").strip().lower()
    with _db() as connection:
        row = _row(connection, name)
        if row is None or not _matches(row, password):
            failed = True
        else:
            failed = False
            connection.execute("UPDATE users SET last_login = ? WHERE id = ?",
                               (_now(), row["id"]))
    if failed:
        if row is not None:
            log(name, "sign-in failed", "")
        raise AccountError("Wrong username or password.")
    log(name, "signed in", "")
    return get_user(name)


def create_user(username: str, password: str, role: str = "user",
                created_by: str | None = None) -> str:
    """CREATE an account.  The sign-up form always passes role="user"."""
    name = clean_username(username)
    _check_password_rules(password)
    if role not in config.ROLES:
        raise AccountError(f"A role must be one of: {', '.join(config.ROLES)}.")
    with _db() as connection:
        if _row(connection, name) is not None:
            raise AccountError(f'The username "{name}" is already taken.')
        _insert(connection, name, password, role)
    config.user_dir(name).mkdir(parents=True, exist_ok=True)
    log(created_by or name, "account created", f"{name} ({role})")
    return f'Created the account "{name}" ({role}).'


# ---------------------------------------------------------------------------
# UPDATE / DELETE
# ---------------------------------------------------------------------------

def change_password(username: str, new_password: str, changed_by: str | None = None,
                    old_password: str | None = None) -> str:
    """UPDATE a password.  A user changing their own must give the old one."""
    _check_password_rules(new_password)
    with _db() as connection:
        row = _row(connection, username)
        if row is None:
            raise AccountError(f'No account called "{username}".')
        if old_password is not None and not _matches(row, old_password):
            raise AccountError("Your current password is not correct.")
        salt = secrets.token_hex(16)
        connection.execute("UPDATE users SET password_hash = ?, salt = ? WHERE id = ?",
                           (hash_password(new_password, salt), salt, row["id"]))
    log(changed_by or username, "password changed", username)
    return f'Password changed for "{username}".'


def set_role(username: str, role: str, changed_by: str) -> str:
    """UPDATE an account's role.  The database refuses to demote the last admin."""
    with _db() as connection:
        row = _row(connection, username)
        if row is None:
            raise AccountError(f'No account called "{username}".')
        connection.execute("UPDATE users SET role = ? WHERE id = ?", (role, row["id"]))
    log(changed_by, "role changed", f"{username} -> {role}")
    return f'"{username}" is now {"an" if role == "admin" else "a"} {role}.'


def delete_user(username: str, deleted_by: str) -> str:
    """DELETE an account and its folder (My List, activity).

    The database refuses to delete the last admin.  The folder is removed only
    after the row is gone, so a refused delete leaves the files untouched.
    """
    if username == deleted_by:
        raise AccountError("You cannot delete the account you are signed in with.")
    with _db() as connection:
        row = _row(connection, username)
        if row is None:
            raise AccountError(f'No account called "{username}".')
        connection.execute("DELETE FROM users WHERE id = ?", (row["id"],))
    file_manager.delete_user_folder(username)
    log(deleted_by, "account deleted", username)
    return f'Deleted "{username}" and all of their files.'


# ---------------------------------------------------------------------------
# ACTIVITY LOG
# ---------------------------------------------------------------------------

def log(username: str, action: str, details: str = "") -> None:
    """APPEND one line to data/activity_log.csv."""
    file_manager.append_csv_row(
        {"timestamp": _now(), "username": username, "action": action, "details": details},
        config.ACTIVITY_LOG_CSV, config.LOG_COLUMNS)


def read_log() -> pd.DataFrame:
    """READ the activity log, newest first."""
    if not config.ACTIVITY_LOG_CSV.exists():
        return pd.DataFrame(columns=config.LOG_COLUMNS)
    df = file_manager.read_csv_file(config.ACTIVITY_LOG_CSV, dtype=str).fillna("")
    return df.iloc[::-1].reset_index(drop=True)
