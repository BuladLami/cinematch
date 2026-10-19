"""
accounts.py -- Sign-in, the two roles (admin and user), and the activity log.

THE DATABASE
------------
Accounts live in one database table:

    users
      id             INTEGER  primary key, numbered automatically
      username       TEXT     unique, 3-20 lowercase characters
      password_hash  TEXT     PBKDF2-SHA256 of password + salt
      salt           TEXT     random, different for every account
      role           TEXT     'admin' or 'user' -- nothing else is accepted
      created_on     TEXT
      last_login     TEXT

The table is kept in one of two places:

    online   a PostgreSQL database, when DATABASE_URL is set (in the
             environment or .streamlit/secrets.toml -- see config.database_url).
             Every computer, phone and deployment that has the URL signs in
             against the same accounts.
    local    otherwise the SQLite file data/cinematch.db.  SQLite is built into
             Python, so there is nothing to install -- but the accounts then
             exist only on the computer the app runs on.

The rules are written into the table itself, in both databases, so they hold
no matter which code touches it:

    CHECK (role IN ('admin', 'user'))     no third role can be stored
    UNIQUE (username)                     two accounts cannot share a name
    TRIGGERS                              the last admin can be neither deleted
                                          nor turned into a user

Every query uses placeholders, so whatever a person types into the sign-in
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

Accounts made before DATABASE_URL was set can be copied into the online
database, passwords and all:   python -m cinematch.accounts

Every sign-in, account change and admin action is appended to
data/activity_log.csv.  (Each user's browsing is logged separately, in
data/users/<username>/activity.csv -- see activity.py.)
"""

from __future__ import annotations

import atexit
import hashlib
import hmac
import re
import secrets
import sqlite3
import sys
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

from . import config, file_manager
from .file_manager import FileManagerError

try:                       # only needed for the online database (DATABASE_URL)
    import psycopg
except ImportError:
    psycopg = None

# 3-20 letters, digits or underscores.  This also makes every username a safe
# folder name, since it doubles as the name of data/users/<username>/.
_USERNAME_PATTERN = re.compile(r"^[a-z0-9_]{3,20}$")

LAST_ADMIN = "This is the only admin. Make someone else an admin first."

_COLUMNS = """
    username      TEXT NOT NULL UNIQUE
                  CHECK (length(username) BETWEEN 3 AND 20 AND username = lower(username)),
    password_hash TEXT NOT NULL,
    salt          TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('admin', 'user')),
    created_on    TEXT NOT NULL,
    last_login    TEXT NOT NULL DEFAULT ''
"""

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,{_COLUMNS});

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

# The same table and rules in PostgreSQL.  Its triggers call a function; the
# advisory lock makes two admins being removed at the same moment take turns,
# so the second one sees that only one admin is left.  ERRCODE 23000 marks the
# refusal as an integrity error, like a broken CHECK or UNIQUE rule.
POSTGRES_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,{_COLUMNS});

CREATE OR REPLACE FUNCTION keep_an_admin() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF OLD.role = 'admin' AND (TG_OP = 'DELETE' OR NEW.role <> 'admin') THEN
        PERFORM pg_advisory_xact_lock(hashtext('cinematch.keep_an_admin'));
        IF (SELECT COUNT(*) FROM users WHERE role = 'admin') <= 1 THEN
            RAISE EXCEPTION '{LAST_ADMIN}' USING ERRCODE = '23000';
        END IF;
    END IF;
    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;
    RETURN NEW;
END $$;

DROP TRIGGER IF EXISTS keep_an_admin ON users;
CREATE TRIGGER keep_an_admin
BEFORE DELETE OR UPDATE OF role ON users
FOR EACH ROW EXECUTE FUNCTION keep_an_admin();
"""

# The columns the pages see.  The password hash and salt never leave this module.
PUBLIC_COLUMNS = ["username", "role", "created_on", "last_login"]

# A CHECK, UNIQUE or trigger rule refused the change.
_REFUSED = (sqlite3.IntegrityError,) + ((psycopg.IntegrityError,) if psycopg else ())


class AccountError(FileManagerError):
    """Raised when an account action is refused (bad password, duplicate name...)."""


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# CONNECTING
# ---------------------------------------------------------------------------

class _Connection:
    """An open connection to either database, used the same way for both.

    The SQL in this module is written with "?" placeholders, which is what
    SQLite expects; PostgreSQL's driver expects "%s", so they are swapped.
    Rows are read by column name (row["role"]) in both.
    """

    def __init__(self, raw, postgres: bool) -> None:
        self.raw = raw
        self.postgres = postgres

    def execute(self, sql: str, params: tuple = ()):
        if self.postgres:
            sql = sql.replace("?", "%s")
        return self.raw.execute(sql, params)


@contextmanager
def _db(seed: bool = True):
    """Open the database, make sure the table exists, and close it again.

    Everything inside the "with" block is one transaction: it is committed
    when the block finishes and rolled back if it raises, so a half-done
    change is never saved.  On the very first run (an empty table) the first
    admin is created, unless seed=False.
    """
    url = config.database_url()
    try:
        with (_postgres(url, seed) if url else _sqlite(seed)) as connection:
            yield connection
    except _REFUSED as error:
        raise AccountError(_explain(error)) from error
    except Exception as error:
        if psycopg and isinstance(error, psycopg.OperationalError):
            raise AccountError("Could not reach the accounts database. Check "
                               f"DATABASE_URL and the internet connection ({error}).") from error
        raise


@contextmanager
def _sqlite(seed: bool):
    """The local data/cinematch.db.

    A connection is opened per call because Streamlit serves each browser tab
    on its own thread and a SQLite connection must stay on the thread that
    opened it.  Opening a local file is quick.
    """
    config.DATABASE.parent.mkdir(parents=True, exist_ok=True)
    raw = sqlite3.connect(config.DATABASE, timeout=10)
    raw.row_factory = sqlite3.Row          # rows behave like dicts
    try:
        with raw:                          # commits, or rolls back on an error
            raw.executescript(SCHEMA)
            connection = _Connection(raw, postgres=False)
            if seed:
                _seed_first_admin(connection)
            yield connection
    finally:
        raw.close()


# One pool of open connections per online database.  Connecting to a server
# over the internet takes a moment, so connections are kept open and shared
# between browser tabs instead of being opened for every query.
_pools: dict[str, dict] = {}
_pools_lock = threading.Lock()


def _pool(url: str) -> dict:
    """The connection pool for this URL, created (with the table) on first use."""
    with _pools_lock:
        if url not in _pools:
            if psycopg is None:
                raise AccountError("DATABASE_URL is set, but the PostgreSQL driver is not "
                                   "installed. Run:  pip install -r requirements.txt")
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool
            pool = ConnectionPool(
                url, min_size=1, max_size=4, open=False, timeout=20,
                check=ConnectionPool.check_connection,   # replace connections the server dropped
                # No prepared statements: the "transaction" poolers of Supabase
                # and Neon cannot keep them between queries.
                kwargs={"row_factory": dict_row, "prepare_threshold": None})
            try:
                pool.open(wait=True, timeout=20)
                with pool.connection() as raw:
                    raw.execute(POSTGRES_SCHEMA)
            except BaseException:
                pool.close()
                raise
            atexit.register(pool.close)          # hang up cleanly when Python exits
            _pools[url] = {"pool": pool, "seeded": False}
        return _pools[url]


@contextmanager
def _postgres(url: str, seed: bool):
    """The online database named by DATABASE_URL."""
    entry = _pool(url)
    with entry["pool"].connection() as raw:   # commits, or rolls back on an error
        connection = _Connection(raw, postgres=True)
        # Once the table has an admin it always keeps one (the triggers), so
        # the first-admin check is only needed once per run.
        seeding = seed and not entry["seeded"]
        if seeding:
            _seed_first_admin(connection)
        yield connection
    if seeding:                               # only once it was committed
        entry["seeded"] = True


def _seed_first_admin(connection: _Connection) -> None:
    if connection.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"] == 0:
        _insert(connection, config.DEFAULT_ADMIN[0], config.default_admin_password(),
                role="admin")
        log(config.DEFAULT_ADMIN[0], "account created", "default admin (first run)")


def _explain(error: Exception) -> str:
    """Turn a database refusal into a sentence for the page."""
    text = str(error)
    if LAST_ADMIN in text:
        return LAST_ADMIN
    if "UNIQUE" in text or "duplicate key" in text:     # SQLite / PostgreSQL wording
        return "That username is already taken."
    if "role" in text:
        return f"A role must be one of: {', '.join(config.ROLES)}."
    return f"The database refused the change ({text})."


def _insert(connection: _Connection, username: str, password: str, role: str) -> None:
    salt = secrets.token_hex(16)
    connection.execute(
        "INSERT INTO users (username, password_hash, salt, role, created_on) "
        "VALUES (?, ?, ?, ?, ?)",
        (username, hash_password(password, salt), salt, role, _now()),
    )


def _row(connection: _Connection, username: str):
    """The account's full row (read by column name), or None."""
    return connection.execute("SELECT * FROM users WHERE username = ?",
                              (str(username or "").strip().lower(),)).fetchone()


# ---------------------------------------------------------------------------
# PASSWORDS AND NAMES
# ---------------------------------------------------------------------------

def hash_password(password: str, salt: str) -> str:
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                 bytes.fromhex(salt), config.PASSWORD_ITERATIONS)
    return digest.hex()


def _matches(row, password: str) -> bool:
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
    """Which database holds the accounts, and how many there are (shown to the admin)."""
    with _db() as connection:
        counts = {row["role"]: row["n"] for row in connection.execute(
            "SELECT role, COUNT(*) AS n FROM users GROUP BY role").fetchall()}
        if connection.postgres:
            size = connection.execute(
                "SELECT pg_total_relation_size('users') AS size").fetchone()["size"]
            kind, where = "PostgreSQL", urlsplit(config.database_url()).hostname or "online"
        else:
            size = config.DATABASE.stat().st_size
            kind, where = "SQLite", config.DATABASE.name
    return {"kind": kind, "where": where, "online": connection.postgres,
            "size_kb": size / 1024,
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


# ---------------------------------------------------------------------------
# MOVING TO THE ONLINE DATABASE
# ---------------------------------------------------------------------------

def copy_accounts_from(sqlite_file: Path | str) -> str:
    """Copy every account in a local SQLite file into the database in use.

    The password hashes and salts are copied as they are, so every account
    keeps its password.  A username that is already in the database is left
    alone.  Run it before the online database is first used and the local
    admin comes along too; after that, the online admin is kept.
    """
    path = Path(sqlite_file)
    if not path.exists():
        raise AccountError(f"There is no database file at {path}.")
    local = sqlite3.connect(path)
    local.row_factory = sqlite3.Row
    try:
        rows = local.execute("SELECT username, password_hash, salt, role, created_on, "
                             "last_login FROM users ORDER BY id").fetchall()
    finally:
        local.close()

    copied, skipped = [], []
    with _db(seed=False) as connection:
        for row in rows:
            added = connection.execute(
                "INSERT INTO users (username, password_hash, salt, role, created_on, last_login) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (username) DO NOTHING",
                tuple(row)).rowcount
            (copied if added else skipped).append(row["username"])
        if not copied and connection.execute(
                "SELECT COUNT(*) AS n FROM users").fetchone()["n"] == 0:
            _seed_first_admin(connection)          # nothing to copy: start as usual
    for name in copied:
        config.user_dir(name).mkdir(parents=True, exist_ok=True)
    log(config.DEFAULT_ADMIN[0], "accounts copied", f"{len(copied)} from {path.name}")
    message = f"Copied {len(copied)} account(s): {', '.join(copied) or 'none'}."
    if skipped:
        message += f"  Already there, left as they were: {', '.join(skipped)}."
    return message


if __name__ == "__main__":
    # python -m cinematch.accounts [file.db]
    # Copies data/cinematch.db (or the file given) into the DATABASE_URL database.
    if not config.database_url():
        sys.exit("Set DATABASE_URL first (in .streamlit/secrets.toml or the environment), "
                 "so there is an online database to copy the accounts into.")
    source = sys.argv[1] if len(sys.argv) > 1 else config.DATABASE
    try:
        print(copy_accounts_from(source))
    except AccountError as error:
        sys.exit(str(error))
