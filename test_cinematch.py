"""
test_cinematch.py -- Plain-Python tests for every major feature.

Run it with:    python test_cinematch.py

No testing library is needed, so this works on any machine that can run the app.
Each check prints PASS or FAIL, and the script exits with code 1 if anything
failed.

Important: the My List tests work on the real data/my_list.csv, so the file is
saved first and restored at the end.  The account and activity tests
(sections 8-9) run in a scratch folder that is deleted afterwards.  The source
dataset folder is never written to at any point.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import traceback

import pandas as pd

from cinematch import config, file_manager, preprocessing, recommender

PASSED = 0
FAILED = 0


def check(description: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {description}")
    else:
        FAILED += 1
        print(f"  FAIL  {description}" + (f"  --> {detail}" if detail else ""))


def section(title: str) -> None:
    print()
    print(f"--- {title} " + "-" * max(0, 60 - len(title)))


# ---------------------------------------------------------------------------

def test_preprocessing() -> None:
    section("1. Preprocessing")

    check("clean_text lowercases and strips punctuation",
          preprocessing.clean_text("Marvel's Avengers: Age of Ultron!")
          == "marvels avengers age of ultron")
    check("clean_text keeps accented letters",
          preprocessing.clean_text("Léa Seydoux") == "léa seydoux")
    check("squash_name makes one token from a name",
          preprocessing.squash_name("James Cameron") == "jamescameron")
    check("split_list splits a stored list",
          preprocessing.split_list("Action|Science Fiction") == ["Action", "Science Fiction"])
    check("split_list survives an empty value",
          preprocessing.split_list("") == [] and preprocessing.split_list(None) == [])
    check("list_tokens gives one token per person",
          preprocessing.list_tokens("Tom Hanks|Tim Allen") == "tomhanks timallen")

    messy = pd.DataFrame({
        "id": [1, 2, 3, 3],
        "title": ["Alpha", "", "Gamma", "Gamma"],
        "genres": ["Action", "Drama", None, None],
        "overview": ["good", "bad", None, None],
        "keywords": [None, None, None, None],
        "cast": ["Some Actor|Other Actor", "C D", None, None],
        "director": ["Dir One", None, None, None],
        "collection": [None, None, None, None],
        "tagline": [None, None, None, None],
        "release_date": ["2001-01-01", "bad-date", None, None],
        "runtime": [100, None, 90, 90],
        "vote_average": [7.5, None, 0, 0],
        "vote_count": [10, 20, None, None],
        "popularity": [1.0, None, 2.0, 2.0],
    })
    clean, report = preprocessing.prepare(messy)

    check("no missing values remain after prepare()",
          int(clean.isna().sum().sum()) == 0, str(clean.isna().sum().to_dict()))
    check("blank title row was removed", report["rows_removed"]["blank_title"] == 1)
    check("duplicate rows were removed",
          report["rows_removed"]["exact_duplicates"] + report["rows_removed"]["duplicate_ids"] == 1)
    check("the cast is shown comma-separated",
          clean.loc[0, "cast_list"] == "Some Actor, Other Actor", clean.loc[0, "cast_list"])
    check("soup column was created and is not empty",
          "soup" in clean.columns and clean["soup"].iloc[0] != "")
    check("genre weighting repeats the genre in the soup",
          clean["soup"].iloc[0].count("action") >= config.FEATURE_WEIGHTS["genres"])
    check("each actor is one token in the soup", "someactor" in clean["soup"].iloc[0])

    summary = preprocessing.dataset_summary(clean)
    check("dataset_summary returns the row count", summary["total_movies"] == len(clean))
    check("dataset_summary ignores 0 ratings in the average",
          summary["average_rating"] == 7.5, str(summary["average_rating"]))


# ---------------------------------------------------------------------------

def test_catalogue() -> pd.DataFrame:
    section("2. Building and reading the catalogue")

    genres = "[{'id': 16, 'name': 'Animation'}, {'id': 35, 'name': 'Comedy'}]"
    check("names are pulled out of the source's list text",
          file_manager._names(genres) == ["Animation", "Comedy"])
    check("a name with an apostrophe (double-quoted) is read correctly",
          file_manager._names("""[{'id': 1, 'name': "Peter O'Toole"}]""") == ["Peter O'Toole"])
    crew = ("[{'credit_id': 'x', 'department': 'Writing', 'gender': 2, 'id': 1, "
            "'job': 'Screenplay', 'name': 'A Writer'}, {'credit_id': 'y', 'department': "
            "'Directing', 'gender': 2, 'id': 2, 'job': 'Director', 'name': 'The Director'}]")
    check("only the director is taken from the crew",
          file_manager._names(crew, pattern=file_manager._DIRECTOR_PATTERN) == ["The Director"])

    path = file_manager.build_catalogue()
    check("the catalogue file was written", path.exists(), str(path))

    raw = file_manager.read_catalogue()
    check("the catalogue has thousands of movies", len(raw) > 5000, f"{len(raw)} rows")
    check("only the catalogue columns were kept",
          list(raw.columns) == config.CATALOGUE_COLUMNS, str(list(raw.columns)))
    check("every movie has enough votes", raw["vote_count"].min() >= config.MIN_VOTES)
    check("movie ids are unique", raw["id"].is_unique)

    movies, report = file_manager.load_prepared_movies()
    check("the catalogue was read and prepared", len(movies) == len(raw))
    check("well known titles are present",
          {"The Dark Knight", "Toy Story", "Spirited Away"} <= set(movies["title"]))

    toy = movies[movies["title"] == "Toy Story"].iloc[0]
    check("the cast comes through in billing order",
          toy["cast_list"].startswith("Tom Hanks, Tim Allen"), toy["cast_list"])
    check("the director comes through", toy["director_list"] == "John Lasseter",
          toy["director_list"])
    check("the collection comes through", toy["collection"] == "Toy Story Collection",
          toy["collection"])
    check("keyword phrases stay whole", "boy next door" in toy["keyword_list"],
          toy["keyword_list"])

    try:
        file_manager.read_csv_file(config.DATA_DIR / "does_not_exist.csv")
        check("a missing file raises FileManagerError", False)
    except file_manager.FileManagerError:
        check("a missing file raises FileManagerError", True)

    return movies


# ---------------------------------------------------------------------------

def test_source_is_protected() -> None:
    section("3. The source dataset is protected")

    target = config.SOURCE_DIR / "movies_metadata.csv"
    before = target.stat()
    try:
        file_manager._atomic_write(pd.DataFrame({"title": ["hack"]}), target)
        check("writing into the source folder is refused", False, "the write went through!")
    except file_manager.FileManagerError:
        check("writing into the source folder is refused", True)

    after = target.stat()
    check("movies_metadata.csv is unchanged",
          before.st_size == after.st_size and before.st_mtime == after.st_mtime)


# ---------------------------------------------------------------------------

def test_recommendations(movies: pd.DataFrame) -> None:
    section("4. Recommendations (the machine-learning part)")

    engine = recommender.build_recommender(movies, use_cache=False)
    info = engine.model_info()
    check("the TF-IDF model was fitted", info["fitted"])
    check("one row per movie in the matrix", info["movies"] == len(movies))
    check("a sensible vocabulary was learned", info["vocabulary"] > 5000,
          str(info["vocabulary"]))

    results = engine.recommend_by_title("The Dark Knight", top_n=5)
    check("5 recommendations were returned", len(results) == 5, str(len(results)))
    check("scores are between 0 and 1", results["similarity"].between(0, 1).all())
    check("scores come back in descending order",
          list(results["similarity"]) == sorted(results["similarity"], reverse=True))
    check("the chosen movie is not recommended to itself",
          "The Dark Knight" not in results["title"].tolist())
    check("Batman films are recommended for The Dark Knight",
          all("Batman" in t or "Dark Knight" in t for t in results["title"]),
          str(results["title"].tolist()))

    toy = engine.recommend_by_title("Toy Story", top_n=10)
    check("top_n=10 returns 10 rows", len(toy) == 10)
    check("Toy Story 2 is the closest match to Toy Story",
          toy["title"].iloc[0] == "Toy Story 2", str(toy["title"].iloc[0]))

    hits = engine.search("dark knight")
    check("search finds movies by partial title", len(hits) >= 2, str(len(hits)))
    check("search is case-insensitive", len(engine.search("DARK KNIGHT")) == len(hits))
    check("search with no match returns an empty table",
          engine.search("zzzzzzzznotamovie").empty)

    dark = engine.find_index("The Dark Knight")
    check("index_of_id finds a movie by its id",
          engine.index_of_id(movies.loc[dark, "id"]) == dark)
    check("index_of_id returns None for an unknown id", engine.index_of_id(-1) is None)

    terms = engine.shared_terms(dark, engine.find_index("The Dark Knight Rises"))
    check("shared terms explain the match", len(terms) > 0, str(terms))
    check("shared terms are turned back into readable names",
          "The Dark Knight Collection" in engine.readable(terms), str(engine.readable(terms)))

    taste = [engine.find_index(t) for t in ["Toy Story", "Finding Nemo", "Up"]]
    profile = engine.recommend_for_profile(taste, [10, 8, 9], top_n=8)
    check("a taste profile returns recommendations", len(profile) == 8, str(len(profile)))
    check("the profile's own movies are not recommended back",
          not {"Toy Story", "Finding Nemo", "Up"} & set(profile["title"]))
    check("a Pixar profile recommends more Pixar-style films",
          "Toy Story 2" in profile["title"].tolist(), str(profile["title"].tolist()))

    try:
        engine.recommend_by_title("This Movie Does Not Exist")
        check("an unknown title raises RecommenderError", False)
    except recommender.RecommenderError:
        check("an unknown title raises RecommenderError", True)


# ---------------------------------------------------------------------------

def test_caching(movies: pd.DataFrame) -> None:
    section("5. TF-IDF caching")

    small = movies.head(400).copy()
    cache_path = config.CACHE_DIR / "test_cache.pkl"
    cache_path.unlink(missing_ok=True)

    first = recommender.MovieRecommender(small)
    check("no cache to load on the first run", first.try_load_cache(cache_path) is False)
    first.fit()
    first.save_cache(cache_path)
    check("the cache file was written", cache_path.exists())

    second = recommender.MovieRecommender(small)
    check("the cache is reused for identical data", second.try_load_cache(cache_path) is True)
    check("the reused matrix has the same shape", second.matrix.shape == first.matrix.shape)

    changed = small.copy()
    changed.loc[changed.index[0], "soup"] += " brandnewkeywordtoken"
    third = recommender.MovieRecommender(changed)
    check("the cache is rejected once the data changes",
          third.try_load_cache(cache_path) is False)
    check("the fingerprint changed with the data", third.fingerprint != first.fingerprint)

    cache_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------

def test_my_list(movies: pd.DataFrame) -> None:
    section("6. My List: create / read / update / delete")

    backup = config.MY_LIST_CSV.with_suffix(".testbackup")
    had_list = config.MY_LIST_CSV.exists()
    if had_list:
        shutil.copy2(config.MY_LIST_CSV, backup)
    config.MY_LIST_CSV.unlink(missing_ok=True)

    toy = movies[movies["title"] == "Toy Story"].iloc[0]
    nemo = movies[movies["title"] == "Finding Nemo"].iloc[0]

    try:
        check("an empty list is read as an empty table",
              file_manager.read_my_list().empty)

        # ---- CREATE ----
        file_manager.add_to_list(toy["id"], toy["title"])
        file_manager.add_to_list(nemo["id"], nemo["title"], status="Watched", my_rating=9)
        on_disk = pd.read_csv(config.MY_LIST_CSV)
        check("adding writes the file", config.MY_LIST_CSV.exists())
        check("both movies are in the file", len(on_disk) == 2)
        check("the default status is 'Want to watch'",
              file_manager.get_list_entry(toy["id"])["status"] == "Want to watch")
        check("the rating was saved", file_manager.get_list_entry(nemo["id"])["my_rating"] == 9)

        try:
            file_manager.add_to_list(toy["id"], toy["title"])
            check("adding the same movie twice is refused", False)
        except file_manager.FileManagerError:
            check("adding the same movie twice is refused", True)

        try:
            file_manager.add_to_list(12345678, "Nope", status="Loved it")
            check("an unknown status is refused", False)
        except file_manager.FileManagerError:
            check("an unknown status is refused", True)

        # ---- READ ----
        my_list = file_manager.read_my_list()
        check("read_my_list returns every column", list(my_list.columns) == config.MY_LIST_COLUMNS)
        check("get_list_entry returns None for a movie not on the list",
              file_manager.get_list_entry(-5) is None)

        # ---- UPDATE ----
        file_manager.update_list_entry(toy["id"], status="Watched", my_rating=10,
                                       notes="  Watched with my cousins  ")
        entry = file_manager.get_list_entry(toy["id"])
        check("the status was updated in the file", entry["status"] == "Watched")
        check("the rating was updated", entry["my_rating"] == 10)
        check("notes are saved and trimmed", entry["notes"] == "Watched with my cousins")

        try:
            file_manager.update_list_entry(toy["id"], my_rating=11)
            check("a rating above 10 is refused", False)
        except file_manager.FileManagerError:
            check("a rating above 10 is refused", True)
        check("a refused update leaves the file unchanged",
              file_manager.get_list_entry(toy["id"])["my_rating"] == 10)

        file_manager.update_list_entry(toy["id"], my_rating=None)
        check("a rating can be cleared again",
              pd.isna(file_manager.get_list_entry(toy["id"])["my_rating"]))

        edited = file_manager.read_my_list()
        edited.loc[edited["movie_id"] == nemo["id"], "status"] = "Watching"
        message = file_manager.replace_my_list(edited)
        check("the bulk editor saves only what changed", message.startswith("Saved 1 "), message)
        check("the bulk edit reached the file",
              file_manager.get_list_entry(nemo["id"])["status"] == "Watching")

        # ---- the list steers recommendations ----
        from cinematch.ui import mylist
        engine = recommender.build_recommender(movies)
        found, based_on = mylist.recommended_for_you(movies, engine, file_manager.read_my_list())
        check("'Recommended for you' uses the list", based_on == 2 and len(found) > 0)
        check("movies already on the list are not recommended",
              not set(found["id"]) & {toy["id"], nemo["id"]})

        file_manager.update_list_entry(nemo["id"], my_rating=2)
        _, based_on = mylist.recommended_for_you(movies, engine, file_manager.read_my_list())
        check("a movie rated below 6 no longer steers recommendations", based_on == 1)

        # ---- DELETE ----
        file_manager.remove_from_list(toy["id"])
        check("the removed movie is gone from the file",
              file_manager.get_list_entry(toy["id"]) is None)
        check("the other movie is still there", len(file_manager.read_my_list()) == 1)

        try:
            file_manager.remove_from_list(toy["id"])
            check("removing a movie that is not on the list is refused", False)
        except file_manager.FileManagerError:
            check("removing a movie that is not on the list is refused", True)

    finally:
        config.MY_LIST_CSV.unlink(missing_ok=True)
        if had_list:
            shutil.move(str(backup), str(config.MY_LIST_CSV))
        print("  (data/my_list.csv was restored to its original state)")


# ---------------------------------------------------------------------------

def test_messy_data() -> None:
    section("7. Handling missing and invalid data")

    messy = pd.DataFrame({
        "id": [1, 2],
        "title": ["No Content Movie", "Normal Movie"],
        "genres": ["", "Action"],
        "keywords": ["", "hero"],
        "cast": ["", "Someone Else"],
        "director": ["", "A Director"],
        "collection": ["", ""],
        "overview": ["", "A normal plot summary about a hero."],
        "tagline": ["", ""],
        "release_date": ["", "2010-05-05"],
        "runtime": [0, 100],
        "vote_average": [0, 7.0],
        "vote_count": [0, 50],
        "popularity": [0, 1.0],
    })
    clean, _ = preprocessing.prepare(messy)
    check("a movie with no text still produces an empty soup, not a crash",
          clean.loc[0, "soup"] == "")

    engine = recommender.MovieRecommender(clean).fit()
    check("a movie with no content returns no recommendations instead of failing",
          engine.recommend(0, top_n=5).empty)

    summary = preprocessing.dataset_summary(clean)
    check("the average rating ignores the unrated movie",
          summary["average_rating"] == 7.0, str(summary["average_rating"]))
    check("the genre count ignores the blank genre", summary["genre_count"] == 1)


# ---------------------------------------------------------------------------
# Sections 8-9 work in a scratch folder: the database, log and per-user paths
# in config are pointed there for the duration, so the real data/cinematch.db,
# data/activity_log.csv and data/users/ are never touched.

class scratch_accounts:
    """Context manager: point every account path at data/_test_accounts/.

    The accounts go in a scratch SQLite file even when a DATABASE_URL is set,
    so the tests never touch the real online accounts -- unless online= gives
    the URL of a test database (see test_online_accounts).
    """

    NAMES = ("DATABASE", "USERS_DIR", "ACTIVITY_LOG_CSV", "database_url")

    def __init__(self, online: str | None = None):
        self.online = online

    def __enter__(self):
        self.root = config.DATA_DIR / "_test_accounts"
        shutil.rmtree(self.root, ignore_errors=True)
        self.saved = {name: getattr(config, name) for name in self.NAMES}
        config.database_url = lambda: self.online
        config.DATABASE = self.root / "test.db"
        config.USERS_DIR = self.root / "users"
        config.ACTIVITY_LOG_CSV = self.root / "activity_log.csv"
        config.USERS_DIR.mkdir(parents=True)
        return self

    def __exit__(self, *exc):
        for name, value in self.saved.items():
            setattr(config, name, value)
        shutil.rmtree(self.root, ignore_errors=True)
        return False


def refused(description: str, action, error=file_manager.FileManagerError) -> None:
    """Check that action() raises the given error."""
    try:
        action()
        check(description, False, "no error was raised")
    except error:
        check(description, True)


def test_accounts() -> None:
    section("8. Accounts: admin and user roles")
    from cinematch import accounts

    with scratch_accounts():
        users = accounts.read_users()
        check("the first run creates one admin",
              list(users["role"]) == ["admin"] and users["username"].iloc[0] == "admin")
        with sqlite3.connect(config.DATABASE) as db:
            stored = db.execute("SELECT password_hash, salt FROM users").fetchone()
        db.close()
        check("the accounts are in a SQLite database file", config.DATABASE.exists())
        check("the password is stored hashed, never as typed",
              config.DEFAULT_ADMIN[1] not in stored and len(stored[0]) == 64)
        check("the default admin can sign in",
              accounts.authenticate(*config.DEFAULT_ADMIN)["role"] == "admin")
        check("the default password is flagged until it is changed",
              accounts.using_default_password())

        accounts.create_user("Alice", "secret1")
        check("sign-up makes a user (and lowercases the name)",
              accounts.get_user("alice")["role"] == "user")
        check("the new user gets their own folder", config.user_dir("alice").is_dir())

        refused("a wrong password is refused",
                lambda: accounts.authenticate("alice", "wrong-password"))
        refused("an unknown user is refused", lambda: accounts.authenticate("nobody", "secret1"))
        refused("a duplicate username is refused",
                lambda: accounts.create_user("alice", "secret1"))
        refused("a too-short password is refused", lambda: accounts.create_user("bob", "123"))
        refused("a path in the username is refused",
                lambda: accounts.create_user("../evil", "secret1"))

        accounts.change_password("alice", "newpass1", old_password="secret1")
        check("a changed password works", accounts.authenticate("alice", "newpass1") is not None)
        refused("changing a password needs the old one",
                lambda: accounts.change_password("alice", "another1", old_password="not-it"))
        refused("the last admin cannot be demoted",
                lambda: accounts.set_role("admin", "user", changed_by="admin"))
        accounts.set_role("alice", "admin", changed_by="admin")
        check("an admin can promote a user", accounts.get_user("alice")["role"] == "admin")

        file_manager.add_to_list(862, "Toy Story", username="alice")
        check("each user has their own My List",
              len(file_manager.read_my_list("alice")) == 1
              and file_manager.read_my_list("admin").empty)

        refused("an admin cannot delete their own account",
                lambda: accounts.delete_user("admin", deleted_by="admin"))
        accounts.delete_user("alice", deleted_by="admin")
        check("deleting a user removes the account and the folder",
              accounts.get_user("alice") is None and not config.user_dir("alice").exists())

        check("the activity log recorded the actions",
              {"signed in", "sign-in failed", "account created", "account deleted"}
              <= set(accounts.read_log()["action"]))

        # The rules live in the table itself, so they hold even for SQL typed
        # straight into the database, past every check in accounts.py.
        def raw_sql(statement: str, *values) -> bool:
            """True if the database accepted the statement."""
            db = sqlite3.connect(config.DATABASE)
            try:
                with db:
                    db.execute(statement, values)
                return True
            except sqlite3.IntegrityError:
                return False
            finally:
                db.close()

        insert = ("INSERT INTO users (username, password_hash, salt, role, created_on) "
                  "VALUES (?, 'x', 'x', ?, 'now')")
        check("the database only accepts the roles admin and user",
              not raw_sql(insert, "carol", "superuser") and raw_sql(insert, "carol", "user"))
        check("the database refuses a duplicate username", not raw_sql(insert, "carol", "user"))
        check("the database refuses an uppercase username", not raw_sql(insert, "Dave", "user"))
        check("the database will not delete the last admin",
              not raw_sql("DELETE FROM users WHERE role = 'admin'"))
        check("the database will not demote the last admin",
              not raw_sql("UPDATE users SET role = 'user' WHERE role = 'admin'"))
        refused("the trigger's refusal reaches the page as a clear message",
                lambda: accounts.set_role("admin", "user", changed_by="admin"),
                accounts.AccountError)
        check("an injection attempt in the sign-in form is just a wrong name",
              accounts.get_user("admin' OR '1'='1") is None)
        info = accounts.database_info()
        check("database_info counts admins and users",
              (info["admins"], info["users"]) == (1, 1), str(info))

    # A deployment sets CINEMATCH_ADMIN_PASSWORD so the first admin never has
    # the well-known password.
    os.environ["CINEMATCH_ADMIN_PASSWORD"] = "deploy-secret-1"
    try:
        with scratch_accounts():
            check("a deployment's secret becomes the first admin's password",
                  accounts.authenticate("admin", "deploy-secret-1")["role"] == "admin"
                  and not accounts.using_default_password())
            refused("the well-known password does not work then",
                    lambda: accounts.authenticate(*config.DEFAULT_ADMIN), accounts.AccountError)
    finally:
        del os.environ["CINEMATCH_ADMIN_PASSWORD"]

    # Moving to an online database: the accounts in the local file are copied
    # with their hashes and salts, so every password still works afterwards.
    with scratch_accounts() as scratch:
        accounts.create_user("alice", "secret1")
        accounts.change_password("admin", "admin-pass-1")
        old_file = config.DATABASE
        config.DATABASE = scratch.root / "new.db"      # stands in for the online database
        message = accounts.copy_accounts_from(old_file)
        check("copying moves every account to the new database", "Copied 2" in message, message)
        check("the copied passwords still work",
              accounts.authenticate("alice", "secret1")["role"] == "user"
              and accounts.authenticate("admin", "admin-pass-1")["role"] == "admin")
        check("the copied admin is used instead of a new first-run admin",
              not accounts.using_default_password())
        check("copying again leaves the accounts already there alone",
              "Copied 0" in accounts.copy_accounts_from(old_file))


def test_online_accounts() -> None:
    """The same account rules in PostgreSQL (the DATABASE_URL database).

    Runs only when CINEMATCH_TEST_DATABASE_URL is set.  Everything happens in a
    schema of its own, cinematch_test, which is dropped afterwards, so even the
    real online database can be given: its own users table is never touched.
    """
    section("8b. Accounts in an online PostgreSQL database")
    url = os.environ.get("CINEMATCH_TEST_DATABASE_URL")
    if not url:
        print("  SKIP  set CINEMATCH_TEST_DATABASE_URL to run these")
        return
    import psycopg
    from psycopg.conninfo import make_conninfo

    from cinematch import accounts

    def reset_schema(create: bool) -> None:
        with psycopg.connect(url, autocommit=True) as db:
            db.execute("DROP SCHEMA IF EXISTS cinematch_test CASCADE")
            if create:
                db.execute("CREATE SCHEMA cinematch_test")

    def raw_sql(statement: str, *values) -> bool:
        """True if the database accepted the statement."""
        try:
            with psycopg.connect(test_url) as db:
                db.execute(statement, values)
            return True
        except psycopg.IntegrityError:
            return False

    reset_schema(create=True)
    test_url = make_conninfo(url, options="-c search_path=cinematch_test")
    try:
        with scratch_accounts(online=test_url) as scratch:
            users = accounts.read_users()
            check("the first run creates one admin online",
                  list(users["username"]) == ["admin"] and list(users["role"]) == ["admin"])
            info = accounts.database_info()
            check("database_info names the online database",
                  info["kind"] == "PostgreSQL" and info["online"], str(info))

            accounts.create_user("Alice", "secret1")
            check("a new account can sign in", accounts.authenticate("alice", "secret1") is not None)
            refused("a wrong password is refused",
                    lambda: accounts.authenticate("alice", "wrong-password"))
            refused("a duplicate username is refused",
                    lambda: accounts.create_user("alice", "secret1"), accounts.AccountError)
            refused("the last admin cannot be demoted",
                    lambda: accounts.set_role("admin", "user", changed_by="admin"),
                    accounts.AccountError)
            refused("the last admin cannot be deleted",
                    lambda: accounts.delete_user("admin", deleted_by="alice"),
                    accounts.AccountError)
            accounts.set_role("alice", "admin", changed_by="admin")
            accounts.set_role("admin", "user", changed_by="alice")
            check("with two admins, one can step down",
                  accounts.get_user("admin")["role"] == "user")
            accounts.set_role("admin", "admin", changed_by="alice")

            insert = ("INSERT INTO users (username, password_hash, salt, role, created_on) "
                      "VALUES (%s, 'x', 'x', %s, 'now')")
            check("the database only accepts the roles admin and user",
                  not raw_sql(insert, "carol", "superuser") and raw_sql(insert, "carol", "user"))
            check("the database refuses a duplicate username", not raw_sql(insert, "carol", "user"))
            check("the database refuses an uppercase username", not raw_sql(insert, "Dave", "user"))
            accounts.delete_user("alice", deleted_by="admin")
            check("the database will not delete the last admin",
                  not raw_sql("DELETE FROM users WHERE role = 'admin'"))
            check("the database will not demote the last admin",
                  not raw_sql("UPDATE users SET role = 'user' WHERE role = 'admin'"))
            check("an injection attempt in the sign-in form is just a wrong name",
                  accounts.get_user("admin' OR '1'='1") is None)

            # Accounts made on this computer before DATABASE_URL was set.
            config.database_url = lambda: None
            config.DATABASE = scratch.root / "local.db"
            accounts.create_user("bob", "bobpass1")
            config.database_url = lambda: test_url
            message = accounts.copy_accounts_from(config.DATABASE)
            check("local accounts are copied online, keeping their passwords",
                  "Copied 1 account(s): bob" in message
                  and accounts.authenticate("bob", "bobpass1")["role"] == "user", message)
    finally:
        entry = accounts._pools.pop(test_url, None)
        if entry:
            entry["pool"].close()
        reset_schema(create=False)


def _movie(movies: pd.DataFrame, title: str, year: int | None = None) -> pd.Series:
    """The catalogue row for a title (the most popular one, or the one from that year)."""
    rows = movies[movies["title"] == title]
    if year:
        rows = rows[rows["year"] == year]
    return rows.iloc[0]


def _log(*events) -> pd.DataFrame:
    """A hand-made activity log: (timestamp, event, movie_id, detail) tuples."""
    return pd.DataFrame([{"timestamp": t, "event": e, "movie_id": "" if m is None else str(m),
                          "title": "", "detail": d} for t, e, m, d in events],
                        columns=config.ACTIVITY_COLUMNS)


NO_LIST = pd.DataFrame(columns=config.MY_LIST_COLUMNS)

# The movies the "viewer" browses in sections 9-11, oldest first: a Christopher
# Nolan fan who has lately been drifting towards science fiction.
SAMPLE = [("The Dark Knight", 2008), ("Inception", None), ("Interstellar", None),
          ("The Prestige", None), ("Batman Begins", None), ("Toy Story", None),
          ("Memento", None), ("Blade Runner", 1982), ("The Matrix", None), ("Arrival", None)]


def test_activity(movies: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    section("9. Learning from what a user does (no uploads)")
    from datetime import datetime, timedelta

    from cinematch import accounts, activity

    now = datetime(2026, 10, 13, 12, 0)
    stamp = lambda days=0, hours=0: (now - timedelta(days=days, hours=hours)).strftime(
        activity.TIME_FORMAT)
    toy, inception, up = (int(_movie(movies, t)["id"]) for t in ("Toy Story", "Inception", "Up"))

    def score(profile: pd.DataFrame, movie_id: int) -> float:
        rows = profile.loc[profile["movie_id"] == movie_id, "score"]
        return float(rows.iloc[0]) if len(rows) else 0.0

    # --- how much each action counts
    profile, _ = activity.build_profile(_log(
        (stamp(), "view", toy, ""), (stamp(), "search", inception, ""),
        (stamp(), "pick", up, "")), NO_LIST, movies, now=now)
    check("a pick counts more than a search, and a search more than a view",
          score(profile, up) > score(profile, inception) > score(profile, toy) > 0)

    profile, _ = activity.build_profile(_log(
        (stamp(), "view", toy, ""), (stamp(days=config.HALF_LIFE_DAYS), "view", up, "")),
        NO_LIST, movies, now=now)
    check(f"an action {config.HALF_LIFE_DAYS} days old counts half as much",
          abs(score(profile, up) - score(profile, toy) / 2) < 1e-6)

    profile, _ = activity.build_profile(_log(*[(stamp(), "view", toy, "")] * 50),
                                        NO_LIST, movies, now=now)
    check("opening one movie over and over is capped",
          score(profile, toy) == config.MAX_EVENT_SCORE and profile["views"].iloc[0] == 50)

    profile, _ = activity.build_profile(_log((stamp(), "search", None, "toy story")),
                                        NO_LIST, movies, now=now)
    check("a typed search counts towards the titles it finds",
          toy in set(profile["movie_id"]) and len(profile) <= config.TEXT_SEARCH_HITS,
          str(list(profile["title"])))

    _, genres = activity.build_profile(_log((stamp(), "genre", None, "Horror"),
                                            (stamp(), "genre", None, "Horror")),
                                       NO_LIST, movies, now=now)
    check("browsing a genre is remembered", genres.get("Horror") == 2.0, str(genres))

    profile, _ = activity.build_profile(_log(
        (stamp(hours=2), "view", toy, ""), (stamp(hours=1), "view", up, "")),
        NO_LIST, movies, now=now)
    check("the profile is ordered by when each movie was last touched",
          list(profile["movie_id"]) == [toy, up])

    with scratch_accounts():
        accounts.create_user("viewer", "secret1")

        activity.record(None, "view", toy)
        activity.record("viewer", "not-an-event", toy)
        check("nothing is recorded for nobody, or for an unknown action",
              activity.read_activity("viewer").empty)

        activity.record_search("viewer", "inception", movies)
        activity.record_search("viewer", "dark kni", movies)
        log = activity.read_activity("viewer")
        check("searching an exact title records that movie",
              log["movie_id"].iloc[0] == str(inception) and log["event"].iloc[0] == "search")
        check("searching other text records the text",
              log["movie_id"].iloc[1] == "" and log["detail"].iloc[1] == "dark kni")
        activity.clear_activity("viewer")

        # --- the viewer browses the sample, oldest first
        for n, (title, year) in enumerate(SAMPLE):
            movie = _movie(movies, title, year)
            when = now - timedelta(hours=3 * (len(SAMPLE) - n))
            activity.record("viewer", "view", movie["id"], title, when=when)
            if n % 2 == 0:
                activity.record("viewer", "search", movie["id"], title,
                                when=when + timedelta(minutes=1))
            if title in ("The Dark Knight", "Interstellar", "Arrival"):
                activity.record("viewer", "pick", movie["id"], title,
                                when=when + timedelta(minutes=2))
        activity.record("viewer", "genre", detail="Science Fiction", when=now)
        # My List: the latest three sci-fi films, and one they did not enjoy.
        for title, year, rating in [("Blade Runner", 1982, 8), ("The Matrix", None, 9),
                                    ("Arrival", None, 8), ("Batman & Robin", None, 2)]:
            movie = _movie(movies, title, year)
            file_manager.add_to_list(movie["id"], title, "Watched", my_rating=rating,
                                     username="viewer")
        profile, genre_interest = activity.profile_for("viewer", movies)

        disliked = profile[profile["disliked"]]
        check("a movie rated low on My List counts as disliked, with no score",
              list(disliked["title"]) == ["Batman & Robin"] and disliked["score"].iloc[0] == 0)
        check("My List and a high rating raise a movie's score",
              score(profile, int(_movie(movies, "The Matrix")["id"]))
              > score(profile, int(_movie(movies, "Memento")["id"])))
        check("every sampled movie is in the profile",
              {t for t, _ in SAMPLE} <= set(profile["title"]))

        message = activity.forget_movies("viewer", [toy])
        check("a movie can be forgotten",
              toy not in set(activity.profile_for("viewer", movies)[0]["movie_id"]), message)
        for n, (title, year) in enumerate(SAMPLE):           # put Toy Story back
            if title == "Toy Story":
                activity.record("viewer", "view", toy, title,
                                when=now - timedelta(hours=3 * (len(SAMPLE) - n)))
        profile, genre_interest = activity.profile_for("viewer", movies)

        summary = activity.summary(activity.read_activity("viewer"))
        check("the actions are counted by kind",
              summary["view"] == 10 and summary["pick"] == 3 and summary["genre"] == 1,
              str(summary))
        saved = activity.read_activity("viewer")
        activity.clear_activity("viewer")
        check("all activity can be cleared", activity.read_activity("viewer").empty)
        file_manager.write_csv_file(saved, config.activity_path("viewer"))
        check("deleting the account removes its activity",
              accounts.delete_user("viewer", deleted_by="admin")
              and not config.activity_path("viewer").exists())
    return profile, genre_interest


def test_collaborative(movies: pd.DataFrame) -> "collaborative.BehaviourModel":
    section("10. Collaborative filtering: learning from viewers' behaviour")
    from cinematch import collaborative

    # Four movies, three viewers: the two who love movie 0 also love movie 1.
    tiny = pd.DataFrame({"row": [0, 1, 0, 1, 2, 3, 2], "viewer": ["a", "a", "b", "b", "c", "c", "a"],
                         "value": [2.0, 2.0, 1.5, 1.0, 2.0, 2.0, -1.0]})
    model = collaborative.build_model(4, tiny)
    scores = model.scores([0], [1.0])
    check("movies liked by the same viewers are close", scores[1] > scores[2] and scores[1] > scores[3],
          str(scores.round(2)))
    check("a movie nobody rated has no behaviour",
          collaborative.build_model(5, tiny).scores([4], [1.0]) is None)

    movielens = collaborative.load_movielens(movies)
    check("the MovieLens ratings load and map onto the catalogue",
          len(movielens) > 50_000 and movielens["row"].between(0, len(movies) - 1).all(),
          f"{len(movielens):,} ratings")
    check("each viewer's ratings are centred on their own average",
          abs(movielens.groupby("viewer")["value"].mean()).max() < 1.5)

    model = collaborative.build_model(len(movies), movielens)
    toy = int(movies.index[movies["title"] == "Toy Story"][0])
    ranked = pd.Series(model.scores([toy], [1.0])).drop(toy).sort_values(ascending=False)
    top = list(movies.loc[ranked.head(15).index, "title"])
    check("people who liked Toy Story liked other family films",
          bool({"Toy Story 2", "Monsters, Inc.", "Finding Nemo", "A Bug's Life",
                "The Incredibles", "Aladdin", "The Lion King", "Shrek"} & set(top)), str(top))
    check("the model reports what it learned from",
          model.info["movielens_viewers"] > 500 and model.info["app_viewers"] == 0, str(model.info))

    profile = pd.DataFrame({"movie_id": [int(movies.loc[toy, "id"]), int(movies["id"].iloc[1])],
                            "score": [6.0, 3.0], "disliked": [False, True]})
    app = collaborative.app_user_ratings({"alice": profile}, movies)
    check("a CineMatch account becomes one more viewer, a disliked movie a negative rating",
          list(app["value"]) == [2.0, -2.0] and set(app["viewer"]) == {"app_alice"})

    # A deployment ships without the MovieLens files: the model learns from the accounts alone.
    saved = config.RATINGS_CSV
    config.RATINGS_CSV = config.DATA_DIR / "_no_such_ratings.csv"
    try:
        without = collaborative.load_movielens(movies)
    finally:
        config.RATINGS_CSV = saved
    accounts_only = collaborative.build_model(len(movies), without, app)
    check("without the MovieLens files the model still builds from the accounts alone",
          without.empty and accounts_only.info["movielens_viewers"] == 0
          and accounts_only.info["app_viewers"] == 1, str(accounts_only.info))
    return collaborative.build_model(len(movies), movielens, app)


def test_patterns(movies: pd.DataFrame, profile: pd.DataFrame, genre_interest: dict,
                  behaviour) -> None:
    section("11. Watch pattern and pattern-based recommendations")
    from cinematch import patterns

    engine = recommender.build_recommender(movies)
    pattern = patterns.analyze(profile, movies, genre_interest)
    check("an empty profile has no pattern", patterns.analyze(profile.head(0), movies) == {})
    check("every liked movie is analysed, the disliked one is not",
          pattern["count"] == 10 and pattern["disliked"] == 1, str(pattern["count"]))
    check("the recurring director is found",
          bool(pattern["directors"]) and pattern["directors"][0][0] == "Christopher Nolan",
          str(pattern["directors"]))
    check("a director seen once is not a pattern",
          all(n >= config.MIN_REPEAT for _, n in pattern["directors"]))
    check("the unfinished Dark Knight series is found",
          any(s["name"] == "The Dark Knight Collection" and s["missing_ids"]
              for s in pattern["series"]))
    check("the favourite decade is found", pattern["top_decade"] in {"2000s", "2010s"},
          pattern["top_decade"])
    check("the recent shift towards Science Fiction is noticed",
          "Science Fiction" in pattern["recent"].get("rising", []), str(pattern["recent"]))
    check("the pattern is described in sentences",
          len(pattern["insights"]) >= 5
          and "Christopher Nolan" in " ".join(pattern["insights"]))

    recs = patterns.recommend(engine, movies, profile, pattern, top_n=12, behaviour=behaviour)
    titles = list(recs["title"])
    check("12 recommendations are returned", len(recs) == 12)
    check("nothing the user already touched is recommended",
          not set(recs["id"]) & set(profile["movie_id"]))
    check("scores are between 0 and 100 and sorted",
          recs["similarity_pct"].between(0, 100).all()
          and recs["similarity_pct"].is_monotonic_decreasing)
    check("the next film of the started series ranks first",
          titles[0] == "The Dark Knight Rises", titles[0])
    check("the recurring director's other films are recommended",
          any("Christopher Nolan" in r for reasons in recs["reasons"] for r in reasons))
    check("the viewers' behaviour feeds the ranking",
          recs["people_like_you"].max() > 0, str(recs["people_like_you"].round(2).tolist()))
    check("every recommendation has at least one reason",
          all(len(r) >= 1 for r in recs["reasons"]))
    check("finish_series lists unwatched sequels",
          "The Dark Knight Rises" in set(patterns.finish_series(pattern, movies)["title"]))
    rows = patterns.because_you_liked(engine, movies, profile, exclude_ids=recs["id"])
    shown = [i for _, matches in rows for i in matches["id"]]
    check("'Because you liked' gives a row of similar films for each favourite",
          len(rows) == 3 and all(len(m) == 6 for _, m in rows)
          and all(seed["id"] in set(profile.loc[profile["score"] > 0, "movie_id"])
                  for seed, _ in rows), str([s["title"] for s, _ in rows]))
    check("no film is shown twice, touched or already in the top picks",
          len(shown) == len(set(shown))
          and not set(shown) & (set(profile["movie_id"]) | set(recs["id"])))
    check("the favourite seeds are the highest-scoring movies",
          rows[0][0]["id"] == profile.loc[profile["score"].idxmax(), "movie_id"])
    check("recommendations still work without the behaviour model",
          len(patterns.recommend(engine, movies, profile, pattern, top_n=6)) == 6)

    low = profile.assign(score=0.0, disliked=True)
    check("a user who disliked everything gets no pattern yet",
          patterns.analyze(low, movies) == {})
    check("the site summary counts genres across users",
          patterns.site_summary({"a": profile, "b": profile}, movies)["Drama"]
          == 2 * pattern["genre_movies"]["Drama"])


# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 68)
    print("CineMatch test suite")
    print("=" * 68)

    config.ensure_dirs()

    try:
        test_preprocessing()
        movies = test_catalogue()
        test_source_is_protected()
        test_recommendations(movies)
        test_caching(movies)
        test_my_list(movies)
        test_messy_data()
        test_accounts()
        test_online_accounts()
        profile, genre_interest = test_activity(movies)
        behaviour = test_collaborative(movies)
        test_patterns(movies, profile, genre_interest, behaviour)
    except Exception:
        print("\nA test crashed before it could finish:\n")
        traceback.print_exc()
        return 1

    print()
    print("=" * 68)
    print(f"RESULT: {PASSED} passed, {FAILED} failed")
    print("=" * 68)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
