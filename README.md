# CineMatch — A Content-Based Movie Recommendation System

A movie recommendation app built for a college midterm project. Browse about
9,000 well-known films, open any movie to see its details, ask for the movies
most like it, and keep your own list of movies to watch. CineMatch uses
**TF-IDF vectorisation** and **cosine similarity** on each movie's own content,
and blends everything on your list into a personal **"Recommended for you"**.

---

## Purpose

Streaming catalogues are too large to browse. CineMatch answers two everyday
questions:

- *"I liked this — what should I watch next?"*
- *"Given everything I've saved, what else would I like?"*

It answers them using only what each movie **is about**: its genres, plot
keywords, cast, director, the series it belongs to, and its plot summary. It
never needs other people's ratings, so it runs from local files and can explain
every recommendation it makes.

## Target users

| User | What they get |
|---|---|
| Movie viewers | "I liked this — what next?", with the reason for every match |
| People who keep a watchlist | One place to track what they want to watch, are watching and have seen, with their own ratings and notes |
| Students / presenters | A recommender whose every step can be explained without a black box |

## Main features

### Accounts: admin and user

Everyone signs in first. There are two roles:

| Role | Can do |
|---|---|
| **User** | Browse, search and keep their own My List; CineMatch **learns their watch pattern from that behaviour** and recommends from it |
| **Admin** | Everything a user can (their behaviour is learned from too), plus **Manage users** (create, delete, reset passwords, change roles), **User patterns** (open what CineMatch has learned about any user and their recommendations), **Admin dashboard** and the **Activity log** |

The first run creates one admin, **`admin` / `admin123`**. Sign in with it and
change the password from the sidebar (the app reminds you until you do). New
people sign up from the *Create account* tab and are always users (an
*Account Created!* pop-up then offers **Proceed to CineMatch**, which signs them
in, or **Back to sign in**, which returns to the sign-in tab with the new
username filled in); only an
admin can make another admin. Passwords are stored hashed (PBKDF2-SHA256 with a
random salt per account), never as typed.

#### The users database

By default, accounts are kept in a **SQLite** database, `data/cinematch.db`,
created automatically on the first run. SQLite is one file and is built into
Python (`sqlite3`), so there is no database server to install or start, but
the file is only on the computer the app runs on. To use the same accounts
on every device, put them in an online PostgreSQL database instead (see
[Sharing accounts between devices](#sharing-accounts-between-devices)). The
table and its rules are the same in both.

```sql
CREATE TABLE users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE
                  CHECK (length(username) BETWEEN 3 AND 20 AND username = lower(username)),
    password_hash TEXT NOT NULL,
    salt          TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('admin', 'user')),
    created_on    TEXT NOT NULL,
    last_login    TEXT NOT NULL DEFAULT ''
);
```

The rules are part of the table, so they hold even for SQL typed straight
into the database:

| Rule | Enforced by |
|---|---|
| A role is only `admin` or `user` | `CHECK (role IN ('admin', 'user'))` |
| No two accounts share a username | `UNIQUE` |
| The last admin can't be deleted or made a user | triggers `keep_an_admin_on_delete` and `keep_an_admin_on_update` (PostgreSQL: `keep_an_admin`) |
| Typed input is never run as SQL | every query uses placeholders |
| A failed change leaves nothing half-saved | each action runs in one transaction |

To look inside it, open `data/cinematch.db` with any SQLite viewer (for
example *DB Browser for SQLite*), or run:

```
python -c "import sqlite3; print(sqlite3.connect('data/cinematch.db').execute('SELECT id, username, role, created_on, last_login FROM users').fetchall())"
```

#### Sharing accounts between devices

When `DATABASE_URL` is set, accounts go into that online PostgreSQL database
instead of `data/cinematch.db`. Every computer and every deployment that has
the URL then signs in against the same accounts. A free database is enough:

1. Sign up at <https://neon.tech> (free plan), create a project, and copy its
   **connection string**. It looks like
   `postgresql://user:password@ep-something.aws.neon.tech/neondb?sslmode=require`.
   (Supabase works too. Use its *Session pooler* connection string.)
2. Put it in `.streamlit/secrets.toml` on each computer that runs the app.
   The file is git-ignored, so the password is never committed:

   ```toml
   DATABASE_URL = "postgresql://user:password@ep-something.aws.neon.tech/neondb?sslmode=require"
   ```

   On Streamlit Community Cloud, paste the same line into the app's
   **Secrets**.
3. Install the driver (`pip install -r requirements.txt`) and start the app.
   The `users` table and its rules are created on the first run.
4. To bring along the accounts already made on this computer, run this once
   **before** first starting the app with the URL. Passwords keep working,
   because the hashes and salts are copied as they are:

   ```
   python -m cinematch.accounts
   ```

   If the app has already created a first-run admin online, that admin is
   kept and the local one is skipped; every other account is still copied.

The admin's **Manage users** page shows which database is in use. Only the
accounts move online. Each user's My List, their activity and the activity log
are still files in `data/` on the computer that runs the app.

### Recommended: picks learned from your behaviour, no uploads

Nobody fills anything in. While someone uses the site, CineMatch records what
they do in `data/users/<name>/activity.csv` (`cinematch/activity.py`):

| Action | Where it happens | Weight |
|---|---|---|
| Opened a movie | clicking any poster | 1.0 |
| Searched a title | Discover's search box (typed text such as "batman" is spread over the top 3 titles it finds, at 0.5 each) | 2.0 |
| *Find movies like this* | the details pop-up on Discover | 3.0 |
| Browsed a genre | Discover's genre filter, Home's genre pills | 1.0 to that genre |
| On My List | *Want to watch* 4, *Watching* / *Watched* 5, multiplied by your rating / 7 | 4 to 7 |

Each movie's **interest score** adds up its actions. Old actions fade (half
the weight every 30 days), one movie can't gain more than 10 points from
clicks alone, and a movie you rate below 6 scores 0: it is kept out of the
pattern and never recommended. Users and admins are learned from the same way.

The **Recommended** page turns that pattern into movies:

1. **Top picks for you**: every movie is scored against the pattern, and each
   pick shows **why** (e.g. *"Continues The Dark Knight Collection (you looked
   at 2 of 3)"*, *"Directed by Christopher Nolan (6 of your movies)"*,
   *"Viewers who like your movies like this one"*).
2. **Because you liked …**: a row of the films most similar to each of your
   three strongest favourites (*Because you liked Arrival*, *Because you liked
   The Matrix*...). No film appears twice on the page, and nothing you have
   already searched, opened or saved is suggested.
3. **Finish the series**: the next films of series you started.

Below the picks, two folded sections:

- **What CineMatch has noticed about your taste**: the pattern in plain
  sentences (top genres and mixes, directors and actors found in 2+ movies,
  recurring themes, favourite decade, typical length and rating level,
  mainstream vs. lesser-known, language, what you have been looking at
  *lately*, which genres you rate highest), with two small charts.
- **Manage what CineMatch learned**: every movie that counts, its score and
  how many times you opened, searched and picked it (tick *Forget* to drop
  movies, or clear all activity), and your recent actions.

Before CineMatch has learned anything, the page explains how to get personal
picks and shows what is popular.

The Home page opens with a *Picked for you* shelf from the pattern and a
*Recently viewed* shelf.

### Browsing and My List

1. **Home**: poster shelves. *Picked for you* (from the learned pattern),
   *Recently viewed*, *Continue watching* (from My List), *Trending now*,
   *Highest rated* and *Browse by genre*.
2. **Discover**: search with live title suggestions, filters for genre,
   minimum rating and decade, a paged poster wall sorted **A–Z** (Home and
   *Recommended* already show what is popular; when searching, titles that start
   with the search come first), and *Surprise me*.
3. **My List**: your saved movies with counts per status, a filter and sort,
   a table for editing several movies at once, a CSV download, and *Recommended
   for you*.

There are also two pop-up windows:

- **Movie details**: opens when you click any poster. It shows the rating,
  runtime, director, cast, keywords, series and plot, plus the **My List
  controls**: add the movie, change its status, rate it 1–10, write notes, or
  remove it.
- **Movies like your pick**: opens from *Find movies like this* or *Surprise
  me*. It shows the 5 or 10 closest matches with a match percentage and the
  features they share. Click a match to explore from it.

---

## Dataset

The project uses **The Movies Dataset** (TMDB metadata + MovieLens), stored in
the `movies data/` folder. It was inspected before any code was written by
running `python inspect_dataset.py`.

| File | Size | Rows | Used for |
|---|---|---|---|
| `movies_metadata.csv` | 34.4 MB | 45,466 | title, genres, overview, tagline, date, runtime, rating, series, poster |
| `credits.csv` | 189.9 MB | 45,476 | cast (billing order) and crew → director |
| `keywords.csv` | 6.2 MB | 46,419 | plot keywords |
| `ratings*.csv`, `links*.csv` | ~710 MB | 26 million | not used (they serve collaborative filtering) |

### The catalogue

On the first run the three files are merged on the movie id and filtered to the
movies worth recommending:

| Filter | Movies left |
|---|---|
| All rows | 45,466 |
| Valid numeric id (3 rows hold dates) | 45,463 |
| Released | 45,098 |
| Not adult | 45,090 |
| At least 50 votes | 9,140 |
| Has an overview and a poster path | 9,106 |
| Duplicate ids removed | **9,097** |

The result is saved as `data/catalogue.csv` (18 columns, ~5.9 MB) and simply
read on every later run.

| Property | Value |
|---|---|
| Release years | 1895 – 2017 |
| Genres | 20 distinct |
| People | 4,205 directors, 17,144 actors (top five billed per film) |
| Series | 1,994 films belong to a collection |
| Average rating | 6.36 / 10 |

### Columns used for recommendations

| Column | Why it is useful |
|---|---|
| `genres` | The strongest signal of what kind of movie it is |
| `keywords` | Plot themes: `superhero`, `gotham city`, `time travel` |
| `collection` | Links sequels together (`Toy Story Collection`) |
| `cast` | The five top-billed actors |
| `director` | Directors have recognisable styles |
| `overview` | The plot summary, in free text |
| `tagline` | Short marketing line, adds a few useful words |

### Missing values found (and what was done)

| Column | Missing (of 9,097) | Handling |
|---|---|---|
| `collection` | 7,103 | Filled with `""`, since most films are not in a series |
| `tagline` | 2,014 | Filled with `""` |
| `keywords` | 653 | Filled with `""` |
| `cast` | 27 | Filled with `""` |
| `director` | 11 | Filled with `""` |
| `genres` | 5 | Filled with `""` |
| `release_date` | 1 | Filled with `""`; year becomes 0 |

9,814 values filled, no rows lost. A movie missing only its tagline is still a
good record, and an empty field simply contributes no words to the model.

### Posters

Every movie has a TMDB `poster_path`, loaded from TMDB's public image server.
The dataset dates from 2017, and TMDB has since replaced many images, so
**roughly nine in ten old poster links no longer load**. Wherever an image
fails, the app shows generated poster art (colours from the genre, the title
set as the artwork) instead of a broken image.

To get real posters for all of them, get a free API key at
<https://www.themoviedb.org/settings/api>, put it in the `TMDB_API_KEY`
environment variable or in `data/tmdb_key.txt`, and run once:

```
python -m cinematch.posters
```

---

## Machine-learning functionality

### 1. One text field per movie (the "content soup")

Seven fields are cleaned and joined, each repeated to set its weight:

| Field | Weight |
|---|---|
| genres, keywords, collection | ×3 |
| cast, director | ×2 |
| overview, tagline | ×1 |

Each person, keyword phrase, genre and series becomes **one token**
(`James Cameron` → `jamescameron`, `time travel` → `timetravel`), so a
different "James", or any film that merely mentions "time", does not look
similar.

```
animation comedy family                  (x3)
jealousy toy boy friendship boynextdoor  (x3)
toystorycollection                       (x3)
tomhanks timallen donrickles             (x2)
johnlasseter                             (x2)
led by woody andys toys live happily     (x1)
```

### 2. TF-IDF vectorisation

`TfidfVectorizer(stop_words="english", max_features=30000, min_df=2,
sublinear_tf=True)` turns the soups into a **9,097 × 29,059** sparse matrix
(0.137% dense, fitted in under a second).

### 3. Cosine similarity

The chosen movie's row is compared with every other row; the highest scores
are the recommendations. Only one row is computed per request. The full
9,097 × 9,097 matrix would need ~650 MB and is never built.

### 4. Explaining the match

`shared_terms()` finds the tokens both movies score on, and `readable()` turns
them back into names: *"Shares: The Dark Knight Collection, batman, tragic
hero"*.

### 5. Recommended for you

`recommend_for_profile()` averages the TF-IDF vectors of the movies on your
list, weighted by your rating (unrated counts as 7), and ranks the catalogue
against that "taste" vector. Movies you rated below 6 are left out, and movies
already on your list are never suggested back.

### Caching

The fitted model is saved to `data/cache/tfidf_cache.pkl` with an MD5
fingerprint of the text. It is reused only while the fingerprint still
matches.

### Example results

| Selected movie | Top recommendations (match %) |
|---|---|
| The Dark Knight | The Dark Knight Rises 42.2, Batman Begins 38.8, Batman: Under the Red Hood 28.6, Batman Returns 27.0 |
| Toy Story | Toy Story 2 41.1, Toy Story 3 34.2, Toy Story of Terror! 23.1 |
| Spirited Away | Hotarubi no Mori e 20.6, Children Who Chase Lost Voices 18.8, Tales from Earthsea 17.7, Pom Poko 15.9 |
| Avatar | Aliens 17.0, Star Trek Into Darkness 16.7, Cargo 13.8 |

### 6. Collaborative filtering (`collaborative.py`)

The TF-IDF model knows what a film is *about*. The collaborative model knows
nothing about the films; it learns only from **behaviour**: which films the
same people go for.

- **Training data**: 100,000 MovieLens ratings by 671 viewers
  (`ratings_small.csv`, mapped to TMDB ids with `links_small.csv`; about
  92,000 land on catalogue movies), plus **every CineMatch account** as one
  more viewer, built from its learned interest scores. Ratings are centred on
  each viewer's own average, so "liked it more than usual" means the same for a
  harsh and a generous rater.
- **Model**: an item x viewer sparse matrix with each row scaled to length 1
  (item-item cosine similarity). Your taste is the score-weighted sum of your
  movies' rows, and every film is scored by its cosine to it.
- It keeps learning: the CineMatch half is rebuilt on every request, so one
  user's behaviour immediately shapes what similar users are shown. The person
  being recommended to is left out of their own model.

Liking *Toy Story* alone, its nearest films by behaviour include other Pixar
and Disney titles, without the model ever reading a genre or keyword.

### 7. The watch pattern recommender (`patterns.py`)

`analyze()` finds what repeats across the movies you showed interest in, each
weighted by its score (a director or actor counts only when found in 2+
movies). `recommend()` then scores every movie in the catalogue:

| Part | Weight | What it measures |
|---|---|---|
| Content similarity | 35% | TF-IDF taste profile of your movies, each weighted by its interest score |
| People like you | 20% | Collaborative filtering (above). If it knows none of your movies, content similarity takes this share |
| Genre fit | 15% | How much of your interest (movies + genres browsed) shares the movie's genres |
| Era fit | 10% | How much of your interest is in its decade |
| Rating fit | 5% | How close its rating is to what you usually go for |
| People / series | 15% | Next film of a series you started, or a recurring director (full) or actor (60%) |

Nothing you have already searched, opened, picked or listed is recommended
back. For a viewer who browsed Nolan films, then *Blade Runner*, *The Matrix*
and *Arrival*, the first pick is *The Dark Knight Rises*.

---

## File-handling functionality

Low-level file I/O lives in `cinematch/file_manager.py`; `accounts.py` and
`activity.py` use its helpers for their own files.

| File | Mode | Purpose |
|---|---|---|
| `movies data/*.csv` | **read only** | The source dataset. Any write into this folder is refused |
| `data/catalogue.csv` | create, then read | The merged catalogue, built on the first run |
| `data/cinematch.db` | SQLite database | The `users` table: username, hashed password, salt, role, dates |
| `movies data/ratings_small.csv`, `links_small.csv` | **read only** | MovieLens ratings for the collaborative model |
| `data/activity_log.csv` | **append** | One line per sign-in, account change and admin action |
| `data/users/<name>/activity.csv` | **append**, read, rewrite | What the user searches, opens, picks and browses; the pattern is learned from it. *Forget* rewrites it without those movies; *Clear* deletes it |
| `data/users/<name>/my_list.csv` | read / write | **The user's list**, changed by every create, update and delete |
| `data/cache/tfidf_cache.pkl` | read / write | The saved model |
| downloads | write (browser) | My List, the learned movies, the activity log |

Usernames are limited to letters, numbers and `_`, and deleting an account
checks that the resolved path is inside `data/users/`, so a username can't be
used to reach other files.

### My List: CRUD from the user's side

Each row of `data/users/<name>/my_list.csv` holds `movie_id, title, status, my_rating,
notes, added_on, updated_on`. The status is one of *Want to watch*, *Watching*
or *Watched*.

| Action | Where you do it | Function |
|---|---|---|
| **Create** | *+ Add to My List* in a movie's pop-up | `add_to_list()` |
| **Read** | the My List page, the pop-up, the sidebar count, *Continue watching* | `read_my_list()`, `get_list_entry()` |
| **Update** | *Save changes* in the pop-up, or the edit table on My List | `update_list_entry()`, `replace_my_list()` |
| **Delete** | *Remove from My List* in the pop-up, or the table's *Remove* box | `remove_from_list()` |

Every change is validated first: no duplicate movies, only the three statuses,
and ratings from 1 to 10 or blank. Every write is **atomic**: it goes to a
temporary file first, then `os.replace()` swaps it in, so a crash cannot leave
a half-written file.

### Data processing pipeline

1. **Filter**: released, non-adult, 50+ votes, has an overview and a poster.
2. **Parse**: pull names out of the list-shaped text
   (`[{'id': 16, 'name': 'Animation'}]`) with a regular expression.
3. **Merge**: join the three files on the id; drop duplicate ids.
4. `fill_missing()`: text → `""`, numbers → `0`.
5. `drop_bad_rows()`: blank titles, duplicate rows and duplicate ids.
6. `add_helper_columns()`: the year, plus readable comma-separated lists.
7. `build_soup()`: the weighted text for the model.

---

## Technologies used

| Technology | Used for |
|---|---|
| Python 3.10+ | The language |
| pandas | Reading, merging, filtering and writing CSV data |
| scikit-learn | `TfidfVectorizer`, `cosine_similarity` |
| SciPy | Sparse item x viewer matrix for collaborative filtering |
| Streamlit (1.50+) | The web interface, pop-up dialogs, the editable table |
| SQLite (`sqlite3`, built in) | The users database (admin / user accounts), by default |
| psycopg 3 + psycopg-pool | The same users database in online PostgreSQL, when `DATABASE_URL` is set |
| Standard library | `csv`, `pathlib`, `pickle`, `hashlib`, `re`, `os`, `json`, `urllib` |

No database server is needed: accounts are in a SQLite file unless an online
PostgreSQL database is set, and everything else is CSV and text files.

## Project structure

```
py prog/
├── app.py                  entry point
├── inspect_dataset.py      dataset report
├── test_cinematch.py       automated tests (166 checks)
├── requirements.txt
├── movies data/            source dataset (READ-ONLY)
│   ├── movies_metadata.csv
│   ├── credits.csv
│   ├── keywords.csv
│   ├── ratings_small.csv   MovieLens ratings (optional, from Kaggle)
│   └── links_small.csv     MovieLens id -> TMDB id (optional)
├── cinematch/
│   ├── config.py           paths, filters, weights, list statuses
│   ├── accounts.py         users database (SQLite or PostgreSQL), sign-in, roles, log
│   ├── activity.py         recording behaviour, learning interest scores from it
│   ├── collaborative.py    item-item collaborative filtering (MovieLens + users)
│   ├── patterns.py         finding the watch pattern, recommending from it
│   ├── file_manager.py     catalogue building + My List CRUD
│   ├── preprocessing.py    cleaning + the content soup
│   ├── recommender.py      TF-IDF, cosine similarity, taste profile
│   ├── posters.py          poster links (+ optional TMDB refresh)
│   └── ui/
│       ├── state.py        cached loading
│       ├── components.py   styling, poster tiles, the details pop-up
│       ├── dashboard.py    Home
│       ├── recommend.py    Discover
│       ├── mylist.py       My List
│       ├── watch_pattern.py  Recommended (picks, similar films, what was learned)
│       ├── login.py        sign in / create account
│       └── admin.py        admin dashboard, users, user patterns, activity log
└── data/                   created automatically
    ├── catalogue.csv
    ├── cinematch.db        the users database (SQLite)
    ├── activity_log.csv
    ├── users/<name>/       my_list.csv, activity.csv
    └── cache/tfidf_cache.pkl
```

## Installation

```
git clone https://github.com/BuladLami/cinematch.git
cd cinematch
pip install -r requirements.txt
streamlit run app.py
```

Python 3.11 or newer is needed (the pinned pandas 3 requires it).

The repository ships the prebuilt catalogue (`data/catalogue.csv`), so the app
runs straight after cloning. The source dataset is **not** in the repository:
it is ~900 MB (two files exceed GitHub's 100 MB limit) and the MovieLens
ratings have their own terms. Download
[The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
from Kaggle into `movies data/` next to `app.py` if you want to:

- rebuild the catalogue, run `inspect_dataset.py` or the full test suite
  (`movies_metadata.csv`, `credits.csv`, `keywords.csv`);
- add 671 MovieLens viewers to collaborative filtering (`ratings_small.csv`,
  `links_small.csv`). Without them, the "people with the same taste" part
  learns from the CineMatch accounts alone, and until it knows any of a user's
  movies the content model takes its share.

## Deploying to Streamlit Community Cloud

1. On <https://share.streamlit.io>, choose **Create app**, then the
   `BuladLami/cinematch` repository, branch `main`, main file `app.py`.
2. Under **Advanced settings**, pick **Python 3.12** (any version from 3.11
   works) and add this secret, with a password of your own:

   ```toml
   CINEMATCH_ADMIN_PASSWORD = "choose-a-strong-password"
   ```

   The first admin account (`admin`) is created with it. Without the secret
   it would be `admin` / `admin123`, which anyone could use on a public app.
3. Deploy. The first start installs the pinned packages and fits the TF-IDF
   model (about a minute); after that the app starts in seconds.

**Storage is temporary on Community Cloud.** The app writes accounts, My Lists
and activity to `data/` on the server's disk, and that disk is wiped whenever
the app restarts or is redeployed (after a code push, or after the app sleeps
from inactivity). Everything works while it runs, but what CineMatch has
learned starts over after a restart. Accounts start over too, unless you add a
`DATABASE_URL` secret (see
[Sharing accounts between devices](#sharing-accounts-between-devices)); with
it, accounts are kept online and survive restarts. Without it, the admin is
recreated from the secret.

## How to run

```
streamlit run app.py
```

The app opens at <http://localhost:8501>. Sign in as `admin` / `admin123` (or
the password in `CINEMATCH_ADMIN_PASSWORD`, if set), or create a user account.
The first run fits the model (a few seconds); later runs start immediately.

Other commands:

```
python inspect_dataset.py      # report on the raw dataset
python test_cinematch.py       # run the automated tests
python -m cinematch.posters    # refresh poster links (needs a TMDB key)
```

## Testing

`python test_cinematch.py` runs **166 checks in 11 groups**, all passing. It
needs no Streamlit, it saves and restores `data/my_list.csv`, and groups 8-9
run in a scratch folder that is deleted afterwards. Group 8b adds 15 more
checks of the online accounts database when `CINEMATCH_TEST_DATABASE_URL` is
set to a PostgreSQL URL (they run in a throwaway `cinematch_test` schema, so
the real accounts are never touched).

| Group | What is checked |
|---|---|
| 1. Preprocessing | Cleaning, accents, name squashing, list splitting, missing values, soup weights |
| 2. Catalogue | Name parsing (incl. apostrophes), directors from the crew, filters, unique ids, Toy Story's cast/series/keywords |
| 3. Protection | Writing into `movies data/` is refused; the file is unchanged |
| 4. Recommendations | Counts, score range and order, no self-match, Batman for The Dark Knight, Toy Story 2 for Toy Story, id lookup, readable explanations, taste profile |
| 5. Caching | Written, reused, rejected after a data change |
| 6. My List CRUD | Every action changes the file; bad input refused; the list steers recommendations |
| 7. Messy data | A movie with no text returns nothing instead of crashing |
| 8. Accounts | Default admin, hashed passwords, wrong/duplicate/short/path-like input refused, role rules, own My List per user, delete removes the folder, the log; and straight SQL against the database: bad roles, duplicates and removing the last admin are refused, injection text is harmless; accounts copied to a new database keep their passwords |
| 8b. Online accounts | The same rules in PostgreSQL (only when `CINEMATCH_TEST_DATABASE_URL` is set) |
| 9. Learning from activity | Pick > search > view, 30-day half-life, click cap, typed searches spread over their hits, genre browsing, ordering by recency, exact vs. typed searches recorded, My List and ratings raise scores, low ratings become dislikes, forget / clear, account deletion removes activity |
| 10. Collaborative filtering | Co-liked movies are close, unrated movies have no behaviour, MovieLens loads and is centred per viewer, Toy Story's neighbours are family films, app accounts become viewers |
| 11. Patterns | Recurring director, unfinished series, decade, recent shift to sci-fi, disliked movies left out, nothing already touched recommended, the series sequel ranks first, behaviour feeds the ranking, works without it, reasons on every pick, "Because you liked" rows without repeats |

---

## How this project satisfies the Midterm Requirements

### File reading
Reads three source files (~230 MB), the catalogue and the user's list, with
clear errors for missing, empty or malformed files (`read_csv_file()` also
falls back to Latin-1 encoding).

### File writing
Writes `data/catalogue.csv` and each user's `my_list.csv` atomically, appends
every search, view and pick to the user's `activity.csv` and every account
action to the activity log, saves the model cache, and lets the user download
the list as CSV.

### File updating
The user adds, updates and removes movies on their list, one at a time in the
pop-up or several at once in the edit table. Every change is saved to
their `my_list.csv` immediately and persists after restarting the app.

### File processing
Filters 45,466 movies to 9,097, parses list-shaped text, merges three files,
fills 9,814 missing values, and powers search, filters and sorting.

### Machine learning
A TF-IDF + cosine-similarity recommender with explanations; a system that
**learns each user's pattern from their behaviour** (searches, views, picks,
genres browsed, My List and ratings, with recency decay); an item-item
**collaborative-filtering** model trained on 100,000 MovieLens ratings plus
every account's learned behaviour; and a watch-pattern recommender that blends
both with the genres, decade, people and series found in the pattern.

### Admin and user
Two roles with separate pages: users browse and CineMatch learns their
pattern and recommends from it; the admin (whose behaviour is learned from too)
manages accounts, can open what was learned about any user and their
recommendations, and reads the activity log.
