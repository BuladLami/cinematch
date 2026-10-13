# CineMatch — Technical Documentation

This document explains how CineMatch works inside: how the code is organised,
how data moves through it, what every module does, and why the key decisions
were made. For a user-level overview see `README.md`.

## Table of contents

1. [Architecture at a glance](#1-architecture-at-a-glance)
2. [Installation and running](#2-installation-and-running)
3. [Data flow](#3-data-flow)
4. [Module reference](#4-module-reference)
5. [The CRUD operations (My List)](#5-the-crud-operations-my-list)
6. [The machine-learning pipeline](#6-the-machine-learning-pipeline)
7. [Learning the watch pattern from behaviour](#7-learning-the-watch-pattern-from-behaviour)
8. [Files on disk](#8-files-on-disk)
9. [Safety rules enforced by code](#9-safety-rules-enforced-by-code)
10. [Testing](#10-testing)
11. [Measured results](#11-measured-results)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Architecture at a glance

```
                     app.py                  entry point, sign-in gate, sidebar navigation
                       |
               cinematch/ui/*.py             Streamlit pages (presentation only)
                       |
   +----------+--------+---------+--------------+---------------+
   |          |                  |              |               |
accounts.py  activity.py     patterns.py   recommender.py  collaborative.py
 (SQLite)    (behaviour log  (the pattern + (TF-IDF content  (item-item CF on
             -> interest      blended recs)  similarity)      viewer behaviour)
              scores)
   |          |                                 |
   +----------+------ file_manager.py ----------+        all file I/O
                            |
                     preprocessing.py                     cleaning + feature building
                            |
                        config.py                         paths, weights and constants
```

Each layer only calls the layers below it. The UI never opens a file and never
calls scikit-learn or SciPy directly: it asks `file_manager` and `activity` for
data, and `recommender`, `collaborative` and `patterns` for predictions.
`preprocessing` never touches a file, so every cleaning step can be tested on a
plain DataFrame.

The movie catalogue is read-only. What people change is their own data: their
account (in `data/cinematch.db`), their My List, and their activity log, which
CineMatch writes for them as they browse and learns their watch pattern from.

---

## 2. Installation and running

```
# 1. install the dependencies
pip install -r requirements.txt

# 2. put the dataset in "movies data/" next to app.py
#    (movies_metadata.csv, credits.csv, keywords.csv,
#     and ratings_small.csv + links_small.csv for collaborative filtering)

# 3. run the app, then sign in as admin / admin123 (change it after)
streamlit run app.py

# 4. run the test suite (no Streamlit needed)
python test_cinematch.py

# optional
python inspect_dataset.py      # report on the raw dataset
python -m cinematch.posters    # refresh poster links (needs a TMDB key)
```

Requirements: Python 3.10+, pandas ≥ 2.0, scikit-learn ≥ 1.3, SciPy ≥ 1.10
(installed with scikit-learn), Streamlit ≥ 1.50 (the app relies on
`st.dialog(on_dismiss=...)`, `st.selectbox(accept_new_options=...)`, `st.pills`,
`st.segmented_control` and keyed containers). Without the two MovieLens files
the app still runs; the collaborative part simply has no MovieLens data.

---

## 3. Data flow

### First run

1. `state.catalogue_signature()` sees that `data/catalogue.csv` is missing and
   calls `file_manager.build_catalogue()` inside a progress box.
2. `build_catalogue()` reads `movies_metadata.csv`, filters it (released,
   non-adult, ≥ 50 votes, has an overview and a poster), then reads
   `keywords.csv` and `credits.csv`, keeping only the surviving ids.
3. Names are extracted from the list-shaped text with regular expressions; the
   three tables are merged on `id`, sorted by popularity and written atomically
   to `data/catalogue.csv`. This takes about 5 seconds.
4. `load_prepared_movies()` reads the catalogue and runs `preprocessing.prepare()`.
5. `recommender.build_recommender()` fits TF-IDF and saves
   `data/cache/tfidf_cache.pkl`.
6. `accounts._db()` creates `data/cinematch.db` with the `users` table and one
   admin (`admin` / `admin123`).

### Every later run

`catalogue_is_current()` is true (the catalogue is newer than every source
file), so the catalogue is simply read. The model's fingerprint matches the
cache, so the matrix is loaded instead of refitted. Both results, and the
MovieLens ratings for the collaborative model, are held in Streamlit caches
(`st.cache_data`, `st.cache_resource`) keyed on the catalogue file's size and
modification time.

### Signing in

`app.main()` shows `login.render()` until `session_state["user"]` holds an
account that still exists. Each run re-reads the account, so a role change made
by an admin applies at once. Admins get four extra pages in the sidebar.

### A user action becomes a signal

| The user... | Code | What is written |
|---|---|---|
| clicks any poster | `components.movie_grid()` | `view` + movie id |
| picks a title in Discover's search | `recommend._record_search()` (`on_change`) | `search` + movie id |
| types free text in the search | `activity.record_search()` | `search` + the text |
| presses *Find movies like this* | `recommend._pick()` | `pick` + movie id |
| changes the genre filter / genre pill | `recommend._record_genre()`, `dashboard._record_genre()` | `genre` + the genre |
| adds, rates or changes a movie on My List | `file_manager` (My List CRUD) | read as it stands, not logged |

Every line is appended to `data/users/<name>/activity.csv` by
`activity.record()`. Nothing is recorded when nobody is signed in.

### A personal recommendation request

Home or Recommended →
`activity.build_profile(log, my_list, movies)` scores every movie the user has
touched → `patterns.analyze(profile, movies, genre_interest)` finds the pattern →
`state.behaviour_model(movies, exclude=username)` builds the collaborative
model (cached MovieLens half + every other account's profile) →
`patterns.recommend(engine, movies, profile, pattern, behaviour=...)` blends
content, behaviour, genre, era, rating and people/series scores and attaches
the reasons.

### A "movies like this" request

Discover → click a poster → **Find movies like this** → `_pick()` records the
pick, `_choose(index)` sets `chosen_index` and a flag → the page reruns and
opens `_recommendations_dialog()` → `engine.recommend(index, top_n)` computes
one row of cosine similarities → `shared_terms()` + `readable()` explain each
match. Clicking a match sets a new `chosen_index` and reruns only the dialog
(`st.rerun(scope="fragment")`), because a pop-up cannot open another pop-up.

### A list change

Movie pop-up → **+ Add to My List** / **Save changes** / **Remove** →
`file_manager` rewrites `data/users/<name>/my_list.csv` → the dialog reruns as
a fragment and redraws with the new state. When the pop-up closes,
`on_dismiss="rerun"` refreshes the page behind it, so My List, the sidebar
count and the learned pattern all reflect the change. My List is **not**
cached: it is a tiny file read fresh whenever a page needs it.

---

## 4. Module reference

### 4.1 `config.py`

All paths and constants in one place.

| Name | Meaning |
|---|---|
| `SOURCE_DIR`, `METADATA_CSV`, `CREDITS_CSV`, `KEYWORDS_CSV` | The read-only source dataset |
| `RATINGS_CSV`, `LINKS_CSV` | MovieLens `ratings_small.csv` and `links_small.csv` (read only) |
| `CATALOGUE_CSV` | `data/catalogue.csv`, the merged catalogue |
| `DATABASE` | `data/cinematch.db`, the accounts database |
| `USERS_DIR`, `my_list_path()`, `activity_path()` | `data/users/<name>/my_list.csv` and `activity.csv` |
| `ACTIVITY_LOG_CSV` | `data/activity_log.csv`: sign-ins, account changes, admin actions |
| `TFIDF_CACHE` | The saved model |
| `MIN_VOTES = 50` | Catalogue filter |
| `CAST_SIZE = 5` | Top-billed actors kept |
| `LIST_SEP = "|"` | Separator for multi-value fields in the catalogue |
| `FEATURE_WEIGHTS` | genres 3, keywords 3, collection 3, cast 2, director 2, overview 1, tagline 1 |
| `MAX_TFIDF_FEATURES = 30000` | Vocabulary cap |
| `LIST_STATUSES` | Want to watch, Watching, Watched |
| `LIKED_RATING = 6` | Below this a listed movie counts as disliked |
| `ROLES`, `DEFAULT_ADMIN`, `MIN_PASSWORD_LENGTH`, `PASSWORD_ITERATIONS` | Accounts |
| `ACTIVITY_COLUMNS` | timestamp, event, movie_id, title, detail |
| `EVENT_WEIGHTS` | view 1.0, search 2.0, search_text 0.5, pick 3.0, genre 1.0 |
| `TEXT_SEARCH_HITS = 3` | A typed search counts towards its top 3 titles |
| `LIST_STATUS_WEIGHTS` | Want to watch 4, Watching 5, Watched 5 (× rating / 7) |
| `HALF_LIFE_DAYS = 30` | An action loses half its weight every 30 days |
| `MAX_EVENT_SCORE = 10` | Cap on what clicks alone can give one movie |
| `MIN_REPEAT = 2` | A director, actor or theme must appear in 2+ movies to be a pattern |
| `MIN_MOVIES_FOR_PATTERN = 5` | Below this the page says the pattern will sharpen |

### 4.2 `file_manager.py`: all file I/O

#### Safety helpers
- `_guard_source(path)`: raises `FileManagerError` if `path` is inside `movies data/`.
- `_atomic_write(df, path)` / `write_csv_file()`: guard → write `path.tmp` → `os.replace()`.
- `append_csv_row(row, path, columns)`: append one line (header first if new);
  used by both logs.
- `delete_user_folder(username)`: refuses any path not directly inside `data/users/`.

#### Reading
- `read_csv_file(path, **options)`: the single low-level read. Raises a clear
  error for a missing, empty or malformed file; retries in Latin-1 on a
  `UnicodeDecodeError`.

#### Building the catalogue
- `_names(text, pattern, limit)`: extracts names from text such as
  `[{'id': 16, 'name': 'Animation'}]`. The pattern accepts both quote styles,
  because a name containing an apostrophe is written as `"Peter O'Toole"`.
- `_DIRECTOR_PATTERN`: in crew entries `'job'` comes just before `'name'`, so
  `'job': 'Director', 'name': ...` isolates directors.
- `build_catalogue(progress)`: filter → parse → merge → write.
- `catalogue_is_current()`, `read_catalogue()`, `load_prepared_movies()`.

#### My List (see §5)
`read_my_list()`, `get_list_entry()`, `add_to_list()`, `update_list_entry()`,
`remove_from_list()`, `replace_my_list()`, `list_as_csv()`. Each takes the
signed-in `username`.

### 4.3 `preprocessing.py`: data cleaning

| Function | What it does |
|---|---|
| `clean_text()` | Lowercase, remove apostrophes, other punctuation → spaces. Accented letters are kept (`\w`) |
| `squash_name()` | `"James Cameron"` → `jamescameron`, one token per person or phrase |
| `split_list()` | `"Action|Comedy"` → `["Action", "Comedy"]` |
| `fill_missing()` | Text → `""`, numbers → `0`; returns a report of what was filled |
| `drop_bad_rows()` | Blank titles, exact duplicates, duplicate ids |
| `add_helper_columns()` | `year`, `genre_list`, `cast_list`, `keyword_list`, `director_list` |
| `build_soup()` | The weighted text column the model reads |
| `prepare()` | Runs all of the above, returns `(df, report)` |

### 4.4 `recommender.py`: content-based model (TF-IDF)

| Method | Purpose |
|---|---|
| `fit()` | Fits `TfidfVectorizer`; relaxes `min_df` to 1 on tiny tables |
| `save_cache()`, `try_load_cache()`, `ensure_model()` | Fingerprinted model cache |
| `search(query)` | Title contains the query; titles starting with it first, then by popularity |
| `index_of_id(movie_id)` | Row of a movie id (titles are not unique) |
| `recommend(index, top_n)` | Top matches for one movie |
| `profile_scores(indices, weights)` | Cosine of every movie to a weighted average of several |
| `recommend_for_profile(indices, weights, top_n)` | The same, ranked |
| `shared_terms(a, b)`, `readable(terms)` | Why two movies match, in display names |

### 4.5 `accounts.py`: users, roles and the activity log

One SQLite table, `users` (id, username, password_hash, salt, role, created_on,
last_login). The rules live in the table: `CHECK (role IN ('admin','user'))`,
`UNIQUE (username)`, and two triggers that refuse to delete or demote the last
admin. Passwords are PBKDF2-SHA256 (100,000 rounds, random salt per account),
compared with `hmac.compare_digest`. Every query uses `?` placeholders.

| Function | Purpose |
|---|---|
| `authenticate()` | Same message for an unknown name and a wrong password |
| `create_user()`, `change_password()`, `set_role()`, `delete_user()` | Account CRUD; deleting also removes `data/users/<name>/` |
| `read_users()`, `get_user()`, `database_info()` | Reads, never exposing hashes |
| `log()`, `read_log()` | `data/activity_log.csv` |

### 4.6 `activity.py`: recording behaviour and learning from it

| Function | Purpose |
|---|---|
| `record(username, event, movie_id, title, detail, when)` | Append one action; ignored for no user or an unknown event |
| `record_movie()`, `record_search()` | Helpers: a catalogue row; an exact title vs. typed text |
| `read_activity()`, `forget_movies()`, `clear_activity()` | Read the log; drop movies from it; delete it |
| `build_profile(log, my_list, movies, now)` | The learning step (§7): one interest score per movie, plus genre interest |
| `profile_for(username, movies)` | `build_profile()` from a user's own files |
| `summary(log)` | Counts per action, for tiles and the admin |

### 4.7 `collaborative.py`: item-item collaborative filtering

| Name | Purpose |
|---|---|
| `load_movielens(movies)` | MovieLens ratings mapped onto catalogue rows, centred per viewer |
| `app_user_ratings(profiles, movies)` | Each account's interest scores as one more viewer (best movie +2, disliked −2) |
| `build_model(n_movies, *parts)` | Sparse item × viewer matrix → `BehaviourModel` |
| `BehaviourModel.scores(rows, weights)` | Cosine of every movie to the weighted taste in viewer space; `None` if no movie is known |
| `BehaviourModel.info` | Viewers, ratings and movies covered |

### 4.8 `patterns.py`: the pattern and the blended recommender

| Function | Purpose |
|---|---|
| `analyze(profile, movies, genre_interest)` | The pattern dict: weighted genres/decades/style, repeated people/themes, series, recent trend, own-rating genres, sentences |
| `recommend(engine, movies, profile, pattern, top_n, exclude_ids, behaviour)` | Blended scores and reasons (§7) |
| `because_you_liked(engine, movies, profile, seeds, per_seed, exclude_ids)` | Rows of the films most similar (TF-IDF) to the user's highest-scoring favourites; no repeats, nothing already touched |
| `finish_series()` | Next unseen films of started series |
| `site_summary(profiles, movies)` | Genres liked across all users (admin dashboard) |
| `profile_with_details()` | The learned movies as a table |

### 4.9 `posters.py`

`poster_url(movie, size)` builds the TMDB image link (never touching the
network); `python -m cinematch.posters` refreshes stale poster paths with a
free TMDB key.

### 4.10 `ui/` package: the Streamlit pages

#### `ui/state.py`: caching and the signed-in user
`catalogue_signature()`, `load_movies()` (`st.cache_data`), `load_recommender()`
and `load_movielens()` (`st.cache_resource`), `behaviour_model(movies, exclude)`
(rebuilt on each call so it includes the latest behaviour), `current_user()`,
`username()`, `is_admin()`, `get_data()`.

#### `ui/components.py`: shared widgets
- `poster()`: HTML poster tile over generated genre art.
- `movie_grid()`: clickable tiles; **records a `view`** for the clicked movie.
- `movie_dialog()`, `list_controls()`: the details pop-up and its My List controls.
- `movie_card()`, `movie_table()`, `metric_tiles()`, `row_header()`,
  `page_header()`, `genre_chips()`, `flash()`/`show_flash()`.

#### The pages
| Page | Module | Notes |
|---|---|---|
| Sign in / Create account | `login.py` | Sign-up always makes a `user`; an *Account Created!* pop-up (`st.dialog`, not dismissible) offers *Proceed to CineMatch* (signs in) or *Back to sign in* (sign-in tab, username filled in) |
| Home | `dashboard.py` | *Picked for you* (learned pattern), *Recently viewed*, *Continue watching*, Trending, Highest rated, genre pills (recorded) |
| Discover | `recommend.py` | Search and genre filter (recorded), paged wall sorted A–Z by `_alphabetical()` (leading quotes/brackets ignored; when searching, titles that start with the search come first), *Find movies like this* (recorded) |
| Recommended | `watch_pattern.py` | Top picks for you (with reasons), *Because you liked …* rows (`patterns.because_you_liked()`), Finish the series; folded: what CineMatch has noticed, and the learned movies (forget / clear) + recent activity. Shows popular films until something is learned |
| My List | `mylist.py` | Tiles, counts, filter, sort, bulk editor, download, *Recommended for you* |
| Admin dashboard | `admin.py` | Accounts, users with a pattern, actions recorded, genres liked, latest log |
| Manage users | `admin.py` | Create, change role, reset password, delete |
| User patterns | `admin.py` | `watch_pattern.show_for()` for any user |
| Activity log | `admin.py` | Filter by user and action, download |

---

## 5. The CRUD operations (My List)

CRUD is done from the user's side: the catalogue cannot be edited, but each
user fully controls their own list in `data/users/<name>/my_list.csv`.

| CRUD | Data layer (`file_manager.py`) | Where in the UI |
|---|---|---|
| **C**reate | `add_to_list(movie_id, title, status, my_rating, notes)` | *+ Add to My List* in the movie pop-up |
| **R**ead | `read_my_list()`, `get_list_entry(movie_id)` | My List page, the pop-up, sidebar count, Home's *Continue watching* |
| **U**pdate | `update_list_entry(movie_id, **changes)`, `replace_my_list(df)` | *Save changes* in the pop-up; the bulk editor on My List |
| **D**elete | `remove_from_list(movie_id)` | *Remove from My List* in the pop-up; the editor's *Remove* box |

- **Create**: refuse duplicates, validate status and rating (blank or 1–10),
  stamp `added_on` = `updated_on`, write atomically.
- **Read**: fixed column order, malformed ids dropped, blank rating → NaN.
- **Update**: only the fields passed are changed; `updated_on` is stamped.
  The bulk version validates **every** row before writing anything and counts
  only real changes (blank ratings are normalised, because `NaN != NaN`).
- **Delete**: write back every other row.

My List is also the strongest input to the learned pattern (§7): adding a movie,
its status and its rating all change its interest score.

---

## 6. The machine-learning pipeline

CineMatch uses two complementary models and blends them.

| | Content-based (`recommender.py`) | Collaborative (`collaborative.py`) |
|---|---|---|
| Learns from | What each film is about (genres, keywords, cast, director, series, plot) | Behaviour: which films the same people like |
| Algorithm | TF-IDF vectors + cosine similarity | Item × viewer matrix + item-item cosine similarity |
| Training data | 9,097 catalogue movies | 91,793 MovieLens ratings by 671 viewers + every CineMatch account |
| Strength | Works for any movie, explains itself | Finds links content can't see (*Pulp Fiction* → *Fargo*) |

### 6.1 Content: the "content soup"

`build_soup_for_row()` cleans seven fields and repeats each by its weight.

```
animation comedy family                  (x3)
jealousy toy boy friendship boynextdoor  (x3)
toystorycollection                       (x3)
tomhanks timallen donrickles             (x2)
johnlasseter                             (x2)
led by woody andys toys live happily     (x1)
```

### 6.2 TF-IDF vectorisation

| Setting | Value | Reason |
|---|---|---|
| `stop_words` | `"english"` | Drop "the", "and", "of" |
| `max_features` | 30,000 | Cap the vocabulary |
| `min_df` | 2 (1 under 50 rows) | A word used by one movie cannot link two movies |
| `sublinear_tf` | True | 1 + log(tf), so repetition does not dominate |

### 6.3 Cosine similarity and explanations

`cosine_similarity(matrix[index], matrix)[0]`: one row per request (the full
9,097 × 9,097 matrix would need ~650 MB). `shared_terms()` multiplies two rows
element-wise; the largest products are the tokens both movies rely on, and
`readable()` maps them back to display names.

A weighted **taste profile** averages several rows:

```
profile = (matrix[rows].T @ weights) / sum(weights)
scores  = cosine_similarity(profile, matrix)
```

### 6.4 Collaborative filtering

1. **Load**: `ratings_small.csv` (userId, movieId, rating) is joined to
   `links_small.csv` (movieId → tmdbId) and then to catalogue rows; ratings for
   movies outside the catalogue are dropped (91,793 of 100,004 remain).
2. **Centre**: `value = rating − that viewer's mean rating`, so a harsh critic's
   3.5 and a generous viewer's 4.5 can both mean "liked it".
3. **Add CineMatch accounts**: each account other than the current user becomes
   one more column. Its values are its learned interest scores scaled so its
   strongest movie is +2; a disliked movie is −2.
4. **Normalise**: each movie's row in the sparse matrix is scaled to length 1.
5. **Score**: `taste = Σ weight × row(movie)` over the user's liked movies;
   `scores = matrix @ taste / ‖taste‖`. Movies with no ratings score 0, so the
   content model still covers them.

The MovieLens half is cached; the accounts half is rebuilt on every request,
so the model keeps learning as people use the site.

### 6.5 Caching and invalidation

`data_fingerprint()` is an MD5 of all soups. `try_load_cache()` accepts the
pickle only if the fingerprint and the row count match, and treats a corrupt or
incompatible pickle as a cache miss.

---

## 7. Learning the watch pattern from behaviour

Nobody uploads or fills in anything: the pattern is learned from how each
person (user or admin) uses CineMatch.

### 7.1 Interest scores (`activity.build_profile`)

For every movie the user has touched:

```
recency      = 0.5 ** (age_in_days / HALF_LIFE_DAYS)
event_score  = min( Σ EVENT_WEIGHTS[action] × recency , MAX_EVENT_SCORE )
list_score   = LIST_STATUS_WEIGHTS[status] × (my_rating or 7) / 7     (if on My List)
score        = event_score + list_score          (0 if my_rating < LIKED_RATING)
```

| Signal | Weight | Why |
|---|---|---|
| Opened the details (`view`) | 1.0 | Passing curiosity |
| Searched the exact title (`search`) | 2.0 | Deliberate |
| Typed search, e.g. "batman" (`search_text`) | 0.5 to each of the top 3 hits | Intent, but not a specific film |
| *Find movies like this* (`pick`) | 3.0 | "I want more of this" |
| On My List | 4–5 × rating / 7 | The strongest, explicit signal |
| Rated below 6 | score 0, `disliked` | Kept out of the pattern, never recommended |
| Browsed a genre (`genre`) | 1.0 to that genre | Feeds genre interest directly |

The profile is ordered by when each movie was last touched, which is how
"lately" is detected.

### 7.2 The pattern (`patterns.analyze`)

Genres, decades, typical runtime, rating level, popularity percentile and
language are **weighted by interest score** (browsed genres add to the genre
weights). Directors, actors, themes and genre pairs are counted in **whole
movies** and must appear at least twice. Series are collections where some, but
not all, films have been touched. The *recent trend* compares the genres of the
most recently touched third of the movies with the overall weighted shares; a
genre is "rising" when it is in at least half of the recent films and at least
15 points above its overall share. Every finding becomes a sentence.

### 7.3 Blended recommendations (`patterns.recommend`)

| Part | Weight | Source |
|---|---|---|
| Content similarity | 35% | TF-IDF taste profile of the liked movies, weighted by score |
| People like you | 20% | Collaborative scores (§6.4), normalised to 0–1 |
| Genre fit | 15% | Share of the user's interest in the movie's genres |
| Era fit | 10% | Share of the user's interest in the movie's decade |
| Rating fit | 5% | Closeness to the user's typical rating |
| People / series | 15% | Next film of a started series or a recurring director (1.0), recurring actor (0.6) |

If the collaborative model knows none of the user's movies, its 20% moves to
content similarity. Everything the user has already touched is excluded. Each
pick carries up to three reasons, most specific first, for example *"Continues
The Dark Knight Collection (you looked at 2 of 3)"*, *"Directed by Christopher
Nolan (6 of your movies)"*, *"Viewers who like your movies like this one"*.

### 7.4 Privacy and control

The activity lives only in `data/users/<name>/activity.csv` on this computer.
On *Recommended → Manage what CineMatch learned* the user can **forget** chosen
movies (rewrites the log without them) or **clear** all activity. Deleting an
account deletes the folder. Admins can view any user's learned pattern on
*User patterns*.

---

## 8. Files on disk

| File | Written by | Mode |
|---|---|---|
| `movies data/movies_metadata.csv`, `credits.csv`, `keywords.csv` | nobody | read only |
| `movies data/ratings_small.csv`, `links_small.csv` | nobody | read only |
| `data/catalogue.csv` | `build_catalogue()` | create, then read |
| `data/cinematch.db` | `accounts.py` | SQLite (users table) |
| `data/activity_log.csv` | `accounts.log()` | append |
| `data/users/<name>/my_list.csv` | the My List functions | read / write |
| `data/users/<name>/activity.csv` | `activity.record()` | append; rewritten by *Forget*, deleted by *Clear* |
| `data/cache/tfidf_cache.pkl` | `save_cache()` | read / write |
| `data/cache/poster_paths.json` | `refresh_posters()` (optional) | read / write |

The three source files total ~230 MB, with cast and crew stored as long
list-shaped strings, so CineMatch builds a ~5.9 MB catalogue of the 9,097
movies worth recommending and the 18 columns it uses.

---

## 9. Safety rules enforced by code

1. **The source dataset is read-only.** `_guard_source()` refuses any write
   inside `movies data/`; every write goes through a function that calls it.
2. **Every rewrite is atomic.** Write to `*.tmp`, then `os.replace()`.
3. **Validate before writing.** Duplicate movies, unknown statuses and ratings
   outside 1–10 raise `FileManagerError` before the file is touched.
4. **Accounts are protected by the database itself**: role CHECK, UNIQUE
   username, last-admin triggers, hashed + salted passwords, `?` placeholders.
5. **Paths can't escape.** Usernames are `[a-z0-9_]{3,20}`, and deleting a
   user folder checks it sits directly inside `data/users/`.
6. **Tests do not damage user data.** The suite saves and restores
   `data/my_list.csv`; accounts and activity tests run in a scratch folder.

---

## 10. Testing

`python test_cinematch.py`: plain Python, no test framework, **162 checks**.

| Group | Checks | Highlights |
|---|---|---|
| 1. Preprocessing | 15 | Cleaning, name squashing, missing values, soup weights |
| 2. Catalogue | 15 | Name parsing, directors, filters, unique ids |
| 3. Protection | 2 | Writing into `movies data/` is refused |
| 4. Recommendations | 21 | Scores, order, Batman for The Dark Knight, taste profile |
| 5. Caching | 6 | Written, reused, rejected after a data change |
| 6. My List CRUD | 23 | Every action changes the file; bad input refused |
| 7. Messy data | 4 | A movie with no text returns nothing instead of crashing |
| 8. Accounts | 30 | Hashing, roles, last-admin rules, raw SQL refusals, injection, admin password from a deployment secret |
| 9. Learning from activity | 16 | pick > search > view, half-life, click cap, typed search, genres, dislikes, forget / clear |
| 10. Collaborative filtering | 8 | Co-liked movies close, MovieLens centred, Toy Story → family films, accounts as viewers, works without the MovieLens files |
| 11. Patterns | 22 | Nolan, Dark Knight series, rising sci-fi, nothing touched recommended, sequel first, behaviour used, *Because you liked* rows without repeats |

Every page was also rendered with Streamlit's `AppTest`, both empty and with a
populated scratch user, with no exceptions.

---

## 11. Measured results

### Dataset and models

| | Value |
|---|---|
| Source movies → catalogue | 45,466 → 9,097 |
| TF-IDF matrix | 9,097 × 29,059, 0.137% dense, fit < 1 s |
| MovieLens ratings on catalogue movies | 91,793 by 671 viewers |
| Movies with behaviour data | 5,759 of 9,097 |
| Collaborative model build | ~0.1 s |

### Content vs. behaviour: nearest films

| Movie | Content (TF-IDF) | Behaviour (collaborative) |
|---|---|---|
| The Dark Knight | The Dark Knight Rises 42.2, Batman Begins 38.8, Batman: Under the Red Hood 28.6 | Batman Begins 45.9, Iron Man 42.8, The Departed 37.3, The Prestige 37.3 |
| Toy Story | Toy Story 2 41.1, Toy Story 3 34.2, Toy Story of Terror! 23.1 | Toy Story 2 39.8, Toy Story 3 26.8, A Bug's Life 23.5, The Incredibles 20.3 |
| Pulp Fiction | — | Fargo 44.9, The Usual Suspects 40.0, Se7en 38.0, GoodFellas 37.2 |

The behaviour model finds what people actually watch together (*The Departed*
next to *The Dark Knight*), which the content model can't see; the content
model covers films nobody in MovieLens rated. The blend uses both.

### A learned pattern

A test viewer who opened ten films (five by Christopher Nolan, then *Blade
Runner*, *The Matrix*, *Arrival*), picked three and listed the sci-fi ones gets:
*Christopher Nolan* as the recurring director, the unfinished *Dark Knight
Collection*, *"Lately you have been looking at more Science Fiction"*, and *The
Dark Knight Rises* as the first recommendation.

---

## 12. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "Source file not found" | The dataset must be in `movies data/` next to `app.py`, with the CSV names unchanged |
| No MovieLens viewers in the "same taste" caption | `ratings_small.csv` / `links_small.csv` are not in `movies data/` (they are not in the repository); the model learns from CineMatch accounts only |
| Recommended shows only popular films | Nothing learned yet: search, open movies, use *Find movies like this* or add to My List |
| Recommendations feel stuck on old tastes | Old actions fade after 30 days; or use *Forget* / *Clear activity* |
| Most posters show drawn art | The 2017 poster links are stale. Run `python -m cinematch.posters` with a TMDB key |
| Forgot the admin password | Delete `data/cinematch.db`; it is recreated with `admin` / `admin123` (all accounts are lost) |
| The catalogue seems out of date | Delete `data/catalogue.csv`; it is rebuilt on the next start |
| Model rebuilt on every start | The cache is rejected if the catalogue changed, or `data/cache/` is not writable |
