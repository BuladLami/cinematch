"""
ui/components.py -- Small reusable pieces of interface.

Keeping the styling and the repeated layouts (metric tiles, movie cards, the
details pop-up) here means the page files stay short and readable.

Posters come from TMDB (see posters.py).  Underneath every image sits artwork
the browser draws itself: a colour gradient picked from the genre plus the
title.  Wherever an image fails to load, that art shows instead, so the site
still looks deliberate with a stale poster path or no internet at all.
"""

from __future__ import annotations

import hashlib
import html

import pandas as pd
import streamlit as st

from .. import activity, config, file_manager, posters
from . import state

# ---------------------------------------------------------------------------
# PALETTE
# ---------------------------------------------------------------------------
# One place for the colours, so a tweak here changes the whole site.
ACCENT = "#7C5CFF"      # violet  -- match scores, primary actions
GOLD = "#FFC247"        # amber   -- star ratings
MUTED = "#97A0B5"

CSS = """
<style>
  /* ----------------------------------------------------------------- base */
  /* The top padding clears Streamlit's floating toolbar, which would
     otherwise sit on top of the page heading. */
  .block-container { padding-top: 3.4rem; padding-bottom: 4rem; max-width: 1280px; }

  /* A soft spotlight behind the page, like light bleeding off a screen. */
  .stApp {
      background:
        radial-gradient(1100px 520px at 12% -12%, rgba(124,92,255,.16), transparent 60%),
        radial-gradient(900px 480px at 92% 0%,   rgba(255,93,143,.10), transparent 55%),
        #0B0E14;
  }

  h1, h2, h3, h4 { letter-spacing: -.015em; }

  /* -------------------------------------------------------------- sidebar */
  section[data-testid="stSidebar"] {
      background: #0D111A;
      border-right: 1px solid rgba(255,255,255,.07);
  }
  .cm-brand { display: flex; align-items: center; gap: .6rem; margin: .2rem 0 .1rem; }
  .cm-brand-mark {
      width: 34px; height: 34px; border-radius: 9px; flex: none;
      background: linear-gradient(135deg, #7C5CFF, #FF5D8F);
      display: grid; place-items: center;
      font-size: 1.05rem; box-shadow: 0 6px 18px rgba(124,92,255,.45);
  }
  .cm-brand-name { font-size: 1.35rem; font-weight: 800; letter-spacing: -.03em; }
  .cm-brand-name span { color: #7C5CFF; }

  /* Turn the sidebar radio into a proper nav rail. */
  section[data-testid="stSidebar"] div[role="radiogroup"] { gap: .15rem; }
  section[data-testid="stSidebar"] div[role="radiogroup"] label {
      padding: .5rem .7rem; border-radius: 9px; width: 100%;
      transition: background .15s ease;
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
      background: rgba(255,255,255,.05);
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
      background: rgba(124,92,255,.16);
      box-shadow: inset 2px 0 0 #7C5CFF;
  }

  /* ----------------------------------------------------------- page header */
  /* Streamlit sizes paragraphs with [data-testid="stMarkdownContainer"] p,
     which outranks a plain class selector.  Naming the container here makes
     these rules more specific, so the headings keep the size they are given. */
  .cm-head { margin: 0 0 1.4rem; }
  [data-testid="stMarkdownContainer"] .cm-eyebrow {
      font-size: .72rem; font-weight: 700; letter-spacing: .13em;
      text-transform: uppercase; color: #7C5CFF; margin: 0 0 .35rem;
  }
  [data-testid="stMarkdownContainer"] .cm-title {
      font-size: 2.1rem; font-weight: 800; margin: 0; letter-spacing: -.03em;
      line-height: 1.18;
  }
  [data-testid="stMarkdownContainer"] .cm-sub {
      color: #97A0B5; margin: .3rem 0 0; font-size: .97rem; max-width: 62ch;
  }

  /* ------------------------------------------------------------ stat tiles */
  .cm-tile {
      background: linear-gradient(180deg, rgba(255,255,255,.055), rgba(255,255,255,.02));
      border: 1px solid rgba(255,255,255,.08);
      border-radius: 14px; padding: .95rem 1.05rem; height: 100%;
  }
  .cm-tile .label { font-size: .7rem; text-transform: uppercase;
                    letter-spacing: .09em; color: #97A0B5; font-weight: 600; }
  .cm-tile .value { font-size: 1.75rem; font-weight: 800; line-height: 1.3;
                    letter-spacing: -.02em; }
  .cm-tile .note  { font-size: .76rem; color: #97A0B5; }

  /* ---------------------------------------------------------------- poster */
  /* Each poster is drawn from three colours picked by genre (--c1 deep,
     --c2 mid, --c3 accent) plus one geometric motif chosen from the title. */
  .cm-poster {
      position: relative; border-radius: 12px; overflow: hidden;
      aspect-ratio: 2 / 3; display: grid; place-items: center;
      background: linear-gradient(165deg, var(--c2), var(--c1) 72%);
      box-shadow: 0 10px 26px rgba(0,0,0,.45);
  }
  /* A fetched poster image, when there is one, covers the drawn art. */
  .cm-poster-img {
      position: absolute; inset: 0; z-index: 6;
      background-size: cover; background-position: center;
  }

  /* ------ the motif layer: the shape that gives each poster its character */
  .cm-art { position: absolute; inset: 0; z-index: 1; }
  .cm-art-rays {
      background: repeating-conic-gradient(from 200deg at 50% 116%,
                  var(--c3) 0 2deg, transparent 2deg 9deg);
      opacity: .38;
  }
  .cm-art-orb {
      background: radial-gradient(circle at 50% 34%, var(--c3) 0 15%, transparent 16%),
                  radial-gradient(circle at 50% 34%, rgba(255,255,255,.16) 0 23%,
                                  transparent 24%);
      opacity: .85;
  }
  .cm-art-horizon {
      background: linear-gradient(180deg, transparent 51.6%, var(--c3) 52%,
                                  transparent 52.6%),
                  radial-gradient(ellipse 70% 30% at 50% 52%, var(--c3), transparent 70%);
      opacity: .55;
  }
  .cm-art-bars {
      background: repeating-linear-gradient(90deg, var(--c3) 0 2px, transparent 2px 13px);
      opacity: .17;
  }
  .cm-art-arc {
      background: radial-gradient(circle at 50% 94%, transparent 0 45%,
                  var(--c3) 45.5% 47.5%, transparent 48%),
                  radial-gradient(circle at 50% 94%, transparent 0 62%,
                  var(--c3) 62.5% 63.5%, transparent 64%);
      opacity: .5;
  }
  .cm-art-grid {
      background: repeating-linear-gradient(0deg, transparent 0 11px, var(--c3) 11px 12px),
                  repeating-linear-gradient(90deg, transparent 0 11px, var(--c3) 11px 12px);
      opacity: .13;
  }

  /* A top light and a bottom scrim: depth, and readable type over the art. */
  .cm-poster::after {
      content: ""; position: absolute; inset: 0; z-index: 3; pointer-events: none;
      background: linear-gradient(200deg, rgba(255,255,255,.14), transparent 38%),
                  linear-gradient(0deg, rgba(0,0,0,.78), rgba(0,0,0,.15) 42%,
                                  transparent 62%);
  }

  /* ------ the type: the title is the artwork, the way a real poster works */
  .cm-poster-text {
      position: absolute; left: 0; right: 0; bottom: 0; z-index: 4;
      padding: .5rem .5rem .6rem; text-align: center;
  }
  .cm-ptitle {
      font-size: .8rem; font-weight: 800; text-transform: uppercase;
      letter-spacing: .035em; line-height: 1.14; color: #fff;
      text-shadow: 0 2px 10px rgba(0,0,0,.8);
      display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;
      overflow: hidden;
  }
  .cm-pcredit {
      font-size: .58rem; letter-spacing: .09em; text-transform: uppercase;
      color: rgba(255,255,255,.7); margin-top: .28rem;
      text-shadow: 0 1px 6px rgba(0,0,0,.8);
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .cm-poster-initials {
      font-size: 1.9rem; font-weight: 800; letter-spacing: .06em; z-index: 4;
      color: rgba(255,255,255,.92); text-shadow: 0 2px 12px rgba(0,0,0,.55);
  }

  /* ------ corner chips */
  .cm-poster .cm-rank {
      position: absolute; top: .45rem; left: .45rem; z-index: 7;
      background: rgba(11,14,20,.82);
      border: 1px solid rgba(255,255,255,.16);
      border-radius: 8px; padding: .1rem .45rem;
      font-size: .78rem; font-weight: 800;
  }
  /* The match score sits top-right so it never fights with the title. */
  .cm-poster .cm-badge {
      position: absolute; top: .45rem; right: .45rem; z-index: 7;
      background: rgba(124,92,255,.94);
      border-radius: 999px; padding: .12rem .45rem;
      font-size: .68rem; font-weight: 700; color: #fff;
      box-shadow: 0 3px 10px rgba(0,0,0,.4);
  }

  /* ------------------------------------------------------------ grid card */
  .cm-gcard { margin-bottom: 1.1rem; }
  .cm-gcard .cm-poster { transition: transform .18s ease, box-shadow .18s ease; }
  .cm-gcard:hover .cm-poster, div[class*="st-key-cmclick_"]:hover .cm-poster {
      transform: translateY(-4px);
      box-shadow: 0 16px 34px rgba(124,92,255,.32);
  }

  /* ------------------------------------------------------ clickable tiles */
  /* Streamlit cannot make a <div> clickable, so each tile carries a real
     button stretched invisibly over the whole card (see _clickable_card). */
  div[class*="st-key-cmclick_"] { position: relative; }
  div[class*="st-key-cmclick_"] [data-testid="stElementContainer"]:has(.stButton) {
      position: absolute; inset: 0; z-index: 10; margin: 0;
  }
  div[class*="st-key-cmclick_"] .stButton,
  div[class*="st-key-cmclick_"] .stButton > button {
      width: 100%; height: 100%;
  }
  div[class*="st-key-cmclick_"] .stButton > button {
      opacity: 0; cursor: pointer;
  }

  /* ---------------------------------------------------------- detail popup */
  .cm-dialog-title { font-size: 1.6rem; font-weight: 800; letter-spacing: -.02em;
                     line-height: 1.2; margin: 0 0 .35rem; }
  .cm-dialog .meta { color: #97A0B5; font-size: .88rem; margin-bottom: .6rem; }
  .cm-dialog .overview { font-size: .93rem; line-height: 1.65; color: #C9CFDD;
                         margin: .7rem 0 .9rem; }
  .cm-facts { display: grid; grid-template-columns: max-content 1fr;
              gap: .35rem .9rem; font-size: .85rem; }
  .cm-facts dt { color: #97A0B5; font-weight: 600; }
  .cm-facts dd { margin: 0; color: #DDE2EC; }
  .cm-gtitle {
      font-size: .92rem; font-weight: 700; line-height: 1.3; margin: .55rem 0 .15rem;
      display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
      overflow: hidden;
  }
  .cm-gmeta { font-size: .78rem; color: #97A0B5; display: flex; gap: .45rem;
              align-items: center; flex-wrap: wrap; }
  .cm-star { color: #FFC247; font-weight: 700; }
  .cm-why  { font-size: .72rem; color: #97A0B5; margin-top: .3rem; line-height: 1.45; }

  /* ----------------------------------------------------------- detail card */
  .cm-card {
      display: grid; grid-template-columns: 132px 1fr; gap: 1.15rem;
      background: linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.018));
      border: 1px solid rgba(255,255,255,.09);
      border-radius: 16px; padding: 1.15rem; margin-bottom: .9rem;
  }
  .cm-card.compact { grid-template-columns: 84px 1fr; padding: .9rem; }
  [data-testid="stMarkdownContainer"] .cm-card h4 {
      margin: 0 0 .3rem; padding: 0; font-size: 1.38rem; font-weight: 800;
      letter-spacing: -.02em;
  }
  .cm-card .meta { color: #97A0B5; font-size: .85rem; margin-bottom: .6rem; }
  .cm-card .overview { font-size: .91rem; line-height: 1.62; color: #C9CFDD;
                       margin-top: .55rem; }
  .cm-tagline { font-style: italic; color: #97A0B5; font-size: .88rem;
                margin-bottom: .45rem; }

  /* ----------------------------------------------------------------- chips */
  .cm-chip {
      display: inline-block; padding: .14rem .6rem; margin: 0 .3rem .3rem 0;
      border-radius: 999px; background: rgba(124,92,255,.14);
      border: 1px solid rgba(124,92,255,.32);
      font-size: .73rem; font-weight: 600; color: #C3B6FF;
  }
  .cm-score { font-weight: 800; color: #7C5CFF; font-size: .95rem; }

  /* ----------------------------------------------------- section headings */
  .cm-row-head {
      display: flex; align-items: baseline; gap: .7rem;
      margin: .4rem 0 .9rem; padding-bottom: .5rem;
      border-bottom: 1px solid rgba(255,255,255,.08);
  }
  [data-testid="stMarkdownContainer"] .cm-row-head h3 {
      margin: 0; font-size: 1.15rem; font-weight: 700; padding: 0;
  }
  .cm-row-head .hint { color: #97A0B5; font-size: .82rem; }

  /* --------------------------------------------------- streamlit widgets */
  .stButton > button, .stDownloadButton > button {
      border-radius: 10px; font-weight: 600; border: 1px solid rgba(255,255,255,.12);
      transition: transform .12s ease, box-shadow .12s ease;
  }
  .stButton > button:hover, .stDownloadButton > button:hover {
      transform: translateY(-1px); box-shadow: 0 8px 20px rgba(0,0,0,.4);
  }
  .stButton > button[kind="primary"] {
      background: linear-gradient(135deg, #7C5CFF, #9B7BFF);
      border: none; box-shadow: 0 8px 22px rgba(124,92,255,.35);
  }
  div[data-testid="stTextInput"] input, div[data-baseweb="select"] > div {
      border-radius: 10px !important;
  }
  div[data-testid="stExpander"] {
      border: 1px solid rgba(255,255,255,.08); border-radius: 12px;
      background: rgba(255,255,255,.02);
  }
  hr { border-color: rgba(255,255,255,.08); }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# POSTER ARTWORK
# ---------------------------------------------------------------------------
# The CSV ships no images, so each poster is designed on the fly from what the
# data does contain: the genre picks the colours, the title picks the motif and
# is set as the artwork itself.  hashlib gives the same number for the same
# title every time, so a movie's poster never changes between page loads.

# (deep background, mid background, accent) per genre.
GENRE_PALETTES = {
    "Action":          ("#2B0A0A", "#8E1B14", "#FF8A3D"),
    "Adventure":       ("#062018", "#1F7A4C", "#FFD166"),
    "Animation":       ("#1A0B2E", "#6D28D9", "#FF6FB5"),
    "Comedy":          ("#2B1A00", "#C77F0A", "#FFE066"),
    "Crime":           ("#0B0E14", "#25313F", "#C0392B"),
    "Documentary":     ("#14181F", "#3D5A6C", "#A8B8C4"),
    "Drama":           ("#131A22", "#3E5C76", "#C9A227"),
    "Family":          ("#062A2E", "#12857E", "#FFD166"),
    "Fantasy":         ("#1A0E2E", "#5B3E86", "#F3C969"),
    "History":         ("#241A12", "#7A5C3E", "#D9C3A5"),
    "Horror":          ("#080404", "#3A0B0B", "#B01515"),
    "Music":           ("#1B0B2E", "#A02A6E", "#FDCB6E"),
    "Mystery":         ("#0C1420", "#2E4057", "#7FB3D5"),
    "Romance":         ("#2A0A1B", "#A1174E", "#FF8FA3"),
    "Science Fiction": ("#030A1A", "#0B3D91", "#22D3EE"),
    "TV Movie":        ("#16181D", "#3F4651", "#9CA3AF"),
    "Thriller":        ("#0A0F0D", "#1B3236", "#E63946"),
    "War":             ("#14170F", "#434A1D", "#D4A017"),
    "Western":         ("#2A1A0A", "#8C4527", "#E9C46A"),
}
DEFAULT_PALETTE = ("#131722", "#37415A", "#88C0D0")

# Six abstract shapes, enough that a wall of posters never looks repetitive.
MOTIFS = ["rays", "orb", "horizon", "bars", "arc", "grid"]


def _title_hash(title: str) -> int:
    digest = hashlib.md5(str(title).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _palette(movie: pd.Series) -> tuple[str, str, str]:
    """The colours for a movie, taken from its first recognised genre."""
    raw = str(movie.get("genre_list") or movie.get("genres") or "")
    for genre in (g.strip() for g in raw.split(",")):
        if genre in GENRE_PALETTES:
            return GENRE_PALETTES[genre]
    return DEFAULT_PALETTE


def _initials(title: str) -> str:
    """Turn a title into two or three display letters.

    "The Dark Knight" -> "DK".  Small words are skipped so the letters mean
    something; a one-word title falls back to its first two letters.
    """
    skip = {"the", "a", "an", "of", "and", "in", "on", "to", "le", "la", "el"}
    words = [w for w in str(title).split() if w.strip()]
    letters = [w[0] for w in words if w.lower() not in skip and w[0].isalnum()]
    if not letters:
        letters = [w[0] for w in words if w[0].isalnum()]
    if not letters:
        return "?"
    if len(letters) == 1:
        first = next((w for w in words if w[0].isalnum()), "?")
        return first[:2].upper()
    return "".join(letters[:3]).upper()


def poster(movie: pd.Series, rank: int | None = None, badge: str | None = None,
           variant: str = "full", size: str = posters.GRID_SIZE) -> str:
    """The HTML for one poster tile, designed from the movie's own data.

    variant "full" sets the title as the artwork, the way a real poster does.
    variant "mini" is for tiles too small to read, and shows initials instead.

    If a TMDB poster happens to have been fetched (see posters.py -- entirely
    optional, and off unless an API key is present) it covers the drawn art.
    """
    title = str(movie.get("title", "Untitled"))
    deep, mid, accent = _palette(movie)
    digest = _title_hash(title)
    motif = MOTIFS[digest % len(MOTIFS)]

    art_layer = f'<div class="cm-art cm-art-{motif}"></div>'

    if variant == "mini":
        text_layer = (f'<span class="cm-poster-initials">'
                      f'{html.escape(_initials(title))}</span>')
    else:
        year = int(movie.get("year", 0) or 0)
        director = _directors(movie)
        credit = " · ".join(bit for bit in [str(year) if year else "", director] if bit)
        credit_html = (f'<div class="cm-pcredit">{html.escape(credit)}</div>'
                       if credit else "")
        text_layer = (f'<div class="cm-poster-text">'
                      f'<div class="cm-ptitle">{html.escape(title)}</div>'
                      f'{credit_html}</div>')

    # The real poster sits on top of the drawn art as a background image.  An
    # image that fails to load (old dataset paths often do) simply draws
    # nothing, so the art and title underneath show through instead of a
    # broken-image icon.
    image_url = posters.poster_url(movie, size=size)
    image_layer = ""
    if image_url:
        image_layer = (f'<div class="cm-poster-img" role="img" '
                       f'aria-label="{html.escape(title)} poster" '
                       f'style="background-image:url(\'{html.escape(image_url)}\')"></div>')

    rank_html = f'<span class="cm-rank">{rank}</span>' if rank else ""
    badge_html = f'<span class="cm-badge">{html.escape(badge)}</span>' if badge else ""

    style = f"--c1:{deep};--c2:{mid};--c3:{accent}"
    return (f'<div class="cm-poster" style="{style}">'
            f'{art_layer}{image_layer}{text_layer}{rank_html}{badge_html}</div>')


# ---------------------------------------------------------------------------
# MESSAGES
# ---------------------------------------------------------------------------
def flash(message: str, kind: str = "success") -> None:
    """Remember a message so it survives the st.rerun() that follows.

    st.rerun() restarts the script immediately, which wipes any st.success()
    written just before it.  Stashing the text in session_state and printing it
    on the next run is what makes the confirmation actually visible.
    """
    st.session_state["flash"] = (kind, message)


def show_flash() -> None:
    """Print and clear the message left by the previous run, if any."""
    stored = st.session_state.pop("flash", None)
    if not stored:
        return
    kind, message = stored
    {"success": st.success, "info": st.info, "warning": st.warning,
     "error": st.error}.get(kind, st.info)(message)


# ---------------------------------------------------------------------------
# LAYOUT PIECES
# ---------------------------------------------------------------------------
def page_header(title: str, subtitle: str, eyebrow: str | None = None) -> None:
    eyebrow_html = f'<p class="cm-eyebrow">{html.escape(eyebrow)}</p>' if eyebrow else ""
    st.markdown(
        f'<div class="cm-head">{eyebrow_html}'
        f'<p class="cm-title">{html.escape(title)}</p>'
        f'<p class="cm-sub">{html.escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def row_header(title: str, hint: str = "") -> None:
    """The heading that sits above a shelf of posters."""
    hint_html = f'<span class="hint">{html.escape(hint)}</span>' if hint else ""
    st.markdown(
        f'<div class="cm-row-head"><h3>{html.escape(title)}</h3>{hint_html}</div>',
        unsafe_allow_html=True,
    )


def metric_tiles(tiles: list[tuple[str, str, str]]) -> None:
    """Draw a row of statistic tiles: (label, value, small note)."""
    columns = st.columns(len(tiles))
    for column, (label, value, note) in zip(columns, tiles):
        with column:
            st.markdown(
                f'<div class="cm-tile"><div class="label">{html.escape(label)}</div>'
                f'<div class="value">{html.escape(str(value))}</div>'
                f'<div class="note">{html.escape(note)}</div></div>',
                unsafe_allow_html=True,
            )


def genre_chips(genre_list: str, limit: int | None = None) -> str:
    """Render "Action, Crime" as small rounded chips."""
    genres = [g.strip() for g in str(genre_list).split(",") if g.strip()]
    if limit:
        genres = genres[:limit]
    return "".join(f'<span class="cm-chip">{html.escape(g)}</span>' for g in genres)


def _directors(movie: pd.Series) -> str:
    """The director(s) as display text: "Joel Coen, Ethan Coen"."""
    listed = str(movie.get("director_list", "") or "").strip()
    return listed or str(movie.get("director", "") or "").replace("|", ", ").strip()


def _meta_line(movie: pd.Series) -> str:
    bits = []
    year = int(movie.get("year", 0) or 0)
    if year:
        bits.append(str(year))
    runtime = float(movie.get("runtime", 0) or 0)
    if runtime:
        bits.append(f"{int(runtime)} min")
    rating = float(movie.get("vote_average", 0) or 0)
    if rating:
        votes = int(movie.get("vote_count", 0) or 0)
        vote_text = f" ({votes:,} votes)" if votes else ""
        bits.append(f'<span class="cm-star">&#9733; {rating:.1f}</span>/10{vote_text}')
    director = _directors(movie)
    if director:
        bits.append(f"Dir. {html.escape(director)}")
    return " &nbsp;&middot;&nbsp; ".join(bits)


# ---------------------------------------------------------------------------
# MOVIE CARDS
# ---------------------------------------------------------------------------
def movie_card(movie: pd.Series, rank: int | None = None,
               similarity: float | None = None,
               why: list[str] | None = None,
               show_overview: bool = True) -> None:
    """One movie shown wide: poster on the left, everything else on the right."""
    title = html.escape(str(movie.get("title", "Untitled")))
    heading = f"{rank}. {title}" if rank else title

    score_html = ""
    if similarity is not None:
        score_html = f'<span class="cm-score">{similarity:.1f}% match</span>'

    tagline = str(movie.get("tagline", "")).strip()
    tagline_html = (f'<div class="cm-tagline">&ldquo;{html.escape(tagline)}&rdquo;</div>'
                    if tagline and show_overview else "")

    overview = str(movie.get("overview", "")).strip()
    if show_overview and overview:
        if len(overview) > 320:
            overview = overview[:320].rsplit(" ", 1)[0] + "..."
        overview_html = f'<div class="overview">{html.escape(overview)}</div>'
    else:
        overview_html = ""

    why_html = ""
    if why:
        why_html = ('<div class="cm-why" style="margin-top:.6rem">Shared with your pick: '
                    + " ".join(f'<span class="cm-chip">{html.escape(w)}</span>' for w in why)
                    + "</div>")

    # The compact card's poster is only 84px wide, too small to set a title in.
    css_class = "cm-card" if show_overview else "cm-card compact"
    variant = "full" if show_overview else "mini"
    st.markdown(
        f'<div class="{css_class}">'
        f'<div>{poster(movie, rank=rank, variant=variant)}</div>'
        f'<div><h4>{heading} &nbsp; {score_html}</h4>'
        f'<div class="meta">{_meta_line(movie)}</div>'
        f'{tagline_html}'
        f'{genre_chips(movie.get("genre_list", movie.get("genres", "")))}'
        f'{overview_html}{why_html}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _grid_card(movie: pd.Series, rank: int | None = None,
               similarity: float | None = None,
               why: list[str] | None = None,
               minimal: bool = False, why_prefix: str = "Shares: ") -> str:
    title = str(movie.get("title", "Untitled"))
    year = int(movie.get("year", 0) or 0)
    rating = float(movie.get("vote_average", 0) or 0)

    meta_bits = []
    if year:
        meta_bits.append(str(year))
    if rating:
        meta_bits.append(f'<span class="cm-star">&#9733; {rating:.1f}</span>')
    runtime = float(movie.get("runtime", 0) or 0)
    if runtime:
        meta_bits.append(f"{int(runtime)}m")

    badge = f"{similarity:.0f}% match" if similarity is not None else None

    why_html = ""
    if why:
        why_html = (f'<div class="cm-why">{html.escape(why_prefix)}'
                    f'{html.escape(", ".join(why[:3]))}</div>')

    # In "minimal" mode the chips are dropped so every tile in a browse wall
    # is exactly the same height and the rows line up.
    extras = "" if minimal else (
        genre_chips(movie.get("genre_list", movie.get("genres", "")), limit=2) + why_html
    )

    return (f'<div class="cm-gcard">{poster(movie, rank=rank, badge=badge)}'
            f'<div class="cm-gtitle">{html.escape(title)}</div>'
            f'<div class="cm-gmeta">{" &middot; ".join(meta_bits)}</div>'
            f'{extras}</div>')


def _clickable_card(card_html: str, key: str, title: str) -> bool:
    """Draw a grid card that works as one big button; True on the run it is clicked.

    The keyed container gets a "st-key-cmclick_..." class, which the CSS uses
    to stretch the (invisible) button over the whole card.
    """
    with st.container(key=f"cmclick_{key}"):
        st.markdown(card_html, unsafe_allow_html=True)
        return st.button(f"View details for {title}", key=f"{key}_btn")


def movie_grid(movies: pd.DataFrame, per_row: int = 5, ranked: bool = False,
               why_lookup: dict | None = None, key_prefix: str = "grid",
               minimal: bool = False, why_prefix: str = "Shares: ") -> int | None:
    """A shelf of clickable poster tiles, the way a streaming site lists its titles.

    why_lookup maps a row index to the words that movie shares with the chosen
    one, so the "why was this recommended" note can sit under each poster.
    Returns the index of the movie clicked on this run, or None.  Every click
    is also recorded as a "view": opening a movie is one of the signals the
    watch pattern is learned from (see activity.py).
    """
    clicked = None
    rows = list(movies.iterrows())
    for start in range(0, len(rows), per_row):
        chunk = rows[start:start + per_row]
        columns = st.columns(per_row)
        for offset, (column, (index, movie)) in enumerate(zip(columns, chunk)):
            similarity = None
            if "similarity_pct" in movie.index and pd.notna(movie["similarity_pct"]):
                similarity = float(movie["similarity_pct"])
            card = _grid_card(movie,
                              rank=(start + offset + 1) if ranked else None,
                              similarity=similarity,
                              why=(why_lookup or {}).get(index),
                              minimal=minimal, why_prefix=why_prefix)
            with column:
                if _clickable_card(card, f"{key_prefix}_{index}", str(movie.get("title", ""))):
                    clicked = index
                    activity.record_movie(state.username(), "view", movie)
    return clicked


# ---------------------------------------------------------------------------
# DETAIL POPUP  (with the My List controls)
# ---------------------------------------------------------------------------
RATING_CHOICES = ["Not rated"] + [str(n) for n in range(1, 11)]


def list_controls(movie: pd.Series, key: str) -> None:
    """Add, update or remove this movie on the user's list, right where it is shown.

    These are the four CRUD actions from the user's side: the add button
    CREATEs an entry, the current status is READ back from the file, the save
    button UPDATEs it and the remove button DELETEs it.  Each change reruns
    only the pop-up, so it stays open and redraws with the new state.
    """
    movie_id = int(movie["id"])
    user = state.username()
    entry = file_manager.get_list_entry(movie_id, username=user)

    notice = st.session_state.pop(f"{key}_notice", None)
    if notice:
        st.success(notice)

    def act(action, *args, **kwargs) -> None:
        try:
            st.session_state[f"{key}_notice"] = action(*args, **kwargs)
        except file_manager.FileManagerError as error:
            st.session_state[f"{key}_notice"] = None
            st.error(str(error))
            return
        st.rerun(scope="fragment")

    if entry is None:
        status_column, add_column = st.columns([1.3, 1], vertical_alignment="bottom")
        with status_column:
            status = st.selectbox("Add to My List as", config.LIST_STATUSES,
                                  key=f"{key}_new_status")
        with add_column:
            if st.button("+ Add to My List", type="primary", width="stretch",
                         key=f"{key}_add"):
                act(file_manager.add_to_list, movie_id, movie.get("title", ""), status,
                    username=user)
        return

    st.markdown(f"**On your list** &nbsp;·&nbsp; added {html.escape(str(entry['added_on']))}",
                unsafe_allow_html=True)
    status_column, rating_column = st.columns(2)
    with status_column:
        status = st.selectbox("Status", config.LIST_STATUSES,
                              index=config.LIST_STATUSES.index(entry["status"])
                              if entry["status"] in config.LIST_STATUSES else 0,
                              key=f"{key}_status")
    with rating_column:
        current = entry["my_rating"]
        rating = st.select_slider(
            "My rating", RATING_CHOICES,
            value=str(int(current)) if pd.notna(current) else "Not rated",
            key=f"{key}_rating",
        )
    notes = st.text_area("My notes", value=entry["notes"], height=80,
                         placeholder="Who recommended it, where to stream it...",
                         key=f"{key}_notes")

    save_column, remove_column = st.columns(2)
    with save_column:
        if st.button("Save changes", type="primary", width="stretch", key=f"{key}_save"):
            act(file_manager.update_list_entry, movie_id, username=user, status=status,
                notes=notes, my_rating=None if rating == "Not rated" else int(rating))
    with remove_column:
        if st.button("Remove from My List", width="stretch", key=f"{key}_remove"):
            act(file_manager.remove_from_list, movie_id, username=user)


# on_dismiss="rerun" refreshes the page behind the pop-up when it closes, so a
# list change made inside it shows up on My List and in "Recommended for you".
@st.dialog("Movie details", width="large", on_dismiss="rerun")
def movie_dialog(movie: pd.Series, action_label: str | None = None,
                 on_action=None) -> None:
    """A pop-up with everything known about one movie, plus the My List controls.

    If action_label is given, a button for it sits at the bottom and on_action()
    is called when it is pressed (Discover uses this to pick the movie).
    """
    poster_column, info_column = st.columns([1, 1.9], gap="medium")
    with poster_column:
        st.markdown(poster(movie, size=posters.LARGE_SIZE), unsafe_allow_html=True)

    tagline = str(movie.get("tagline", "") or "").strip()
    overview = str(movie.get("overview", "") or "").strip()
    facts = {
        "Original title": movie.get("original_title", "")
        if movie.get("original_title", "") != movie.get("title", "") else "",
        "Release date": movie.get("release_date", ""),
        "Director": _directors(movie),
        "Cast": movie.get("cast_list", ""),
        "Keywords": movie.get("keyword_list", ""),
        "Collection": movie.get("collection", ""),
        "Language": str(movie.get("original_language", "")).upper(),
    }
    facts_html = "".join(
        f"<dt>{html.escape(label)}</dt><dd>{html.escape(str(value))}</dd>"
        for label, value in facts.items()
        if pd.notna(value) and str(value).strip()
    )

    with info_column:
        st.markdown(
            f'<div class="cm-dialog">'
            f'<p class="cm-dialog-title">{html.escape(str(movie.get("title", "Untitled")))}</p>'
            f'<div class="meta">{_meta_line(movie)}</div>'
            + (f'<div class="cm-tagline">&ldquo;{html.escape(tagline)}&rdquo;</div>'
               if tagline else "")
            + genre_chips(movie.get("genre_list", ""))
            + (f'<div class="overview">{html.escape(overview)}</div>' if overview else "")
            + f'<dl class="cm-facts">{facts_html}</dl></div>',
            unsafe_allow_html=True,
        )
        if action_label and on_action:
            st.write("")
            # on_click runs before the rerun, so the action lands even if the
            # dialog is not redrawn; the full rerun then closes the pop-up.
            if st.button(action_label, type="primary", width="stretch",
                         on_click=on_action):
                st.rerun()

    st.divider()
    list_controls(movie, key=f"dlg_{int(movie['id'])}")


# ---------------------------------------------------------------------------
# TABLES
# ---------------------------------------------------------------------------
DISPLAY_COLUMNS = ["id", "title", "year", "genre_list", "director",
                   "vote_average", "runtime"]

COLUMN_LABELS = {
    "id": "ID", "title": "Title", "year": "Year", "genre_list": "Genres",
    "director": "Director", "vote_average": "Rating", "runtime": "Minutes",
    "similarity_pct": "Match %", "cast": "Cast", "keywords": "Keywords",
}


def _column_config(table: pd.DataFrame) -> dict:
    """Ratings and match scores read faster as a filled bar than as a number."""
    config = {
        "Rating": st.column_config.ProgressColumn(
            "Rating", help="Average viewer score out of 10",
            format="%.1f", min_value=0, max_value=10),
        "Match %": st.column_config.ProgressColumn(
            "Match %", help="How similar this movie is to the one you picked",
            format="%.1f%%", min_value=0, max_value=100),
        "Minutes": st.column_config.NumberColumn("Minutes", format="%d min"),
        "Year": st.column_config.NumberColumn("Year", format="%d"),
    }
    return {name: column for name, column in config.items() if name in table.columns}


def movie_table(df: pd.DataFrame, extra_columns: list[str] | None = None,
                height: int | None = None) -> None:
    """Show movies as a tidy table with friendly column names."""
    columns = [c for c in DISPLAY_COLUMNS if c in df.columns]
    for extra in extra_columns or []:
        if extra in df.columns and extra not in columns:
            columns.insert(2, extra)

    table = df[columns].rename(columns=COLUMN_LABELS)

    # height=None is not accepted by st.dataframe, so it is only passed when set.
    options = {"height": height} if height else {}
    st.dataframe(table, width="stretch", hide_index=True,
                 column_config=_column_config(table), **options)
