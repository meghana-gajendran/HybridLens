"""
CineMatch — Hybrid Movie Recommender
Combines collaborative filtering (user-based + item-based) with content-based
filtering on the MovieLens ml-latest-small dataset.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import os
import urllib.request
import zipfile

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — dark cinema theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Base & background ── */
  .stApp { background: #0d0d0f; }
  section[data-testid="stSidebar"] { background: #131318; border-right: 1px solid #2a2a35; }
  
  /* ── Typography ── */
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Inter:wght@400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #e2e2e9; }
  
  /* ── Hero title ── */
  .hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    background: linear-gradient(135deg, #e8c97e 0%, #f0a05a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.2rem;
  }
  .hero-sub {
    color: #7c7c94;
    font-size: 0.95rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 2rem;
  }

  /* ── Recommendation card ── */
  .rec-card {
    background: #18181f;
    border: 1px solid #25252f;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.85rem;
    transition: border-color 0.2s;
  }
  .rec-card:hover { border-color: #e8c97e44; }
  .rec-rank {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: #e8c97e55;
    float: right;
    margin-top: -4px;
  }
  .rec-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #f0ede8;
    margin-bottom: 4px;
  }
  .rec-genre-pill {
    display: inline-block;
    background: #1e1e2a;
    border: 1px solid #35354a;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    color: #9090aa;
    margin-right: 4px;
    margin-bottom: 6px;
  }
  .rec-why {
    font-size: 0.82rem;
    color: #7c7c94;
    margin-top: 6px;
    border-left: 2px solid #e8c97e55;
    padding-left: 8px;
  }
  .rec-score {
    font-size: 0.78rem;
    color: #e8c97e;
    margin-top: 4px;
  }
  
  /* ── Mode badge ── */
  .mode-badge {
    display: inline-block;
    background: #e8c97e18;
    border: 1px solid #e8c97e44;
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 0.78rem;
    color: #e8c97e;
    margin-bottom: 1.2rem;
  }

  /* ── Streamlit widget overrides ── */
  div[data-baseweb="select"] > div,
  div[data-baseweb="input"] > div > input {
    background: #18181f !important;
    border-color: #2a2a35 !important;
    color: #e2e2e9 !important;
  }
  .stSlider > div { color: #e2e2e9; }
  .stRadio label { color: #c0c0cc; }
  label[data-testid="stWidgetLabel"] { color: #9090aa; font-size: 0.82rem; }
  
  /* ── Section divider ── */
  hr { border-color: #22222e; }
  
  /* ── Info box ── */
  .info-box {
    background: #18181f;
    border: 1px solid #25252f;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    color: #7c7c94;
    font-size: 0.85rem;
  }
  
  /* ── Spinner ── */
  .stSpinner > div { border-top-color: #e8c97e !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Data loading & caching
# ─────────────────────────────────────────────
DATA_DIR = "ml-latest-small"
ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

def download_data():
    """Download MovieLens ml-latest-small if not already present."""
    if not os.path.exists(DATA_DIR):
        with st.spinner("Downloading MovieLens dataset (~1MB)…"):
            urllib.request.urlretrieve(ZIP_URL, "ml-latest-small.zip")
            with zipfile.ZipFile("ml-latest-small.zip", "r") as zf:
                zf.extractall(".")
            os.remove("ml-latest-small.zip")


@st.cache_data(show_spinner=False)
def load_data():
    download_data()
    movies = pd.read_csv(f"{DATA_DIR}/movies.csv")
    ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")

    # Drop timestamp — not needed
    ratings = ratings.drop(columns=["timestamp"])

    # Merge for EDA convenience
    df = pd.merge(ratings, movies, on="movieId")
    return movies, ratings, df


@st.cache_data(show_spinner=False)
def build_utility_matrix(ratings: pd.DataFrame, min_user_ratings=100, min_movie_ratings=100):
    """
    Filter aggressively to keep matrix small for Streamlit Cloud (1GB RAM).
    Thresholds of 100 keep the utility matrix under ~200x300.
    """
    rc = ratings["movieId"].value_counts()
    popular = rc[rc >= min_movie_ratings].index
    df = ratings[ratings["movieId"].isin(popular)]

    uc = df["userId"].value_counts()
    active = uc[uc >= min_user_ratings].index
    df = df[df["userId"].isin(active)]

    utility = df.pivot_table(index="userId", columns="movieId", values="rating")
    return utility, df


@st.cache_data(show_spinner=False)
def build_user_similarity(_utility: pd.DataFrame):
    """Mean-centred cosine similarity → equivalent to Pearson correlation."""
    means = _utility.mean(axis=1)
    norm_u = _utility.subtract(means, axis=0).fillna(0).astype("float32")
    sim = cosine_similarity(norm_u)
    return pd.DataFrame(sim, index=_utility.index, columns=_utility.index, dtype="float32")


@st.cache_data(show_spinner=False)
def build_item_similarity(_utility: pd.DataFrame):
    """Item-item cosine similarity on the transposed utility matrix."""
    item_mat = _utility.T.fillna(0).astype("float32")
    sim = cosine_similarity(item_mat)
    return pd.DataFrame(sim, index=_utility.columns, columns=_utility.columns, dtype="float32")


@st.cache_data(show_spinner=False)
def build_content_matrix(_movies: pd.DataFrame):
    """TF-IDF on genre strings for content-based similarity."""
    genre_text = _movies["genres"].str.replace("|", " ", regex=False).fillna("")
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(genre_text)
    sim = cosine_similarity(tfidf_matrix, tfidf_matrix).astype("float32")
    return pd.DataFrame(sim, index=_movies["movieId"].values, columns=_movies["movieId"].values, dtype="float32")


# ─────────────────────────────────────────────
# Recommendation engines
# ─────────────────────────────────────────────

def user_based_cf(user_id: int, utility: pd.DataFrame, user_sim: pd.DataFrame,
                  movies: pd.DataFrame, top_n: int = 10):
    """
    User-based CF: find similar users, predict ratings for unseen movies.
    Returns list of dicts with movieId, title, genres, score, explanation.
    """
    if user_id not in utility.index:
        return [], f"User {user_id} not found in the filtered dataset."

    user_ratings = utility.loc[user_id]
    unseen = user_ratings[user_ratings.isnull()].index
    similar_users = user_sim[user_id].drop(user_id).sort_values(ascending=False)

    weighted, total_sim = defaultdict(float), defaultdict(float)
    top_contributors: dict[int, list] = defaultdict(list)

    for v_id, sim_score in similar_users.items():
        if sim_score <= 0:
            continue
        v_ratings = utility.loc[v_id]
        for mid in unseen:
            if mid in v_ratings.index and pd.notna(v_ratings[mid]):
                weighted[mid] += sim_score * v_ratings[mid]
                total_sim[mid] += sim_score
                top_contributors[mid].append((sim_score, v_id))

    scores = {mid: weighted[mid] / total_sim[mid]
              for mid, w in weighted.items() if total_sim[mid] > 0}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    movie_info = movies.set_index("movieId")
    results = []
    for mid, score in ranked:
        if mid not in movie_info.index:
            continue
        row = movie_info.loc[mid]
        n_contrib = len(top_contributors[mid])
        genres = [g for g in row["genres"].split("|") if g != "(no genres listed)"]
        results.append({
            "movieId": mid,
            "title": row["title"],
            "genres": genres,
            "score": round(score, 3),
            "why": f"Predicted {score:.2f}★ based on {n_contrib} users with similar taste to you",
            "method": "User-Based CF",
        })
    return results, None


def item_based_cf(user_id: int, utility: pd.DataFrame, item_sim: pd.DataFrame,
                  movies: pd.DataFrame, top_n: int = 10):
    """
    Item-based CF: for each unseen movie, compute weighted average over rated movies.
    """
    if user_id not in utility.index:
        return [], f"User {user_id} not found."

    user_ratings = utility.loc[user_id]
    rated = user_ratings.dropna()
    unseen = user_ratings[user_ratings.isnull()].index
    # Only use movies that appear in item_sim
    rated = rated[rated.index.isin(item_sim.index)]
    unseen = [m for m in unseen if m in item_sim.index]

    predictions = {}
    best_anchor: dict[int, str] = {}

    for mid_M in unseen:
        sims_to_M = item_sim[mid_M]
        num, den = 0.0, 0.0
        best_sim, best_mid = -1.0, None
        for mid_R, r in rated.items():
            if mid_R not in sims_to_M.index:
                continue
            s = sims_to_M[mid_R]
            if s > 0:
                num += s * r
                den += s
                if s > best_sim:
                    best_sim, best_mid = s, mid_R
        if den > 0:
            predictions[mid_M] = (num / den, best_mid, best_sim)

    ranked = sorted(predictions.items(), key=lambda x: x[1][0], reverse=True)[:top_n]
    movie_info = movies.set_index("movieId")
    results = []
    for mid, (score, anchor_id, anchor_sim) in ranked:
        if mid not in movie_info.index:
            continue
        row = movie_info.loc[mid]
        anchor_title = movie_info.loc[anchor_id]["title"] if anchor_id and anchor_id in movie_info.index else "a movie you rated"
        genres = [g for g in row["genres"].split("|") if g != "(no genres listed)"]
        results.append({
            "movieId": mid,
            "title": row["title"],
            "genres": genres,
            "score": round(score, 3),
            "why": f"Similar to '{anchor_title}' that you rated (item similarity: {anchor_sim:.2f})",
            "method": "Item-Based CF",
        })
    return results, None


def content_based(movie_title: str, movies: pd.DataFrame,
                  content_sim: pd.DataFrame, top_n: int = 10):
    """
    Content-based: find movies with similar genre profiles.
    """
    import re
    clean_title = movie_title.strip()

    # Reject if fewer than 2 alphanumeric characters
    if sum(c.isalnum() for c in clean_title) < 2:
        return [], "Please enter a valid movie title (at least 2 letters or numbers)."

    # Reject if contains special characters beyond letters, numbers, spaces, ' - : .
    if re.search(r"[^a-zA-Z0-9\s'\-:\.]", clean_title):
        return [], f"'{clean_title}' doesn't look like a movie title. Try something like 'Toy Story', 'Matrix', or 'Pulp Fiction'."

    matches = movies[movies["title"].str.lower().str.contains(clean_title.lower(), na=False, regex=False)]
    if matches.empty:
        return [], f"No movie matching '{clean_title}' found. Try 'Toy Story', 'Matrix', or 'Pulp Fiction'."

    # Pick the first match (most likely the intended one)
    seed = matches.iloc[0]
    seed_id = seed["movieId"]
    seed_genres = seed["genres"]

    if seed_id not in content_sim.index:
        return [], f"'{seed['title']}' found in catalog but has no genre data for similarity."

    sim_scores = content_sim[seed_id].drop(seed_id).sort_values(ascending=False)
    top_ids = sim_scores.head(top_n).index
    movie_info = movies.set_index("movieId")
    results = []
    for mid in top_ids:
        if mid not in movie_info.index:
            continue
        row = movie_info.loc[mid]
        score = sim_scores[mid]
        genres = [g for g in row["genres"].split("|") if g != "(no genres listed)"]
        results.append({
            "movieId": mid,
            "title": row["title"],
            "genres": genres,
            "score": round(float(score), 3),
            "why": f"Shares genre profile with '{seed['title']}' ({seed_genres})",
            "method": "Content-Based (Genre TF-IDF)",
        })
    return results, seed["title"]


def hybrid_recommend(user_id: int, movie_title: str,
                     utility, user_sim, item_sim, content_sim,
                     movies, cf_mode: str, cf_weight: float, top_n: int = 10):
    """
    Hybrid: blend CF score with content-based score.
    We normalise both score lists to [0,1] and combine with cf_weight.
    """
    cb_weight = 1.0 - cf_weight

    # ── CF part ──
    if cf_mode == "User-Based":
        cf_recs, cf_err = user_based_cf(user_id, utility, user_sim, movies, top_n=30)
    else:
        cf_recs, cf_err = item_based_cf(user_id, utility, item_sim, movies, top_n=30)

    # ── Content-Based part ──
    cb_recs, cb_meta = content_based(movie_title, movies, content_sim, top_n=30)

    if cf_err and not cb_recs:
        return [], cf_err
    if not cb_recs and not cf_recs:
        return [], "No recommendations found. Try a different user ID or movie title."

    # ── Normalize and merge ──
    def normalize(recs):
        scores = [r["score"] for r in recs]
        if not scores:
            return {}
        mn, mx = min(scores), max(scores)
        span = mx - mn if mx != mn else 1.0
        return {r["movieId"]: (r["score"] - mn) / span for r in recs}

    cf_norm = normalize(cf_recs)
    cb_norm = normalize(cb_recs)

    # Collect all candidate movie IDs
    all_ids = set(cf_norm) | set(cb_norm)
    cf_why = {r["movieId"]: r["why"] for r in cf_recs}
    cb_why = {r["movieId"]: r["why"] for r in cb_recs}
    all_info = {r["movieId"]: r for r in (cf_recs + cb_recs)}

    combined = []
    for mid in all_ids:
        cf_s = cf_norm.get(mid, 0.0)
        cb_s = cb_norm.get(mid, 0.0)
        hybrid_s = cf_weight * cf_s + cb_weight * cb_s
        info = all_info[mid]

        # Build explanation
        parts = []
        if mid in cf_norm:
            parts.append(f"CF ({cf_mode}): {cf_why.get(mid, '')}")
        if mid in cb_norm:
            parts.append(f"Content: {cb_why.get(mid, '')}")
        why_hybrid = " · ".join(parts)

        combined.append({
            "movieId": mid,
            "title": info["title"],
            "genres": info["genres"],
            "score": round(hybrid_s, 4),
            "cf_score": round(cf_s, 3),
            "cb_score": round(cb_s, 3),
            "why": why_hybrid,
            "method": f"Hybrid ({cf_mode} CF {cf_weight:.0%} + Content {cb_weight:.0%})",
        })

    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:top_n], None


# ─────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────

def render_card(rec: dict, rank: int):
    genres_html = "".join(
        f'<span class="rec-genre-pill">{g}</span>' for g in rec["genres"]
    )
    score_line = f'<span class="rec-score">Score: {rec["score"]}</span>'
    if "cf_score" in rec:
        score_line += f' &nbsp;·&nbsp; <span style="color:#7c7c94;font-size:0.75rem">CF: {rec["cf_score"]} | CB: {rec["cb_score"]}</span>'

    st.markdown(f"""
    <div class="rec-card">
      <span class="rec-rank">#{rank}</span>
      <div class="rec-title">{rec["title"]}</div>
      <div>{genres_html}</div>
      {score_line}
      <div class="rec-why">{rec["why"]}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load everything
# ─────────────────────────────────────────────
with st.spinner("Loading data and building similarity matrices…"):
    movies, ratings, df = load_data()
    utility, df_filtered = build_utility_matrix(ratings)
    user_sim = build_user_similarity(utility)
    item_sim = build_item_similarity(utility)
    content_sim = build_content_matrix(movies)

valid_users = sorted(utility.index.tolist())
all_titles = sorted(movies["title"].tolist())


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")

    mode = st.radio(
        "Recommendation mode",
        ["Hybrid (CF + Content)", "Collaborative Filtering only", "Content-Based only"],
        index=0,
    )

    st.markdown("---")

    cf_mode = st.radio("CF algorithm", ["User-Based", "Item-Based"], index=0)

    if "Hybrid" in mode:
        cf_weight = st.slider("CF weight in hybrid blend", 0.0, 1.0, 0.6, 0.05,
                               help="1.0 = pure CF, 0.0 = pure content")
    else:
        cf_weight = 1.0 if "Collaborative" in mode else 0.0

    top_n = st.slider("Number of recommendations", 5, 20, 10)

    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    <strong>Dataset</strong><br>
    MovieLens ml-latest-small<br>
    ~100K ratings · ~9K movies · ~600 users<br><br>
    <strong>Methods</strong><br>
    • User-Based CF (Pearson/cosine)<br>
    • Item-Based CF (cosine)<br>
    • Content-Based (TF-IDF genres)<br>
    • Hybrid blend
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">CineMatch</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Hybrid Movie Recommendations · MovieLens ml-latest-small</div>', unsafe_allow_html=True)

col_l, col_r = st.columns([1, 1], gap="large")

needs_user = "Collaborative" in mode or "Hybrid" in mode
needs_movie = "Content" in mode or "Hybrid" in mode

with col_l:
    if needs_user:
        user_id_input = st.selectbox(
            "Your User ID",
            options=valid_users,
            index=0,
            help="Select a user ID from the MovieLens dataset",
        )
    else:
        user_id_input = None

with col_r:
    if needs_movie:
        movie_input = st.text_input(
            "Seed movie (for content-based)",
            placeholder="e.g. Toy Story, Pulp Fiction, Matrix…",
            help="Type any part of a movie title",
        )
    else:
        movie_input = ""

# Show what the user has rated (for context)
if needs_user and user_id_input:
    rated_by_user = df_filtered[df_filtered["userId"] == user_id_input].merge(movies, on="movieId")
    rated_by_user = rated_by_user.sort_values("rating", ascending=False)

    with st.expander(f"User {user_id_input}'s top-rated movies ({len(rated_by_user)} total in filtered set)", expanded=False):
        top_rated = rated_by_user.head(10)[["title", "genres", "rating"]]
        st.dataframe(top_rated, hide_index=True)

st.markdown("<br>", unsafe_allow_html=True)
run_btn = st.button("Get Recommendations", use_container_width=False, type="primary")

st.markdown("---")

# ─────────────────────────────────────────────
# Recommendation output
# ─────────────────────────────────────────────
if run_btn:
    error = None
    warning_msg = None
    results = []

    with st.spinner("Computing recommendations…"):
        if "Hybrid" in mode:
            if not movie_input.strip():
                st.warning("For Hybrid mode, please enter a seed movie title for the content-based component.")
                st.stop()

            # Check if seed movie exists before running hybrid
            seed_check = movies[movies["title"].str.lower().str.contains(
                movie_input.strip().lower(), na=False, regex=False)]
            if seed_check.empty:
                warning_msg = (
                    f"Seed movie '{movie_input.strip()}' not found in the dataset — "
                    f"showing pure CF results instead. Try a title like 'Toy Story', 'Matrix', or 'Pulp Fiction'."
                )

            results, error = hybrid_recommend(
                user_id=user_id_input,
                movie_title=movie_input.strip(),
                utility=utility,
                user_sim=user_sim,
                item_sim=item_sim,
                content_sim=content_sim,
                movies=movies,
                cf_mode=cf_mode,
                cf_weight=cf_weight,
                top_n=top_n,
            )
        elif "Collaborative" in mode:
            if cf_mode == "User-Based":
                results, error = user_based_cf(user_id_input, utility, user_sim, movies, top_n)
            else:
                results, error = item_based_cf(user_id_input, utility, item_sim, movies, top_n)
        else:  # Content-Based only
            if not movie_input.strip():
                st.warning("Please enter a seed movie title.")
                st.stop()
            results, seed_title = content_based(movie_input.strip(), movies, content_sim, top_n)
            if not results:
                error = f"No movie matching '{movie_input.strip()}' found. Try a partial title like 'star' or 'toy'."

    if warning_msg:
        st.warning(warning_msg)

    if error:
        st.error(error)
    elif not results:
        st.warning("No recommendations generated. Try different inputs or a lower filter threshold.")
    else:
        method_label = results[0]["method"] if results else ""
        st.markdown(f'<div class="mode-badge">{method_label}</div>', unsafe_allow_html=True)

        st.markdown(f"### Top {len(results)} Recommendations")
        for i, rec in enumerate(results, 1):
            render_card(rec, i)

# ─────────────────────────────────────────────
# Footer stats
# ─────────────────────────────────────────────
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total movies", f"{len(movies):,}")
c2.metric("Total ratings", f"{len(ratings):,}")
c3.metric("Active users (filtered)", f"{len(valid_users):,}")
c4.metric("Movies in CF matrix", f"{utility.shape[1]:,}")