# CineMatch — Hybrid Movie Recommender

A Streamlit app combining **user-based CF**, **item-based CF**, and **content-based filtering** on the MovieLens ml-latest-small dataset.

**Live demo:** https://cinematch-bymeghana.streamlit.app/

**Stack:** Python · Streamlit · scikit-learn · pandas · NumPy

---

## Quickstart (local)

```bash
# 1. Clone the repo
git clone https://github.com/meghana-gajendran/CineMatch.git
cd CineMatch

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

The app auto-downloads the MovieLens ml-latest-small dataset (~1 MB) on first launch into `./ml-latest-small/`. No manual data setup needed.

---

## Deploy on Streamlit Cloud

1. Push `app.py` and `requirements.txt` to a **public GitHub repo** (no other files needed).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch `main`, main file `app.py`.
4. Click **Deploy**. The dataset downloads automatically at startup.

> **Note:** Streamlit Cloud has ephemeral storage — the dataset re-downloads on each cold start (takes ~5 s). This is fine for a demo. For production, serve the CSVs from a GitHub release asset or S3.

---

## How it works

### Data
- `ratings.csv` — userId, movieId, rating (0.5–5.0)
- `movies.csv` — movieId, title, genres
- Filtered to users with ≥ 20 ratings and movies with ≥ 20 ratings (configurable in `build_utility_matrix`).

### Recommendation engines

| Mode | Algorithm | Signal |
|------|-----------|--------|
| User-Based CF | Mean-centred cosine similarity (≡ Pearson correlation) | Other users' rating patterns |
| Item-Based CF | Cosine similarity on item vectors | Movies similar to ones you've rated |
| Content-Based | TF-IDF on genre strings + cosine similarity | Genre profile of a seed movie |
| **Hybrid** | Weighted blend of normalised CF + CB scores | Both |

### Hybrid scoring
Both CF and content-based score lists are independently min-max normalised to [0, 1], then combined:

```
hybrid_score = cf_weight × cf_norm_score + (1 − cf_weight) × cb_norm_score
```

The `cf_weight` slider (default 0.6) is adjustable in the UI.

### Why each movie was recommended
Every recommendation card shows a plain-English explanation:
- **CF**: "Predicted 4.2★ based on 14 users with similar taste to you"
- **Item-Based**: "Similar to 'Toy Story (1995)' that you rated (item similarity: 0.87)"
- **Content**: "Shares genre profile with 'The Matrix (1999)' (Action|Sci-Fi|Thriller)"
- **Hybrid**: both signals shown

---

## Project structure

```
.
├── app.py              ← Streamlit app (single file)
├── requirements.txt
├── README.md
└── ml-latest-small/    ← auto-created on first run
    ├── movies.csv
    └── ratings.csv
```
