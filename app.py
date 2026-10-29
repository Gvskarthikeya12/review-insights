"""
Review Insights dashboard - Zomato vs Swiggy Play Store reviews.
Run:  streamlit run app.py
"""
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score

DB = "data/reviews.db"
COLORS = {"zomato": "#E23744", "swiggy": "#FC8019"}

st.set_page_config(page_title="Review Insights: Zomato vs Swiggy", page_icon="📊", layout="wide")


@st.cache_data
def load():
    conn = sqlite3.connect(DB)
    reviews = pd.read_sql_query(
        "SELECT review_id, app, rating, review_text, word_count, thumbs_up, review_date, has_reply FROM reviews",
        conn, parse_dates=["review_date"])
    sent = pd.read_sql_query("SELECT * FROM sentiment", conn)
    clusters = pd.read_sql_query("SELECT * FROM clusters", conn)
    conn.close()
    names = pd.read_csv("data/theme_names.csv")
    return reviews, sent, clusters, names


reviews, sent, clusters, names = load()

# ---------- sidebar ----------
st.sidebar.title("Filters")
apps = st.sidebar.multiselect("App", ["zomato", "swiggy"], default=["zomato", "swiggy"])
if not apps:
    st.warning("Pick at least one app.")
    st.stop()
r = reviews[reviews.app.isin(apps)]
st.sidebar.markdown(
    f"**Data:** {len(reviews):,} Google Play reviews  \n"
    f"{reviews.review_date.min():%d %b %Y} to {reviews.review_date.max():%d %b %Y}"
)

# ---------- header + KPIs ----------
st.title("What are Zomato and Swiggy users complaining about?")
st.caption("80,000 Play Store reviews · SQL · RoBERTa sentiment · sentence-embedding clustering")

by_app = r.groupby("app").agg(reviews=("rating", "size"), avg=("rating", "mean"),
                              neg=("rating", lambda s: (s <= 2).mean() * 100))
cols = st.columns(len(by_app) * 2)
for i, (app, row) in enumerate(by_app.iterrows()):
    cols[i * 2].metric(f"{app.title()} avg rating", f"{row.avg:.2f} ★")
    cols[i * 2 + 1].metric(f"{app.title()} 1–2 star reviews", f"{row.neg:.1f}%")

# ---------- trend ----------
st.subheader("Share of negative reviews by week")
wk = (r.assign(week=r.review_date.dt.to_period("W").dt.start_time)
        .groupby(["app", "week"])
        .agg(n=("rating", "size"), pct_negative=("rating", lambda s: (s <= 2).mean() * 100))
        .reset_index())
wk = wk[wk.n >= 200]  # drop partial weeks
fig = px.line(wk, x="week", y="pct_negative", color="app", color_discrete_map=COLORS, markers=True,
              labels={"pct_negative": "% of reviews rated 1–2 ★", "week": ""})
fig.update_layout(height=340, legend_title_text="", yaxis_ticksuffix="%")
st.plotly_chart(fig, use_container_width=True)

# ---------- themes ----------
st.subheader("Complaint themes")
st.caption("14,000+ negative reviews (5+ words), grouped by meaning with all-MiniLM-L6-v2 embeddings + KMeans. "
           "Brand names were masked before embedding so themes form around the complaint, not the app.")
c = (clusters.merge(reviews[["review_id", "app"]], on="review_id")
             .merge(names, on="cluster"))
c = c[c.app.isin(apps)]
share = (c.groupby(["app", "theme_name"]).size()
          .div(c.groupby("app").size(), level="app").mul(100).rename("pct").reset_index())
order = share.groupby("theme_name").pct.mean().sort_values().index.tolist()
fig = px.bar(share, y="theme_name", x="pct", color="app", barmode="group", orientation="h",
             color_discrete_map=COLORS, category_orders={"theme_name": order[::-1]},
             labels={"pct": "% of that app's complaints", "theme_name": ""})
fig.update_layout(height=520, legend_title_text="", xaxis_ticksuffix="%")
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns([1, 2])
with left:
    theme = st.selectbox("Explore a theme", order[::-1])
    st.markdown(f"*{names.set_index('theme_name').loc[theme, 'description']}*")
with right:
    ex = (c[c.theme_name == theme]
          .merge(reviews[["review_id", "review_text", "thumbs_up", "rating"]], on="review_id")
          .sort_values("thumbs_up", ascending=False).head(6))
    for _, e in ex.iterrows():
        st.markdown(f"**{e.app.title()} · {e.rating}★ · 👍 {e.thumbs_up}**  \n{e.review_text[:400]}")

# ---------- sentiment models ----------
st.subheader("Do star ratings tell the whole story?")
s = sent.merge(reviews[["review_id", "app", "rating", "review_text"]], on="review_id")
s = s[s.app.isin(apps)]
m1, m2 = st.columns(2)
with m1:
    rows = []
    for m, label in [("vader", "VADER (rule-based)"), ("roberta", "RoBERTa (transformer)")]:
        rows.append({"Model": label,
                     "Accuracy vs stars": f"{accuracy_score(s.star_label, s[m + '_label']):.1%}",
                     "Macro-F1": f"{f1_score(s.star_label, s[m + '_label'], average='macro'):.2f}"})
    st.markdown("**Sentiment model vs star rating**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Ground truth: 1–2★ negative, 3★ neutral, 4–5★ positive. "
               "Both models struggle with neutral: 3★ reviews are genuinely mixed.")
with m2:
    mm = s[(s.rating == 5) & (s.roberta_label == "negative") & (s.roberta_conf > 0.9)]
    st.metric("Reviews where text contradicts the stars", f"{s.mismatch.mean():.1%}")
    st.markdown(f"**{len(mm):,} five-star reviews read as strongly negative**, e.g.:")
    for t in mm.sort_values("roberta_conf", ascending=False).review_text.head(3):
        st.markdown(f"> {t[:200]}")

# ---------- method ----------
with st.expander("How this was built"):
    st.markdown("""
1. **Scraped** 40,000 recent Google Play reviews per app with `google-play-scraper` (no user names stored).
2. **Cleaned and loaded** into SQLite; analysis queries in `sql/analysis.sql`.
3. **Sentiment:** VADER baseline vs `cardiffnlp/twitter-roberta-base-sentiment-latest`, scored against star ratings.
4. **Themes:** negative reviews embedded with `all-MiniLM-L6-v2`, clustered with KMeans (k chosen by silhouette score),
   labelled with c-TF-IDF keywords and named by hand.
5. **Bug caught:** the first clustering grouped reviews by which app they named. Masking brand names fixed it.

Limitations: Play Store reviewers skew towards unhappy users; English and Hinglish only; themes are named by a human.
""")
