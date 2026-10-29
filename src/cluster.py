"""
Phase 4 - Topic clustering of complaints.
Embeds negative reviews, clusters them with KMeans, and labels each
cluster with its most distinctive words (c-TF-IDF).

Usage:
    python src/cluster.py              # sentence-transformer embeddings
    python src/cluster.py --k 10       # force a number of clusters
    python src/cluster.py --fast       # TF-IDF + SVD instead (no model download)
Embeddings are cached in data/embeddings.npy, so reruns are quick.
"""
import argparse
import os
import sqlite3

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_score

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=0, help="number of clusters (0 = choose automatically)")
ap.add_argument("--fast", action="store_true", help="TF-IDF + SVD embeddings, no download")
args = ap.parse_args()

conn = sqlite3.connect("data/reviews.db")
df = pd.read_sql_query("""
    SELECT r.review_id, r.app, r.rating, r.review_text, r.review_date
    FROM reviews r JOIN sentiment s USING (review_id)
    WHERE s.roberta_label = 'negative' AND r.word_count >= 5
    ORDER BY r.review_id
""", conn)
print(f"Complaints to cluster: {len(df):,}")

# Mask brand names so clusters form around the complaint, not around which app is named.
# (Without this, the model groups "bad experience with Swiggy" vs "...with Zomato".)
BRANDS = r"(?i)\b(zomato|swiggy|instamart|blinkit|zepto|swigy|zomatto|zomoto)\b"
df["text_masked"] = df["review_text"].str.replace(BRANDS, "the app", regex=True)

# --- 1. Embeddings (cached) ---
CACHE = "data/embeddings_fast_masked.npy" if args.fast else "data/embeddings_masked.npy"
if os.path.exists(CACHE) and np.load(CACHE, mmap_mode="r").shape[0] == len(df):
    X = np.load(CACHE)
    print(f"Loaded cached embeddings {X.shape}")
elif args.fast:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize
    tf = TfidfVectorizer(stop_words="english", min_df=5, ngram_range=(1, 2)).fit_transform(df["text_masked"])
    X = normalize(TruncatedSVD(100, random_state=42).fit_transform(tf))
    np.save(CACHE, X)
else:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    X = model.encode(df["text_masked"].tolist(), batch_size=64, show_progress_bar=True,
                     normalize_embeddings=True)
    np.save(CACHE, X)
    print(f"Embeddings {X.shape} saved to {CACHE}")

# --- 2. Choose k with silhouette score ---
if args.k:
    best_k = args.k
else:
    print("\nChoosing number of clusters:")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), min(5000, len(X)), replace=False)
    scores = {}
    for k in range(6, 15):
        labels = KMeans(k, n_init=10, random_state=42).fit_predict(X[idx])
        scores[k] = silhouette_score(X[idx], labels, metric="cosine")
        print(f"  k={k:2d}  silhouette {scores[k]:.4f}")
    best_k = max(scores, key=scores.get)
print(f"\nUsing k = {best_k}")

km = KMeans(best_k, n_init=10, random_state=42).fit(X)
df["cluster"] = km.labels_

# --- 3. Label clusters with c-TF-IDF ---
docs = df.groupby("cluster")["review_text"].apply(" ".join)
cv = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3,
                     token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
counts = cv.fit_transform(docs).toarray().astype(float)
tf = counts / counts.sum(axis=1, keepdims=True)
idf = np.log(1 + counts.sum() / (counts.sum(axis=0) + 1))
ctfidf = tf * idf
vocab = np.array(cv.get_feature_names_out())
generic = {"zomato", "swiggy", "app", "order", "ordered", "food", "worst", "bad", "service", "instamart",
           "don", "didn", "just", "experience", "good", "use", "really", "pathetic", "poor"}

rows = []
for c in docs.index:
    order = ctfidf[c].argsort()[::-1]
    words = [w for w in vocab[order] if not set(w.split()) & generic][:8]
    sub = df[df.cluster == c]
    centre = km.cluster_centers_[c]
    example = sub.iloc[np.argmax(X[sub.index] @ centre)]["review_text"]
    rows.append({
        "cluster": c, "size": len(sub), "pct": round(100 * len(sub) / len(df), 1),
        "top_words": ", ".join(words),
        "zomato_pct": round(100 * (sub.app == "zomato").sum() / (df.app == "zomato").sum(), 1),
        "swiggy_pct": round(100 * (sub.app == "swiggy").sum() / (df.app == "swiggy").sum(), 1),
        "example": example[:200],
    })
themes = pd.DataFrame(rows).sort_values("size", ascending=False)
themes["theme_name"] = ""  # fill in after reading the top words

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 90)
print("\n=== Complaint themes ===")
print(themes[["cluster", "size", "pct", "zomato_pct", "swiggy_pct", "top_words"]].to_string(index=False))
print("\n=== Most typical complaint per theme ===")
for _, r in themes.iterrows():
    print(f"[{r.cluster}] {r.example}")

# --- 4. Save ---
df[["review_id", "cluster"]].to_sql("clusters", conn, if_exists="replace", index=False)
themes.to_sql("themes", conn, if_exists="replace", index=False)
themes.to_csv("data/themes.csv", index=False)
print("\nSaved tables `clusters` and `themes`, and data/themes.csv")
