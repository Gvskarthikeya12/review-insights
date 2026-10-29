"""
Phase 3 - Sentiment analysis.
Compares a rule-based baseline (VADER) with a transformer (RoBERTa)
against star ratings, and saves results to the `sentiment` table.

Usage:
    python src/sentiment.py            # full run (reviews with 5+ words)
    python src/sentiment.py --sample 500   # quick test
"""
import argparse
import sqlite3

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ap = argparse.ArgumentParser()
ap.add_argument("--sample", type=int, default=0, help="score only N random reviews")
ap.add_argument("--no-transformer", action="store_true", help="VADER only (fast)")
args = ap.parse_args()

conn = sqlite3.connect("data/reviews.db")
df = pd.read_sql_query(
    "SELECT review_id, app, rating, review_text FROM reviews WHERE word_count >= 5", conn
)
if args.sample:
    df = df.sample(args.sample, random_state=42)
print(f"Scoring {len(df):,} reviews")

# ground truth from stars: 1-2 negative, 3 neutral, 4-5 positive
df["star_label"] = pd.cut(df["rating"], [0, 2, 3, 5], labels=["negative", "neutral", "positive"]).astype(str)

# --- 1. VADER baseline ---
vader = SentimentIntensityAnalyzer()
df["vader_score"] = df["review_text"].map(lambda t: vader.polarity_scores(t)["compound"])
df["vader_label"] = pd.cut(df["vader_score"], [-1.01, -0.05, 0.05, 1.01],
                           labels=["negative", "neutral", "positive"]).astype(str)
print("VADER done")

# --- 2. RoBERTa transformer ---
if not args.no_transformer:
    from transformers import pipeline
    clf = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        truncation=True, max_length=256,
    )
    texts = df["review_text"].tolist()
    out = []
    for i in range(0, len(texts), 64):
        out.extend(clf(texts[i:i + 64], batch_size=64))
        if (i // 64) % 20 == 0:
            print(f"  RoBERTa {min(i + 64, len(texts)):,}/{len(texts):,}")
    df["roberta_label"] = [o["label"].lower() for o in out]
    df["roberta_conf"] = [round(o["score"], 4) for o in out]

# --- 3. Evaluate against stars ---
models = ["vader"] + ([] if args.no_transformer else ["roberta"])
print("\n=== Model vs star rating ===")
for m in models:
    acc = accuracy_score(df["star_label"], df[f"{m}_label"])
    f1 = f1_score(df["star_label"], df[f"{m}_label"], average="macro")
    print(f"{m:8s} accuracy {acc:.3f}   macro-F1 {f1:.3f}")
best = max(models, key=lambda m: f1_score(df["star_label"], df[f"{m}_label"], average="macro"))
print(f"\nBest model: {best}")
print(classification_report(df["star_label"], df[f"{best}_label"], digits=3))

# --- 4. Mismatches: text says one thing, stars say another ---
df["mismatch"] = ((df["rating"] >= 4) & (df[f"{best}_label"] == "negative")) | \
                 ((df["rating"] <= 2) & (df[f"{best}_label"] == "positive"))
print(f"Mismatched reviews: {df['mismatch'].sum():,} ({100 * df['mismatch'].mean():.1f}%)")
print(df[(df["rating"] == 5) & (df[f"{best}_label"] == "negative")]
      [["app", "review_text"]].head(5).to_string(index=False, max_colwidth=110))

# --- 5. Save ---
keep = ["review_id", "star_label", "vader_score", "vader_label"] + \
       ([] if args.no_transformer else ["roberta_label", "roberta_conf"]) + ["mismatch"]
df[keep].to_sql("sentiment", conn, if_exists="replace", index=False)
print("\nSaved to table `sentiment`")
