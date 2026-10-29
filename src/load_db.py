import sqlite3
import pandas as pd

RENAME = {
    "reviewId": "review_id", "content": "review_text", "score": "rating",
    "thumbsUpCount": "thumbs_up", "appVersion": "app_version",
    "at": "review_date", "replyContent": "reply_text", "repliedAt": "reply_date",
}

df = pd.concat(
    [pd.read_csv(f"data/raw/{a}.csv") for a in ["zomato", "swiggy"]],
    ignore_index=True,
).rename(columns=RENAME)

before = len(df)

# --- cleaning ---
df = df.dropna(subset=["review_text"])
df["review_text"] = df["review_text"].astype(str).str.strip()
df = df[df["review_text"].str.len() > 0]
df = df.drop_duplicates(subset="review_id")
df["review_date"] = pd.to_datetime(df["review_date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
df["reply_date"] = pd.to_datetime(df["reply_date"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
df["word_count"] = df["review_text"].str.split().str.len()
df["has_reply"] = df["reply_text"].notna().astype(int)

cols = ["review_id", "app", "rating", "review_text", "word_count", "thumbs_up",
        "app_version", "review_date", "has_reply", "reply_text", "reply_date"]
df = df[cols]

# --- schema ---
conn = sqlite3.connect("data/reviews.db")
conn.executescript("""
DROP TABLE IF EXISTS reviews;
CREATE TABLE reviews (
    review_id    TEXT PRIMARY KEY,
    app          TEXT NOT NULL,
    rating       INTEGER CHECK (rating BETWEEN 1 AND 5),
    review_text  TEXT NOT NULL,
    word_count   INTEGER,
    thumbs_up    INTEGER,
    app_version  TEXT,
    review_date  TEXT,
    has_reply    INTEGER,
    reply_text   TEXT,
    reply_date   TEXT
);
CREATE INDEX idx_app_date ON reviews (app, review_date);
""")
df.to_sql("reviews", conn, if_exists="append", index=False)
conn.commit()

print(f"Raw rows:     {before}")
print(f"Clean rows:   {len(df)}")
print(f"Removed:      {before - len(df)} (empty or duplicate)")
print(df.groupby("app").size().to_string())
