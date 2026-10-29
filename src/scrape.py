from google_play_scraper import reviews, Sort
import pandas as pd
import os
import time

APPS = {
    "zomato": "com.application.zomato",
    "swiggy": "in.swiggy.android",
}
TARGET = 40000  # reviews per app (~2-3 months)

os.makedirs("data/raw", exist_ok=True)

def fetch(app_id, n):
    out, token = [], None
    while len(out) < n:
        batch, token = reviews(
            app_id, lang="en", country="in",
            sort=Sort.NEWEST, count=200, continuation_token=token,
        )
        if not batch:
            break
        out.extend(batch)
        print(f"  {len(out)} reviews so far")
        if token is None:
            break
        time.sleep(1)  # be polite to the server
    return out[:n]

for name, app_id in APPS.items():
    print(f"Scraping {name}...")
    df = pd.DataFrame(fetch(app_id, TARGET))
    df = df.drop(columns=["userName", "userImage"], errors="ignore")  # no personal data
    df["app"] = name
    df.to_csv(f"data/raw/{name}.csv", index=False)
    print(f"Saved {len(df)} rows to data/raw/{name}.csv\n")
