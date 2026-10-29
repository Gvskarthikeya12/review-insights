import sqlite3, sys
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 120)

path = sys.argv[1] if len(sys.argv) > 1 else "sql/analysis.sql"
conn = sqlite3.connect("data/reviews.db")
sql = open(path, encoding="utf-8").read()

for block in sql.split(";"):
    body = "\n".join(l for l in block.splitlines() if not l.strip().startswith("--")).strip()
    if not body:
        continue
    title = next((l.strip("- ").strip() for l in block.splitlines() if l.strip().startswith("--")), "Query")
    print(f"\n=== {title}")
    print(pd.read_sql_query(body, conn).to_string(index=False))
