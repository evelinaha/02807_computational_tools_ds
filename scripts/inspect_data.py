"""Step 0 of the graph-clustering part: inspect the books_50pct data before choosing any threshold.

Reads the two Parquet files out of core with DuckDB (the reviews file is 2.3 GB, mostly review
text) and prints: schema and row counts, sample rows, missing values, rating and year
distributions, reviews per user / book, metadata coverage (titles, descriptions, author/brand,
genre categories), and iterative k-core simulations for candidate thresholds and cut-offs.

Usage (from the repo root):
    uv run python scripts/inspect_data.py [DATA_DIR] > docs/results/data_inspection.txt

DATA_DIR defaults to ../data/books_50pct (or the CTDS_DATA_DIR environment variable).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

DATA_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "CTDS_DATA_DIR", Path(__file__).resolve().parents[1].parent / "data" / "books_50pct"))
REVIEWS = DATA_DIR / "books_50pct_users_reviews.parquet"
META = DATA_DIR / "books_50pct_users_meta.parquet"
R = f"read_parquet('{REVIEWS}')"
M = f"read_parquet('{META}')"

con = duckdb.connect(config={"memory_limit": "6GB", "threads": 8})
con.execute("set TimeZone = 'UTC'")


def q(title: str, sql: str) -> None:
    t = time.time()
    print(f"\n### {title}")
    print(con.execute(sql).fetchdf().to_string(max_colwidth=100))
    print(f"({time.time() - t:.1f}s)", flush=True)


def one(sql: str):
    return con.execute(sql).fetchone()


# ------------------------------------------------------------------ files and schema
for path in (REVIEWS, META):
    pf = pq.ParquetFile(path)
    print(f"== {path.name}: {path.stat().st_size / 1e6:,.0f} MB, {pf.metadata.num_rows:,} rows, "
          f"{pf.metadata.num_row_groups} row groups")
    print(pf.schema_arrow.remove_metadata())

# ------------------------------------------------------------------ reviews
con.execute(f"create table r as select user_id, item_id, rating, timestamp, "
            f"year(to_timestamp(timestamp)) yr from {R}")
q("reviews: size, ids, time range (timestamp is Unix SECONDS), missing values", f"""
  select count(*) n, count(distinct user_id) users, count(distinct item_id) items,
         to_timestamp(min(timestamp)) first_review, to_timestamp(max(timestamp)) last_review,
         sum((rating is null)::int) rating_null,
         sum((review_text is null or review_text = '')::int) empty_text,
         sum((summary is null or summary = '')::int) empty_summary
  from {R}""")
q("sample rows", f"select user_id, item_id, rating, left(summary, 40) summary, left(review_text, 60) text, "
                 f"helpful_votes, timestamp from {R} limit 5")
q("duplicate (user, item) pairs", "select count(*) dup_pairs from (select user_id, item_id from r group by 1, 2 having count(*) > 1)")
q("rating distribution", "select rating, count(*) n, round(100.0 * count(*) / sum(count(*)) over (), 2) pct from r group by 1 order by 1")
q("reviews per year", "select yr, count(*) n, round(100.0 * sum(count(*)) over (order by yr) / sum(count(*)) over (), 1) cum_pct "
                      "from r group by 1 order by 1")
for name, col in [("user", "user_id"), ("book", "item_id")]:
    q(f"reviews per {name}", f"""
      with c as (select {col}, count(*) n from r group by 1)
      select count(*) cnt, min(n) mn, quantile_cont(n, 0.25) p25, median(n) med, quantile_cont(n, 0.75) p75,
             quantile_cont(n, 0.9) p90, quantile_cont(n, 0.99) p99, max(n) mx, round(avg(n), 1) mean,
             sum((n < 5)::int) lt5, sum((n < 10)::int) lt10, sum((n < 20)::int) lt20, sum((n >= 1000)::int) ge1000
      from c""")

# ------------------------------------------------------------------ metadata
q("metadata: missing-value rates (%)", f"""
  select count(*) n,
         round(100 * avg((title is null or title = '')::int), 1) title_missing,
         round(100 * avg((description is null or description = '')::int), 1) description_missing,
         round(100 * avg((brand is null or brand = '')::int), 1) brand_author_missing,
         round(100 * avg((price is null or isnan(price))::int), 1) price_missing,
         round(100 * avg((len(categories) = 1 and categories[1] = ['Books'])::int), 1) only_books_category
  from {M}""")
q("review items without metadata", f"select count(distinct item_id) n from r anti join {M} m using (item_id)")
con.execute(f"create table paths as select item_id, unnest(categories) p from {M}")
q("top-level categories", "select p[1] lvl1, count(*) n from paths group by 1 order by 2 desc limit 8")
q("second-level genres (Books > X)", "select p[2] genre, count(distinct item_id) books from paths "
                                     "where p[1] = 'Books' and len(p) >= 2 group by 1 order by 2 desc limit 35")
q("Kindle eBooks genres (Kindle Store > Kindle eBooks > X)", "select p[3] genre, count(distinct item_id) books from paths "
  "where p[1] = 'Kindle Store' and p[2] = 'Kindle eBooks' group by 1 order by 2 desc limit 15")


# ------------------------------------------------------------------ k-core simulations
def kcore(where: str, ku: int, kb: int) -> dict:
    con.execute(f"create or replace temp table k as select user_id, item_id, rating, timestamp from r where {where}")
    for it in range(1, 100):
        n0 = one("select count(*) from k")[0]
        con.execute(f"create or replace temp table k as select k.* from k semi join "
                    f"(select user_id from k group by 1 having count(*) >= {ku}) u using (user_id)")
        con.execute(f"create or replace temp table k as select k.* from k semi join "
                    f"(select item_id from k group by 1 having count(*) >= {kb}) b using (item_id)")
        if one("select count(*) from k")[0] == n0:
            break
    n, u, b = one("select count(*), count(distinct user_id), count(distinct item_id) from k")
    out = {"window": where, "user>=": ku, "book>=": kb, "reviews": n, "users": u, "books": b, "iterations": it}
    for cut in ["2013-07-01", "2014-01-01"]:
        out[f"test_share_cut_{cut}"] = round(one(f"select avg((timestamp >= epoch(timestamp '{cut}'))::int) from k")[0], 3)
    return out


print("\n### iterative k-core simulations (applied to the whole window, before the time split)")
rows = [kcore(w, ku, kb) for w in ["true", "yr >= 2010"] for ku, kb in [(5, 10), (5, 15), (5, 20)]]
print(pd.DataFrame(rows).to_string())

print("\n### evaluable test users for the chosen k-core (5/15, all years); liked = rating >= 4")
kcore("true", 5, 15)
for cut in ["2013-07-01", "2014-01-01"]:
    q(f"cut-off {cut}", f"""
      with t as (select *, timestamp < epoch(timestamp '{cut}') tr from k),
           tb as (select distinct item_id from t where tr),
           u as (select user_id, sum((tr and rating >= 4)::int) train_likes,
                        sum((not tr and rating >= 4 and item_id in (select item_id from tb))::int) test_likes
                 from t group by 1)
      select (select sum(tr::int) from t) train_reviews, (select sum((not tr)::int) from t) test_reviews,
             (select count(*) from tb) train_books, sum((train_likes > 0 and test_likes > 0)::int) evaluable_users
      from u""")
