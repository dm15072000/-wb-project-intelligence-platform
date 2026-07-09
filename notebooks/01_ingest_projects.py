# Databricks notebook source
import requests, json, time
from pyspark.sql import functions as F

BASE = "https://search.worldbank.org/api/v2/projects"
rows, offset, page_size = [], 0, 200   # smaller pages are gentler on their API

def fetch_page(offset, retries=4):
    for attempt in range(retries):
        try:
            r = requests.get(BASE, params={"format": "json", "rows": page_size, "os": offset}, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            wait = 2 ** attempt  # 1s, 2s, 4s, 8s
            print(f"offset {offset} attempt {attempt+1} failed ({e}), retrying in {wait}s")
            time.sleep(wait)
    return None  # give up on this page after 4 tries

failed_pages = 0
while True:
    payload = fetch_page(offset)
    if payload is None:
        failed_pages += 1
        if failed_pages >= 3:          # 3 dead pages in a row -> stop, keep what we have
            print(f"Stopping at offset {offset}; keeping {len(rows)} projects")
            break
        offset += page_size            # skip the bad page and continue
        continue
    failed_pages = 0
    projects = payload.get("projects", {})
    if not projects:
        break
    rows.extend(projects.values())
    total = int(payload.get("total", 0))
    print(f"{len(rows)} / {total}")
    offset += page_size
    if offset >= total or offset > 25000:
        break
    time.sleep(0.5)                    # be polite; reduces 500s noticeably

print(f"Fetched {len(rows)} projects")



# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE VOLUME IF NOT EXISTS worldbank.bronze.landing;

# COMMAND ----------

import json
from pyspark.sql import functions as F

# 1. Write the fetched projects as JSON Lines into a UC Volume
landing_path = "/Volumes/worldbank/bronze/landing/projects.jsonl"
with open(landing_path, "w") as f:
    for p in rows:
        f.write(json.dumps(p) + "\n")

# 2. Read it back with Spark (serverless-safe, schema auto-inferred)
df = spark.read.json(landing_path)

# 3. Write Bronze Delta table
(df.withColumn("_ingested_at", F.current_timestamp())
   .write.format("delta")
   .mode("overwrite")
   .option("overwriteSchema", "true")
   .saveAsTable("worldbank.bronze.projects_raw"))

print(f"Bronze table written: {spark.table('worldbank.bronze.projects_raw').count()} rows")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT Count(boardapprovaldate), Count(closingdate)
# MAGIC FROM worldbank.bronze.projects_raw
# MAGIC ;

# COMMAND ----------

from pyspark.sql import functions as F

silver = spark.table("worldbank.silver.projects")
counts = silver.select([F.count(c).alias(c) for c in silver.columns])
display(counts)