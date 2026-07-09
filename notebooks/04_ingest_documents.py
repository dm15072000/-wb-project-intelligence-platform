# Databricks notebook source
# MAGIC %pip install pypdf

# COMMAND ----------

# --- Idempotency guard: skip re-ingestion if documents already processed ---
FORCE_REFRESH = False   # flip to True (or parameterize) to re-ingest from scratch

def table_has_rows(name):
    try:
        return spark.table(name).count() > 0
    except Exception:       # table doesn't exist yet -> first run
        return False

SKIP_INGEST = (not FORCE_REFRESH) and table_has_rows("worldbank.docs.icr_embeddings")
print(f"SKIP_INGEST = {SKIP_INGEST}")

# COMMAND ----------

if SKIP_INGEST:
    print("Embeddings already exist — skipping PDF download.")
else:
    import requests

    search_url = "https://search.worldbank.org/api/v3/wds"
    params = {"format": "json", "docty": "Implementation Completion and Results Report",
            "rows": 30, "fl": "id,pdfurl,projectid,docdt"}
    docs = requests.get(search_url, params=params, timeout=60).json().get("documents", {})

    vol = "/Volumes/worldbank/docs/raw_pdfs"
    meta = []
    for k, d in docs.items():
        pdf_url = d.get("pdfurl")
        pid = d.get("projectid", "unknown")
        if not pdf_url or pid == "unknown":
            continue
        path = f"{vol}/{pid}_{k}.pdf"
        try:
            pdf = requests.get(pdf_url, timeout=90)
            pdf.raise_for_status()
            with open(path, "wb") as f:
                f.write(pdf.content)
            meta.append({"doc_id": k, "project_id": pid, "path": path, "doc_date": d.get("docdt")})
            print(f"saved {pid}")
        except Exception as e:
            print(f"skip {k}: {e}")

    print(f"\n{len(meta)} PDFs downloaded")
    spark.createDataFrame(meta).write.format("delta").mode("overwrite") \
        .saveAsTable("worldbank.docs.icr_metadata")

# COMMAND ----------

from pypdf import PdfReader

def pdf_to_chunks(path, chunk_size=1500, overlap=200):
    try:
        text = "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    except Exception as e:
        print(f"parse failed {path}: {e}")
        return []
    text = " ".join(text.split())  # normalize whitespace
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i+chunk_size])
        i += chunk_size - overlap
    return chunks

meta_rows = spark.table("worldbank.docs.icr_metadata").collect()
rows = []
for m in meta_rows:
    cs = pdf_to_chunks(m.path)
    for j, c in enumerate(cs):
        rows.append({"doc_id": m.doc_id, "project_id": m.project_id,
                     "chunk_id": f"{m.doc_id}_{j}", "text": c})
    print(f"{m.project_id}: {len(cs)} chunks")

spark.createDataFrame(rows).write.format("delta").mode("overwrite") \
     .saveAsTable("worldbank.docs.icr_chunks")
print(f"\n{len(rows)} total chunks")

# COMMAND ----------

if SKIP_INGEST:
    print("Skipping embedding generation.")
else:
    spark.sql("""
        CREATE OR REPLACE TABLE worldbank.docs.icr_embeddings AS
        SELECT chunk_id, project_id, text,
               ai_query('databricks-gte-large-en', text) AS embedding
        FROM worldbank.docs.icr_chunks
    """)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT chunk_id, size(embedding) AS dims, text
# MAGIC FROM worldbank.docs.icr_embeddings
# MAGIC LIMIT 3;

# COMMAND ----------

import mlflow
import numpy as np
from databricks.sdk import WorkspaceClient
import requests

HOST = "https://dbc-0db6d6cc-81dd.cloud.databricks.com"
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

def llm_chat(prompt, max_tokens=600):
    r = requests.post(
        f"{HOST}/serving-endpoints/databricks-meta-llama-3-3-70b-instruct/invocations",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens},
        timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]
emb = spark.table("worldbank.docs.icr_embeddings").toPandas()
matrix = np.vstack(emb["embedding"].values)
norms = np.linalg.norm(matrix, axis=1)
w = WorkspaceClient()

def embed_query(q):
    r = w.serving_endpoints.query(name="databricks-gte-large-en", input=q)
    d = r.data[0]
    return np.array(d["embedding"] if isinstance(d, dict) else d.embedding)

@mlflow.trace
def retrieve(question, k=5):
    qv = embed_query(question)
    sims = matrix @ qv / (norms * np.linalg.norm(qv))
    top = emb.iloc[np.argsort(-sims)[:k]]
    return top[["project_id", "text"]].to_dict("records")

@mlflow.trace
def rag_answer(question, k=5):
    chunks = retrieve(question, k)
    context = "\n\n---\n\n".join(f"[Project {c['project_id']}]\n{c['text']}" for c in chunks)
    prompt = f"""You are an analyst of World Bank project evaluations. Answer the question using ONLY the context below, drawn from Implementation Completion Reports. Cite project IDs in your answer. If the context is insufficient, say so.

Context:
{context}

Question: {question}"""
    answer = llm_chat(prompt)
    return {"answer": answer,
            "sources": sorted({c["project_id"] for c in chunks})}
result = rag_answer("What are common causes of delays or underperformance in World Bank projects?")
print(result["answer"])
print("\nSources:", result["sources"])

# COMMAND ----------

question_sector = "Social Protection"   # a sector present in both your tables and the ICRs

# --- Structured half: what do the numbers say? ---
stats = spark.sql(f"""
    SELECT region,
           COUNT(*) AS n_projects,
           ROUND(AVG(label) * 100, 1) AS completion_rate_pct,
           ROUND(AVG(commitment_usd) / 1e6, 1) AS avg_commitment_musd
    FROM worldbank.ml.project_features
    WHERE primary_sector = '{question_sector}' AND region != 'Unknown'
    GROUP BY region
    ORDER BY completion_rate_pct
""").toPandas()
print("=== Structured view: completion rates ===")
print(stats.to_string(index=False))

# --- Unstructured half: what do the evaluations say? ---
result = rag_answer(f"What challenges and lessons are reported in {question_sector} projects?")
print("\n=== Narrative view: evidence from ICRs ===")
print(result["answer"])
print("\nSources:", result["sources"])
