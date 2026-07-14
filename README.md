# World Bank Project Intelligence Platform

> **Live demo:** https://wbintelligence2-7474649658971091.aws.databricksapps.com
> Try asking: *"What causes delays in World Bank projects?"* or *"What lessons are reported in Social Protection projects?"*

**22,729 projects · 30 ICR PDFs · 3,142 chunks · AUC 0.892 · RAG relevance 4.9/5.0 · 10 Databricks services**

---

## What this is

The World Bank publishes two kinds of data that nobody combines: structured project records (sector, region, commitment size, outcome) and unstructured evaluation reports (ICRs — documents written by field teams after a project closes, describing what worked and what didn't). This platform ingests both, links them by project ID, and lets you query them together.

The result: you can ask "why do Social Protection projects underperform in Europe?" and get a quantitative answer from Delta tables alongside narrative evidence retrieved from the actual evaluation reports — in one response.

Built entirely on Databricks Free Edition, covering all 10 core services end-to-end.

---

## Architecture

```
World Bank Projects API          World Bank Documents & Reports API
(22,729 projects)                (30 ICR PDFs)
        │                                │
        ▼                                ▼
  worldbank.bronze                UC Volume: docs/raw_pdfs
  (raw JSON, Delta)               worldbank.docs.icr_chunks
        │                         worldbank.docs.icr_embeddings
        ▼                         (GTE Large EN, 1024-dim)
  worldbank.silver                        │
  (typed, deduped)                        │
        │                                 │
        ▼                                 ▼
  worldbank.gold              cosine similarity retrieval
  worldbank.ml.project_features           │
        │                                 ▼
        ▼                        Llama 3.3 70B generation
  RandomForest classifier                 │
  MLflow tracked                          ▼
  UC model registry              LLM-as-judge evaluation
  Model Serving endpoint         worldbank.ml.rag_eval_results
        │                                 │
        └──────────────┬──────────────────┘
                       ▼
              Databricks Workflows
              (5-task scheduled DAG)
                       │
                       ▼
              system.* audit tables
```

---

## The ML story

Training a classifier to predict whether a project closes successfully vs. gets dropped sounds straightforward. It wasn't.

The first model scored AUC 0.999 — which should always be a red flag, not a celebration. Every dropped project in the dataset was cancelled before board approval, so none had an approval date. The `has_approval_date` flag I'd engineered as a safety feature was perfectly separating the two classes — the model wasn't learning risk signals, it was reading a bookkeeping artifact.

I trained a second model without the date-derived features. AUC dropped to 0.892. That's the deployed model, registered as `@champion` in Unity Catalog. The leaky model is version 1; the honest model is version 3. The MLflow comparison view shows both runs side by side.

**The rule I now follow:** any AUC above 0.95 on a real-world business outcome warrants a leakage investigation before you trust it.

| Run | Features | AUC | F1 | Notes |
|---|---|---|---|---|
| `rf_all_features` | All incl. date flags | 0.999 | 0.995 | Leaky — not deployed |
| `rf_no_leakage` | Region, sector, instrument, commitment | 0.892 | 0.967 | Champion — deployed |

---

## The RAG pipeline

30 ICR PDFs → 3,142 chunks → GTE Large EN embeddings stored in a Delta table → cosine similarity retrieval → Llama 3.3 70B generation.

No external vector database. Embeddings live in `worldbank.docs.icr_embeddings` as array columns in Delta, loaded into a NumPy matrix at app startup, and scored with a matrix multiply. At 3,142 chunks this is sub-second. At millions of chunks the right answer is Databricks Vector Search with a Delta Sync index — migration would be one configuration change.

Evaluated with LLM-as-judge: 8 hand-written question/reference-answer pairs, Llama 3.3 70B scoring each system answer on relevance and correctness (1–5). Mean relevance: **4.9/5.0**. Mean correctness: **4.4/5.0**. One question scored 2/5 on correctness — the relevant passage was at a chunk boundary. Documented as a known limitation; fix is reducing chunk size from 1,500 to 800 characters.

---

## Databricks services

| Service | Where used |
|---|---|
| **Workspace** | All notebooks, Catalog Explorer |
| **Delta Lake** | Every table — ACID writes, time travel, schema evolution |
| **Unity Catalog** | `worldbank` catalog, 5 schemas, 2 Volumes, model registry |
| **Lakehouse** | Files (PDFs, JSONL) + governed Delta tables on one platform |
| **Databricks SQL** | 4 saved queries, serverless warehouse, published dashboard |
| **MLflow** | 4 tracked runs, experiment comparison, UC model registry |
| **Model Serving** | `wb-project-success` REST endpoint, scale-to-zero |
| **Evaluation** | `mlflow.models.evaluate` + LLM-as-judge with logged metrics |
| **Workflows** | 5-task DAG, two parallel branches, weekly schedule |
| **Audit Logs** | `system.lakeflow`, `system.query.history`, `system.access.audit`, `DESCRIBE HISTORY` |

---

## Notebooks

| Notebook | What it does |
|---|---|
| `01_ingest_projects.py` | Paginated fetch from World Bank Projects API with retry/backoff, lands raw JSON in a UC Volume, writes Bronze Delta table |
| `02_transform_projects.py` | Silver cleaning (two date format fix, commitment casting, deduplication), Gold aggregates, ML feature table with missingness flags |
| `03_train_model.py` | RandomForest training, leakage detection experiment, MLflow tracking, UC model registry, Model Serving endpoint test |
| `04_ingest_documents.py` | ICR PDF download, pypdf parsing, chunking, GTE Large EN embedding via `ai_query()`, Delta storage, RAG pipeline with `@mlflow.trace` |
| `05_evaluate.py` | `mlflow.models.evaluate` classifier eval, LLM-as-judge RAG eval over golden set, results to Delta |
| `06_audit_logs.py` | System table queries: job run history, query history, access audit, serving usage, Delta DESCRIBE HISTORY lineage |

---

## Interesting bugs

Three data problems I hit that aren't in any tutorial:

**Date format split.** The Projects API returns `boardapprovaldate` in ISO format (`2013-06-28T00:00:00Z`) and `closingdate` in US 12-hour format with single-digit months (`6/30/2025 12:00:00 AM`). Same API, two columns, two formats. Found it by querying the exact failing values rather than guessing.

**The vanishing label.** All 1,772 Dropped projects had `approval_date = NULL` — they were cancelled before board approval. `na.drop` on `approval_year` was silently removing the entire failure class. The feature table had 13,517 rows, all label=1. Fixed by relaxing the drop constraint and adding a `has_approval_date` flag (then caught that flag as the leakage source in training).

**SparkContext on serverless.** `sparkContext.parallelize()` works interactively (you skip the cell) but crashes in a Workflow job (every cell runs). Replaced with writing raw JSON to a UC Volume and reading back with `spark.read.json()` — which is actually the better production pattern anyway.

---

## Repo structure

```
wb-project-intelligence-platform/
├── app/
│   ├── app.py              # Streamlit app — 18 functions, live Spark via Databricks SDK
│   ├── app.yml             # Databricks Apps deploy config
│   ├── requirements.txt
│   └── style.css
├── notebooks/
│   ├── 01_ingest_projects.py
│   ├── 02_transform_projects.py
│   ├── 03_train_model.py
│   ├── 04_ingest_documents.py
│   ├── 05_evaluate.py
│   └── 06_audit_logs.py
├── sql/
│   ├── 01_catalog_setup.sql
│   ├── 02_top_sectors.sql
│   ├── 03_portfolio_growth.sql
│   ├── 04_completion_rates.sql
│   └── 05_audit_queries.sql
├── docs/
│   ├── dashboard.png
│   ├── mlflow_comparison.png
│   ├── rag_eval_results.png
│   └── workflow_dag.png
├── README.md
└── .gitignore
```

---

## Stack

Python · PySpark · Databricks · Delta Lake · Unity Catalog · MLflow · scikit-learn · GTE Large EN · Llama 3.3 70B · Streamlit · Databricks Apps · Databricks SDK

---

## Setup

1. Sign up for [Databricks Free Edition](https://databricks.com/learn/free-edition)
2. Run `sql/01_catalog_setup.sql` to create the `worldbank` catalog and schemas
3. Import notebooks from `notebooks/` into your workspace
4. Run `01` → `06` in order (`01` takes ~10 min due to API pagination)
5. Create a Model Serving endpoint named `wb-project-success` pointing at the registered UC model
6. Deploy the app: copy `app/` contents into a workspace folder, deploy via Apps → Create app
