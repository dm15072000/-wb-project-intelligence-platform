# World Bank Project Intelligence Platform

An end-to-end data intelligence platform built on **Databricks** and **Unity Catalog**, covering the full lifecycle of World Bank project portfolio data: ingestion from the public World Bank Projects API, a bronze/silver/gold medallion pipeline, a machine learning model that predicts project completion likelihood (served live via **Databricks Model Serving**), a **Retrieval-Augmented Generation (RAG)** system over project evaluation reports using **Foundation Model APIs**, a **Databricks SQL** portfolio analytics dashboard, and system-table-driven audit/observability queries.

The project is intentionally built the way a production data platform team would build it — governed through Unity Catalog, versioned and tracked through MLflow, and observable through Databricks' `system` schema — rather than as a one-off notebook demo.

## Why this project

World Bank operations generate large volumes of structured project data (commitments, sectors, regions, lending instruments) alongside unstructured narrative data (Implementation Completion and Results Reports, or ICRs). This platform brings both together: a classifier that scores the likelihood a project reaches "Closed" rather than "Dropped" status, and a document QA system that lets an analyst ask natural-language questions and get answers grounded in and cited against real ICR text — with both views cross-checked against each other (e.g., "what do the numbers say about Social Protection projects in this region" vs. "what do the evaluation reports say").

## Architecture overview

```
World Bank Projects API ─┐
                          ▼
                  ┌───────────────┐
                  │  BRONZE       │  worldbank.bronze.projects_raw
                  │  raw, as-is   │  (Delta, ingested via UC Volume landing zone)
                  └───────┬───────┘
                          ▼
                  ┌───────────────┐
                  │  SILVER       │  worldbank.silver.projects
                  │  cleaned,     │  (typed, deduplicated, standardized schema)
                  │  typed        │
                  └───────┬───────┘
                          ▼
              ┌───────────┴────────────┐
              ▼                        ▼
      ┌───────────────┐        ┌───────────────┐
      │  GOLD          │        │  ML            │  worldbank.ml.project_features
      │  BI-ready      │        │  feature store │  worldbank.ml.project_success_model (UC Registry)
      │  aggregates    │        │  + MLflow      │  → Model Serving endpoint: wb-project-success
      └───────┬────────┘        └───────┬────────┘
              ▼                         ▼
      Databricks SQL Dashboard   Real-time predictions
      "World Bank Portfolio        (REST invocations against
       Analytics"                  the live serving endpoint)

World Bank ICR PDFs ─┐
                     ▼
             ┌───────────────┐
             │  DOCS schema  │  worldbank.docs.raw_pdfs (Volume)
             │  chunk + embed│  worldbank.docs.icr_chunks
             │               │  worldbank.docs.icr_embeddings
             └───────┬───────┘  (embeddings via Foundation Model API
                     ▼           `databricks-gte-large-en`, ai_query())
             Delta-native cosine similarity retrieval
                     ▼
          `databricks-meta-llama-3-3-70b-instruct`
             grounded, cited RAG answers
                     ▼
        LLM-as-judge evaluation → worldbank.ml.rag_eval_results

system.* tables ──▶ Governance / lineage / cost / model-serving audit queries
```

See [`docs/architecture.png`](docs/architecture.png) for a visual diagram *(add your own — see Setup step 7)*.

## Pipeline components

| # | Notebook | Layer | What it does |
|---|----------|-------|---------------|
| 1 | [`01_ingest_projects.py`](notebooks/01_ingest_projects.py) | Bronze | Pulls project records from the public [World Bank Projects API](https://search.worldbank.org/api/v2/projects) with retry/backoff, lands them as JSON in a Unity Catalog Volume, and writes the Bronze Delta table `worldbank.bronze.projects_raw`. |
| 2 | [`02_transform_projects.py`](notebooks/02_transform_projects.py) | Silver → ML features | Cleans and standardizes Silver project records, engineers a feature table (`worldbank.ml.project_features`) with leakage-aware feature selection (date-derived signals excluded from the "honest" model), missingness flags, and imputation. |
| 3 | [`03_train_model.py`](notebooks/03_train_model.py) | ML / MLflow | Trains a `RandomForestClassifier` predicting project completion (`Closed` vs. `Dropped`), compares an all-features run against a leakage-free run, logs params/metrics/models to **MLflow**, registers the model to **Unity Catalog** as `worldbank.ml.project_success_model`, and calls the live **`wb-project-success`** Model Serving endpoint for real-time inference. Includes feature-importance and confusion-matrix analysis. |
| 4 | [`04_ingest_documents.py`](notebooks/04_ingest_documents.py) | Docs / RAG | Downloads Implementation Completion and Results Report (ICR) PDFs, chunks them, generates embeddings via the **Foundation Model API** (`databricks-gte-large-en` through `ai_query`), and persists everything as Delta tables. Implements retrieval via **cosine similarity computed directly against the Delta-backed embedding table** (no external vector database) and generation via `databricks-meta-llama-3-3-70b-instruct`, with `mlflow.trace` instrumentation. |
| 5 | [`05_evaluate.py`](notebooks/05_evaluate.py) | Evaluation | Runs formal `mlflow.evaluate()` classifier evaluation against the registered model, then evaluates the RAG pipeline against a golden question set using an **LLM-as-judge** pattern (relevance + correctness scoring), logging results to MLflow and to `worldbank.ml.rag_eval_results`. |
| 6 | [`06_audit_logs.py`](notebooks/06_audit_logs.py) | Governance | Queries Databricks `system` schema tables — job/task run history, query history, Unity Catalog access audit logs, Model Serving usage/token counts, billing usage, and Delta `DESCRIBE HISTORY` lineage across bronze/silver/gold — for operational and governance visibility. |

## Model Serving

The classifier trained in notebook 3 is registered to Unity Catalog (`worldbank.ml.project_success_model`) and deployed to a **live Databricks Model Serving endpoint named `wb-project-success`**. Notebook 3 invokes it directly over REST with a Databricks personal access token, demonstrating the full loop from training → registry → real-time serving → inference. See [`docs/mlflow_comparison.png`](docs/mlflow_comparison.png) for the MLflow run comparison between the full-feature and leakage-free models.

## RAG pipeline

Rather than standing up a separate vector database, retrieval is implemented natively against Delta: embeddings generated through the Foundation Model API are stored as array columns in `worldbank.docs.icr_embeddings`, loaded into memory, and scored against a query embedding using vectorized cosine similarity (`matrix @ query_vector / norms`). This keeps the whole pipeline inside Unity Catalog governance boundaries and avoids extra infrastructure. Generation is grounded strictly in retrieved ICR text, with project IDs cited in every answer, and the whole pipeline is evaluated with an LLM-as-judge rubric — see [`docs/rag_eval_results.png`](docs/rag_eval_results.png).

## Dashboard

The **"World Bank Portfolio Analytics"** Databricks SQL / Lakeview dashboard (queries extracted to [`sql/`](sql/)) surfaces:
- Top sectors by commitment within each region
- Portfolio growth over time (projects approved and commitment volume by year)
- Completion rate by region
- Top 15 countries by total commitment

See [`docs/dashboard.png`](docs/dashboard.png). The underlying pipeline is orchestrated as a Databricks Workflow — see [`docs/workflow_dag.png`](docs/workflow_dag.png) for the job DAG.

## Repository structure

```
wb-project-intelligence-platform/
├── README.md
├── .gitignore
├── notebooks/
│   ├── 01_ingest_projects.py       # Bronze: API ingestion
│   ├── 02_transform_projects.py    # Silver → ML feature engineering
│   ├── 03_train_model.py           # MLflow training + UC registry + serving call
│   ├── 04_ingest_documents.py      # RAG: PDF ingestion, embeddings, retrieval
│   ├── 05_evaluate.py              # Model evaluation + RAG LLM-as-judge eval
│   └── 06_audit_logs.py            # Governance / system-table audit queries
├── docs/
│   ├── architecture.png            # Architecture diagram
│   ├── dashboard.png               # Databricks SQL dashboard screenshot
│   ├── mlflow_comparison.png       # MLflow run comparison screenshot
│   ├── rag_eval_results.png        # RAG LLM-as-judge evaluation results
│   └── workflow_dag.png            # Databricks Workflow job DAG
└── sql/
    ├── 01_catalog_setup.sql        # Unity Catalog + schema + volume bootstrap
    ├── 02_top_sectors.sql          # Dashboard: top sectors / top countries
    ├── 03_portfolio_growth.sql     # Dashboard: portfolio growth over time
    ├── 04_completion_rates.sql     # Dashboard + supporting completion-rate queries
    └── 05_audit_queries.sql        # Governance, lineage, cost/usage queries
```

## Tech stack

| Layer | Technology |
|---|---|
| Storage & governance | Delta Lake, Unity Catalog (catalogs, schemas, volumes, lineage) |
| Compute | Databricks (PySpark, Spark SQL) |
| ML training & tracking | scikit-learn, MLflow (experiments, model registry, `mlflow.evaluate`, `mlflow.trace`) |
| Model serving | Databricks Model Serving (`wb-project-success` endpoint) |
| GenAI / RAG | Databricks Foundation Model APIs (`databricks-gte-large-en` embeddings, `databricks-meta-llama-3-3-70b-instruct` chat), `ai_query()`, Delta-native cosine similarity |
| BI | Databricks SQL / Lakeview dashboards |
| Observability | Databricks `system` schema (`lakeflow`, `query.history`, `access.audit`, `serving.endpoint_usage`, `billing.usage`) |

## Setup / reproduction

1. **Prerequisites**: a Databricks workspace with Unity Catalog enabled, a SQL warehouse, cluster/serverless compute, and access to Databricks Foundation Model APIs.
2. **Bootstrap the catalog**: run [`sql/01_catalog_setup.sql`](sql/01_catalog_setup.sql) to create the `worldbank` catalog, its `bronze` / `silver` / `gold` / `docs` / `ml` schemas, and the required volumes.
3. **Import the notebooks**: import everything under `notebooks/` into your workspace (e.g. `databricks workspace import-dir ./notebooks /Workspace/Users/<you>/wb-project-intelligence-platform`).
4. **Run the pipeline in order**: execute notebooks `01` → `06` sequentially. Notebook 4 requires `pypdf` (`%pip install pypdf`, included in the notebook).
5. **Deploy the model**: after notebook 3 registers `worldbank.ml.project_success_model` to Unity Catalog, create a Model Serving endpoint named `wb-project-success` from the registered model version (Serving → Create serving endpoint).
6. **Build the dashboard**: recreate the four widgets in a new Databricks SQL / Lakeview dashboard using the queries in `sql/02_top_sectors.sql`, `sql/03_portfolio_growth.sql`, and `sql/04_completion_rates.sql`, pointed at your SQL warehouse.
7. **Screenshots**: drop your own architecture diagram and dashboard/evaluation screenshots into `docs/` using the filenames referenced above (`architecture.png` is still needed).

## Notes on this repository

This repo was assembled from live notebooks exported directly out of the author's Databricks workspace — the ingestion logic, feature engineering, model training, RAG retrieval, evaluation, and audit queries reflect an actual working pipeline, not illustrative pseudocode.
