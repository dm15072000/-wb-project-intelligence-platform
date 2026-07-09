-- =====================================================================
-- 01_catalog_setup.sql
-- Unity Catalog bootstrap: catalog, medallion schemas, and volumes
-- Source: 00_setup notebook
-- =====================================================================

CREATE CATALOG IF NOT EXISTS worldbank;
USE CATALOG worldbank;

CREATE SCHEMA IF NOT EXISTS bronze COMMENT 'Raw ingested data, as-is';
CREATE SCHEMA IF NOT EXISTS silver COMMENT 'Cleaned, typed, deduplicated';
CREATE SCHEMA IF NOT EXISTS gold   COMMENT 'Aggregated, ML-ready, BI-ready';
CREATE SCHEMA IF NOT EXISTS docs   COMMENT 'Unstructured document pipeline';
CREATE SCHEMA IF NOT EXISTS ml     COMMENT 'Models, features, evaluation results';

-- Volume for raw ICR PDFs consumed by the RAG ingestion pipeline
CREATE VOLUME IF NOT EXISTS worldbank.docs.raw_pdfs;

-- Volume for landing raw JSON pulled from the World Bank Projects API
-- (Source: 01_ingest_projects.py)
CREATE VOLUME IF NOT EXISTS worldbank.bronze.landing;
