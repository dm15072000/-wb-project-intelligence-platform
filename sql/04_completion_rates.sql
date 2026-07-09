-- =====================================================================
-- 04_completion_rates.sql
-- Dashboard: "World Bank Portfolio Analytics"
-- Widget: "Completion Rate by Region"
-- Plus supporting completion-rate queries used in notebooks 02 and 04
-- =====================================================================

-- Dashboard widget: completion rate by region (regions with >50 projects)
SELECT region,
       COUNT(*) AS n_projects,
       ROUND(AVG(label) * 100, 1) AS completion_rate_pct
FROM worldbank.ml.project_features
WHERE region != 'Unknown'
GROUP BY region
HAVING COUNT(*) > 50
ORDER BY completion_rate_pct DESC;

-- Completion rate + average commitment by region, for a given sector
-- (Source: 04_ingest_documents.py — pairs with the RAG narrative view
--  to compare "what the numbers say" against "what the ICRs say")
SELECT region,
       COUNT(*) AS n_projects,
       ROUND(AVG(label) * 100, 1) AS completion_rate_pct,
       ROUND(AVG(commitment_usd) / 1e6, 1) AS avg_commitment_musd
FROM worldbank.ml.project_features
WHERE primary_sector = 'Social Protection' AND region != 'Unknown'
GROUP BY region
ORDER BY completion_rate_pct;

-- Label balance check (Closed vs. Dropped) used when building the
-- ML feature table (Source: 02_transform_projects.py)
SELECT label, COUNT(*) AS n,
       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
FROM worldbank.ml.project_features
GROUP BY label;
