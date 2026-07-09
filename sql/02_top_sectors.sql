-- =====================================================================
-- 02_top_sectors.sql
-- Dashboard: "World Bank Portfolio Analytics"
-- Widgets: "Top Sectors by Region" and "Top 15 countries by total commitment"
-- =====================================================================

-- Top 3 sectors by total commitment, within each region
SELECT region, primary_sector, total_commitment_usd, project_count
FROM worldbank.gold.commitments_by_sector
WHERE region IS NOT NULL AND primary_sector IS NOT NULL
QUALIFY ROW_NUMBER() OVER (PARTITION BY region ORDER BY total_commitment_usd DESC) <= 3
ORDER BY region, total_commitment_usd DESC;

-- Top 15 countries by total commitment (USD billions)
SELECT country,
       COUNT(*) AS projects,
       ROUND(SUM(commitment_usd) / 1e9, 2) AS total_commitment_billions
FROM worldbank.silver.projects
WHERE country IS NOT NULL
GROUP BY country
ORDER BY total_commitment_billions DESC
LIMIT 15;
