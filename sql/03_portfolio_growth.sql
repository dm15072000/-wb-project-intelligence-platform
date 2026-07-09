-- =====================================================================
-- 03_portfolio_growth.sql
-- Dashboard: "World Bank Portfolio Analytics"
-- Widget: "Portfolio Growth over Time"
-- =====================================================================

SELECT YEAR(approval_date) AS approval_year,
       COUNT(*) AS projects_approved,
       ROUND(SUM(commitment_usd) / 1e9, 2) AS total_commitment_billions
FROM worldbank.silver.projects
WHERE approval_date IS NOT NULL
GROUP BY YEAR(approval_date)
ORDER BY approval_year;
