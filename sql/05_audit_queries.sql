-- =====================================================================
-- 05_audit_queries.sql
-- Governance, lineage, and cost/usage observability queries
-- Source: 06_audit_logs.py
-- =====================================================================

-- Job run history for the pipeline orchestration job
SELECT job_id, run_id, run_name, result_state,
       period_start_time, period_end_time,
       ROUND((unix_timestamp(period_end_time) - unix_timestamp(period_start_time))/60, 1) AS duration_mins
FROM system.lakeflow.job_run_timeline
WHERE job_id = '249553591127197'
ORDER BY period_start_time;

-- Per-task run history for the same job
SELECT job_id, run_id, task_key, result_state,
       period_start_time, period_end_time,
       ROUND((unix_timestamp(period_end_time) - unix_timestamp(period_start_time))/60, 1) AS duration_mins
FROM system.lakeflow.job_task_run_timeline
WHERE job_id = '249553591127197'
ORDER BY period_start_time;

-- Job metadata
SELECT job_id, name, creator_id, create_time
FROM system.lakeflow.jobs
WHERE job_id = '249553591127197';

-- Recent query history for the pipeline owner
SELECT statement_text, executed_by,
       total_duration_ms, execution_status, start_time
FROM system.query.history
WHERE executed_by = 'agrawaldhruv2000@gmail.com'
  AND start_time > '2026-07-08'
ORDER BY start_time DESC
LIMIT 20;

-- Unity Catalog access audit log
SELECT event_time, user_identity.email,
       service_name, action_name
FROM system.access.audit
WHERE event_time > '2026-07-08'
ORDER BY event_time DESC
LIMIT 20;

-- Model Serving endpoint usage (token volume, request counts)
SELECT served_entity_id,
       COUNT(*) AS total_requests,
       SUM(input_token_count) AS total_input_tokens,
       SUM(output_token_count) AS total_output_tokens,
       MIN(request_time) AS first_request,
       MAX(request_time) AS last_request
FROM system.serving.endpoint_usage
GROUP BY served_entity_id
ORDER BY total_requests DESC;

-- Usage specifically for the wb-project-success serving endpoint
SELECT * FROM system.serving.endpoint_usage
WHERE endpoint_name = 'wb-project-success'
ORDER BY window_start_time DESC
LIMIT 10;

-- Delta Lake lineage / operation history across the medallion layers
SELECT 'bronze' AS layer, operation, timestamp, userName
FROM (DESCRIBE HISTORY worldbank.bronze.projects_raw)
UNION ALL
SELECT 'silver', operation, timestamp, userName
FROM (DESCRIBE HISTORY worldbank.silver.projects)
UNION ALL
SELECT 'gold', operation, timestamp, userName
FROM (DESCRIBE HISTORY worldbank.gold.commitments_by_sector)
ORDER BY timestamp DESC;

-- Full operation history (with parameters/metrics) for the bronze table
SELECT operation, timestamp, userName,
       operationParameters, operationMetrics
FROM (DESCRIBE HISTORY worldbank.bronze.projects_raw)
ORDER BY timestamp DESC;

-- Platform billing/usage, most recent
SELECT * FROM system.billing.usage
WHERE usage_date >= '2026-07-06'
ORDER BY usage_date DESC
LIMIT 20;
