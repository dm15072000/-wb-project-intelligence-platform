# Databricks notebook source
# MAGIC %sql
# MAGIC SHOW TABLES IN system.lakeflow;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT job_id, run_id, run_name, result_state,
# MAGIC        period_start_time, period_end_time,
# MAGIC        ROUND((unix_timestamp(period_end_time) - unix_timestamp(period_start_time))/60, 1) AS duration_mins
# MAGIC FROM system.lakeflow.job_run_timeline
# MAGIC WHERE job_id = '249553591127197'
# MAGIC ORDER BY period_start_time;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT job_id, run_id, task_key, result_state,
# MAGIC        period_start_time, period_end_time,
# MAGIC        ROUND((unix_timestamp(period_end_time) - unix_timestamp(period_start_time))/60, 1) AS duration_mins
# MAGIC FROM system.lakeflow.job_task_run_timeline
# MAGIC WHERE job_id = '249553591127197'
# MAGIC ORDER BY period_start_time;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT job_id, name, creator_id, create_time
# MAGIC FROM system.lakeflow.jobs
# MAGIC WHERE job_id = '249553591127197';

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT statement_text, executed_by, 
# MAGIC        total_duration_ms, execution_status, start_time
# MAGIC FROM system.query.history
# MAGIC WHERE executed_by = 'agrawaldhruv2000@gmail.com'
# MAGIC   AND start_time > '2026-07-08'
# MAGIC ORDER BY start_time DESC
# MAGIC LIMIT 20;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT event_time, user_identity.email, 
# MAGIC        service_name, action_name
# MAGIC FROM system.access.audit
# MAGIC WHERE event_time > '2026-07-08'
# MAGIC ORDER BY event_time DESC
# MAGIC LIMIT 20;

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW COLUMNS IN system.serving.endpoint_usage;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT served_entity_id,
# MAGIC        COUNT(*) AS total_requests,
# MAGIC        SUM(input_token_count) AS total_input_tokens,
# MAGIC        SUM(output_token_count) AS total_output_tokens,
# MAGIC        MIN(request_time) AS first_request,
# MAGIC        MAX(request_time) AS last_request
# MAGIC FROM system.serving.endpoint_usage
# MAGIC GROUP BY served_entity_id
# MAGIC ORDER BY total_requests DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'bronze' AS layer, operation, timestamp, userName
# MAGIC FROM (DESCRIBE HISTORY worldbank.bronze.projects_raw)
# MAGIC UNION ALL
# MAGIC SELECT 'silver', operation, timestamp, userName
# MAGIC FROM (DESCRIBE HISTORY worldbank.silver.projects)
# MAGIC UNION ALL
# MAGIC SELECT 'gold', operation, timestamp, userName
# MAGIC FROM (DESCRIBE HISTORY worldbank.gold.commitments_by_sector)
# MAGIC ORDER BY timestamp DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM system.serving.endpoint_usage
# MAGIC WHERE endpoint_name = 'wb-project-success'
# MAGIC ORDER BY window_start_time DESC
# MAGIC LIMIT 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM system.billing.usage
# MAGIC WHERE usage_date >= '2026-07-06'
# MAGIC ORDER BY usage_date DESC
# MAGIC LIMIT 20;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT operation, timestamp, userName, 
# MAGIC        operationParameters, operationMetrics
# MAGIC FROM (DESCRIBE HISTORY worldbank.bronze.projects_raw)
# MAGIC ORDER BY timestamp DESC;