# Databricks notebook source
bronze = spark.table("worldbank.bronze.projects_raw")
bronze.printSchema()

# COMMAND ----------

from pyspark.sql import functions as F

silver = spark.table("worldbank.silver.projects")

features = (silver
    .filter(F.col("status").isin("Closed", "Dropped"))
    .withColumn("label", (F.col("status") == "Closed").cast("int"))
    .withColumn("approval_year", F.year("approval_date"))
    .withColumn("planned_duration_days", F.datediff("closing_date", "approval_date"))
    # Missingness flags — carry the signal explicitly
    .withColumn("has_approval_date", F.col("approval_date").isNotNull().cast("int"))
    .withColumn("has_closing_date",  F.col("closing_date").isNotNull().cast("int"))
    # Impute: fill missing numerics with sentinel values
    .fillna({"approval_year": 0, "planned_duration_days": -1,
             "region": "Unknown", "primary_sector": "Unknown",
             "lending_instrument": "Unknown"})
    .select("project_id","region","primary_sector","lending_instrument",
            "commitment_usd","approval_year","planned_duration_days",
            "has_approval_date","has_closing_date","label")
    .na.drop(subset=["commitment_usd"])   # only require commitment now — both classes have it
)

features.write.format("delta").mode("overwrite").option("overwriteSchema","true") \
    .saveAsTable("worldbank.ml.project_features")

display(spark.sql("""
    SELECT label, COUNT(*) AS n,
           ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
    FROM worldbank.ml.project_features GROUP BY label
"""))

# COMMAND ----------

from pyspark.sql import functions as F

silver = spark.table("worldbank.silver.projects")

step1 = silver.filter(F.col("status").isin("Closed", "Dropped"))
print("after status filter:")
step1.groupBy("status").count().show()

step2 = (step1
    .withColumn("label", (F.col("status") == "Closed").cast("int"))
    .withColumn("approval_year", F.year("approval_date"))
    .withColumn("planned_duration_days", F.datediff("closing_date", "approval_date"))
    .withColumn("has_approval_date", F.col("approval_date").isNotNull().cast("int"))
    .withColumn("has_closing_date",  F.col("closing_date").isNotNull().cast("int"))
    .fillna({"approval_year": 0, "planned_duration_days": -1,
             "region": "Unknown", "primary_sector": "Unknown",
             "lending_instrument": "Unknown"}))
print("after feature engineering:")
step2.groupBy("label").count().show()

step3 = step2.na.drop(subset=["commitment_usd"])
print("after na.drop on commitment_usd:")
step3.groupBy("label").count().show()

print("what's currently in the saved table:")
spark.table("worldbank.ml.project_features").groupBy("label").count().show()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT status, COUNT(*) AS n
# MAGIC FROM worldbank.silver.projects
# MAGIC GROUP BY status
# MAGIC ORDER BY n DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT status,
# MAGIC        COUNT(*) AS total,
# MAGIC        COUNT(approval_date) AS has_approval_date,
# MAGIC        COUNT(CASE WHEN commitment_usd IS NOT NULL THEN 1 END) AS has_commitment
# MAGIC FROM worldbank.silver.projects
# MAGIC WHERE status NOT IN ('Closed')
# MAGIC GROUP BY status;

# COMMAND ----------

