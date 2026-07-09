# Databricks notebook source
import pandas as pd

pdf = spark.table("worldbank.ml.project_features").toPandas()
print(pdf.shape)          
print(pdf["label"].value_counts(normalize=True))

# COMMAND ----------

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score

mlflow.set_registry_uri("databricks-uc")   # register models into Unity Catalog
mlflow.set_experiment("/Users/agrawaldhruv2000@gmail.com/wb_project_success")

FEATURES = ["region","primary_sector","lending_instrument",
            "commitment_usd","approval_year","planned_duration_days",
            "has_approval_date","has_closing_date"]
CAT = ["region","primary_sector","lending_instrument"]

X = pdf[FEATURES]
y = pdf["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

pre = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), CAT)],
    remainder="passthrough")

with mlflow.start_run(run_name="rf_all_features"):
    model = Pipeline([
        ("pre", pre),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=10,
                                      class_weight="balanced", random_state=42))])
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    auc, f1 = roc_auc_score(y_test, proba), f1_score(y_test, preds)
    mlflow.log_params({"n_estimators": 200, "max_depth": 10,
                       "class_weight": "balanced", "features": "all"})
    mlflow.log_metrics({"auc": auc, "f1": f1})
    mlflow.sklearn.log_model(model, "model",
        input_example=X_train.head(3),
        registered_model_name="worldbank.ml.project_success_model")
    print(f"AUC: {auc:.3f} | F1: {f1:.3f}")

# COMMAND ----------

HONEST_FEATURES = ["region","primary_sector","lending_instrument","commitment_usd"]
CAT2 = ["region","primary_sector","lending_instrument"]

X2 = pdf[HONEST_FEATURES]
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y, test_size=0.2, random_state=42, stratify=y)

pre2 = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), CAT2)],
    remainder="passthrough")

with mlflow.start_run(run_name="rf_no_leakage"):
    model2 = Pipeline([
        ("pre", pre2),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=10,
                                      class_weight="balanced", random_state=42))])
    model2.fit(X2_train, y2_train)
    proba2 = model2.predict_proba(X2_test)[:, 1]

    auc2 = roc_auc_score(y2_test, proba2)
    f1_2 = f1_score(y2_test, model2.predict(X2_test))
    mlflow.log_params({"n_estimators": 200, "max_depth": 10,
                       "class_weight": "balanced", "features": "no_date_leakage"})
    mlflow.log_metrics({"auc": auc2, "f1": f1_2})
    mlflow.sklearn.log_model(model2, "model", input_example=X2_train.head(3),
    registered_model_name="worldbank.ml.project_success_model")
    print(f"Honest model — AUC: {auc2:.3f} | F1: {f1_2:.3f}")

# COMMAND ----------

import requests

url = "https://dbc-0db6d6cc-81dd.cloud.databricks.com/serving-endpoints/wb-project-success/invocations"
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

payload = {"dataframe_records": [
    {"region": "Eastern and Southern Africa", "primary_sector": "Rural and Inter-Urban Roads",
     "lending_instrument": "Investment Project Financing", "commitment_usd": 150_000_000.0},
    {"region": "Latin America and Caribbean", "primary_sector": "Central Government (Central Agencies)",
     "lending_instrument": "Development Policy Lending", "commitment_usd": 5_000_000.0},
]}

r = requests.post(url, headers={"Authorization": f"Bearer {token}"}, json=payload)
print(r.json())

# COMMAND ----------

import pandas as pd
import numpy as np

# 1. What features drive the predictions?
rf = model2.named_steps["rf"]
ohe = model2.named_steps["pre"].named_transformers_["cat"]
feature_names = list(ohe.get_feature_names_out(CAT2)) + ["commitment_usd"]

importances = pd.DataFrame({
    "feature": feature_names,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)
display(importances.head(15))

# COMMAND ----------

# 2. Where does the model make mistakes? (confusion matrix at the default threshold)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y2_test, model2.predict(X2_test)))
print(classification_report(y2_test, model2.predict(X2_test), target_names=["Dropped","Closed"]))

# COMMAND ----------

# 3. Sanity-check its opinions: score some contrasting hypotheticals
probe = pd.DataFrame([
    {"region": "Western and Central Africa", "primary_sector": "Other Transportation",
     "lending_instrument": "Investment Project Financing", "commitment_usd": 500_000_000.0},
    {"region": "Europe and Central Asia", "primary_sector": "Banking Institutions",
     "lending_instrument": "Development Policy Lending", "commitment_usd": 20_000_000.0},
    {"region": "South Asia", "primary_sector": "Social Protection",
     "lending_instrument": "Investment Project Financing", "commitment_usd": 80_000_000.0},
])
probe["p_success"] = model2.predict_proba(probe)[:, 1].round(3)
display(probe)

# COMMAND ----------

# MAGIC %md
# MAGIC