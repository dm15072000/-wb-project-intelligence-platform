# Databricks notebook source
import mlflow
import pandas as pd

pdf = spark.table("worldbank.ml.project_features").toPandas()
FEATURES = ["region","primary_sector","lending_instrument","commitment_usd"]

from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(
    pdf[FEATURES], pdf["label"], test_size=0.2, random_state=42, stratify=pdf["label"])

eval_df = X_test.copy()
eval_df["label"] = y_test.values

mlflow.set_experiment("/Users/agrawaldhruv2000@gmail.com/wb_project_success")

with mlflow.start_run(run_name="formal_evaluation"):
    results = mlflow.evaluate(
        model="models:/worldbank.ml.project_success_model/3",
        data=eval_df,
        targets="label",
        model_type="classifier",
        evaluators=["default"])
    print({k: round(v, 3) for k, v in results.metrics.items() if isinstance(v, (int, float))})

# COMMAND ----------

import requests
import numpy as np
import mlflow

HOST = "https://dbc-0db6d6cc-81dd.cloud.databricks.com"
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# Load embeddings from Delta (persisted in Phase 6)
emb = spark.table("worldbank.docs.icr_embeddings").toPandas()
matrix = np.vstack(emb["embedding"].values)
norms = np.linalg.norm(matrix, axis=1)

def llm_chat(prompt, max_tokens=600):
    r = requests.post(
        f"{HOST}/serving-endpoints/databricks-meta-llama-3-3-70b-instruct/invocations",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens},
        timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def embed_query(q):
    r = requests.post(
        f"{HOST}/serving-endpoints/databricks-gte-large-en/invocations",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={"input": q}, timeout=60)
    r.raise_for_status()
    return np.array(r.json()["data"][0]["embedding"])

@mlflow.trace
def retrieve(question, k=5):
    qv = embed_query(question)
    sims = matrix @ qv / (norms * np.linalg.norm(qv))
    top = emb.iloc[np.argsort(-sims)[:k]]
    return top[["project_id", "text"]].to_dict("records")

@mlflow.trace
def rag_answer(question, k=5):
    chunks = retrieve(question, k)
    context = "\n\n---\n\n".join(f"[Project {c['project_id']}]\n{c['text']}" for c in chunks)
    prompt = f"""You are an analyst of World Bank project evaluations. Answer the question using ONLY the context below, drawn from Implementation Completion Reports. Cite project IDs in your answer. If the context is insufficient, say so.

Context:
{context}

Question: {question}"""
    return {"answer": llm_chat(prompt),
            "sources": sorted({c["project_id"] for c in chunks})}

# COMMAND ----------

import pandas as pd

golden = pd.DataFrame([
    {"question": "What are common causes of delays in World Bank projects?",
     "reference": "Delays in recruiting PIU staff, delays contracting implementation partners due to unfamiliarity with Bank procurement, fragmented coordination, staff turnover affecting financial management, and procurement bottlenecks."},
    {"question": "What challenges are reported in Social Protection projects?",
     "reference": "Maintaining adequate benefit levels amid inflation and food prices, fiscal constraints, recurrent climate-induced shocks, and ensuring predictable financing for safety net programs."},
    {"question": "What lessons are reported about emergency response programs?",
     "reference": "Emergency responses are most effective when anchored in existing adaptive government social protection systems, with periodic adjustment of transfer values during shocks."},
    {"question": "What institutional arrangements strengthen safety net systems?",
     "reference": "Government-led Social Protection Working Groups, a social protection policy framework with a medium-term roadmap, and investment in adaptive delivery systems and government ownership."},
    {"question": "What issues affect financial management in projects?",
     "reference": "Staff turnover, delays documenting expenditures, weaknesses in internal controls, and institutional capacity constraints."},
    {"question": "What role does government ownership play in project outcomes?",
     "reference": "Government ownership and capacity strengthening enhance efficiency, transparency, and sustainability of program delivery."},
    {"question": "How do climate shocks affect World Bank operations?",
     "reference": "Recurrent climate-induced shocks strain fiscal space and require adaptive, shock-responsive program design."},
    {"question": "What is the purpose of Development Policy Financing?",
     "reference": "DPF provides budget support tied to policy and institutional reforms (prior actions) rather than financing specific investments."},
])



# COMMAND ----------

import json, re, mlflow

JUDGE_PROMPT = """You are an impartial evaluator of a question-answering system over World Bank project evaluation reports.

Question: {question}
Reference answer (ground truth): {reference}
System answer: {answer}

Score the system answer on two criteria, each an integer 1-5:
- relevance: does it directly address the question?
- correctness: is it consistent with the reference answer (no contradictions, covers the key points)?

Respond with ONLY a JSON object: {{"relevance": <int>, "correctness": <int>, "rationale": "<one sentence>"}}"""

def judge(question, reference, answer):
    raw = llm_chat(JUDGE_PROMPT.format(question=question, reference=reference, answer=answer), max_tokens=200)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    return json.loads(m.group()) if m else {"relevance": None, "correctness": None, "rationale": raw[:100]}

results = []
for _, row in golden.iterrows():
    out = rag_answer(row.question)
    scores = judge(row.question, row.reference, out["answer"])
    results.append({"question": row.question, **scores, "sources": str(out["sources"])})
    print(f"{scores['relevance']}/{scores['correctness']}  {row.question[:60]}")

res_df = pd.DataFrame(results)

with mlflow.start_run(run_name="rag_llm_judge_eval"):
    mlflow.log_metrics({
        "mean_relevance": res_df["relevance"].mean(),
        "mean_correctness": res_df["correctness"].mean(),
        "n_questions": len(res_df)})
    mlflow.log_table(res_df, "rag_eval_results.json")

spark.createDataFrame(res_df).write.format("delta").mode("overwrite") \
     .saveAsTable("worldbank.ml.rag_eval_results")
display(res_df)