import os, time
import numpy as np
import pandas as pd
import streamlit as st
from databricks.sdk import WorkspaceClient

st.set_page_config(page_title="WB Project Intelligence", page_icon="🌍", layout="wide")



# Load CSS from file — avoids all string escaping issues
css_path = os.path.join(os.path.dirname(__file__), "style.css")
with open(css_path) as f:
    st.markdown(f.read(), unsafe_allow_html=True)

@st.cache_resource
def get_client():
    return WorkspaceClient()

@st.cache_resource
def get_warehouse_id():
    whs = list(get_client().warehouses.list())
    for wh in whs:
        s = str(wh.state)
        if "RUNNING" in s or "STARTING" in s:
            return wh.id
    return whs[0].id

def sql_to_df(query):
    client = get_client()
    stmt = client.statement_execution.execute_statement(
        warehouse_id=get_warehouse_id(),
        statement=query,
        wait_timeout="0s",
    )
    deadline = time.time() + 90
    while time.time() < deadline:
        s = str(stmt.status.state)
        if any(x in s for x in ("SUCCEEDED","FAILED","CANCELED","CLOSED")):
            break
        time.sleep(3)
        stmt = client.statement_execution.get_statement(stmt.statement_id)
    if "SUCCEEDED" not in str(stmt.status.state):
        raise Exception(f"Query failed: {stmt.status.state} {stmt.status.error}")
    cols = [c.name for c in stmt.manifest.schema.columns]
    rows = [list(r) for r in stmt.result.data_array] if stmt.result and stmt.result.data_array else []
    return pd.DataFrame(rows, columns=cols)

@st.cache_data(ttl=3600)
def load_silver():
    df = sql_to_df("SELECT project_id,project_name,country,region,primary_sector,status,approval_date,commitment_usd FROM worldbank.silver.projects")
    df["commitment_usd"] = pd.to_numeric(df["commitment_usd"], errors="coerce")
    return df

@st.cache_data(ttl=3600)
def load_features():
    df = sql_to_df("SELECT * FROM worldbank.ml.project_features")
    for col in ["label","commitment_usd","approval_year","planned_duration_days","has_approval_date","has_closing_date"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@st.cache_data(ttl=3600)
def load_embeddings():
    import json, urllib.request
    client = get_client()
    wh_id  = get_warehouse_id()

    # Use EXTERNAL_LINKS for large result sets (embeddings exceed inline limit)
    from databricks.sdk.service.sql import Disposition, Format
    stmt = client.statement_execution.execute_statement(
        warehouse_id=wh_id,
        statement="SELECT chunk_id, project_id, text, embedding FROM worldbank.docs.icr_embeddings",
        wait_timeout="0s",
        disposition=Disposition.EXTERNAL_LINKS,
        format=Format.JSON_ARRAY,
    )
    deadline = time.time() + 120
    while time.time() < deadline:
        s = str(stmt.status.state)
        if any(x in s for x in ("SUCCEEDED","FAILED","CANCELED","CLOSED")):
            break
        time.sleep(3)
        stmt = client.statement_execution.get_statement(stmt.statement_id)
    if "SUCCEEDED" not in str(stmt.status.state):
        raise Exception(f"Embeddings query failed: {stmt.status.state}")

    # Download chunks from external links
    rows = []
    for chunk in stmt.result.external_links:
        with urllib.request.urlopen(chunk.external_link) as r:
            rows.extend(json.loads(r.read().decode()))

    cols = [c.name for c in stmt.manifest.schema.columns]
    df = pd.DataFrame(rows, columns=cols)
    df["embedding"] = df["embedding"].apply(
        lambda x: np.array(x if isinstance(x,list) else eval(str(x)), dtype=np.float32))
    mat = np.vstack(df["embedding"].values).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat/(norms+1e-9), df[["chunk_id","project_id","text"]].to_dict("records")

def get_auth_headers():
    """Get auth headers — works on Databricks Apps and Streamlit Cloud."""
    try:
        # Try OAuth via SDK (Databricks Apps)
        h = get_client().config.authenticate()
        if h.get("Authorization","").replace("Bearer ",""):
            return h
    except Exception:
        pass
    # Fallback: env var PAT
    token = os.environ.get("DATABRICKS_TOKEN","")
    return {"Authorization": f"Bearer {token}"}

def get_host():
    host = get_client().config.host or os.environ.get("DATABRICKS_HOST","")
    host = host.rstrip("/")
    return host if host.startswith("http") else f"https://{host}"

def embed_query(text):
    import requests as req
    r = req.post(
        f"{get_host()}/serving-endpoints/databricks-gte-large-en/invocations",
        headers={**get_auth_headers(), "Content-Type": "application/json"},
        json={"input": text}, timeout=30)
    r.raise_for_status()
    v = np.array(r.json()["data"][0]["embedding"], dtype=np.float32)
    return v/(np.linalg.norm(v)+1e-9)

def retrieve(question, mat, meta, k=5):
    sims = mat @ embed_query(question)
    return [(meta[i],float(sims[i])) for i in np.argsort(-sims)[:k]]

def generate(prompt):
    import requests as req
    r = req.post(
        f"{get_host()}/serving-endpoints/databricks-meta-llama-3-3-70b-instruct/invocations",
        headers={**get_auth_headers(), "Content-Type": "application/json"},
        json={"messages":[{"role":"user","content":prompt}],"max_tokens":700},
        timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def rag(question, mat, meta, k=5):
    results = retrieve(question, mat, meta, k)
    context = "\n\n---\n\n".join(f"[Project {r[0]['project_id']}]\n{r[0]['text']}" for r in results)
    prompt = f"Answer using ONLY the ICR context below. Cite project IDs.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    return {"answer":generate(prompt),"sources":sorted({r[0]["project_id"] for r in results}),"chunks":results}

with st.sidebar:
    k = st.slider("Chunks (k)", 3, 10, 5)
    st.markdown("---")
    st.success("Databricks Apps — auto auth")
    st.markdown("---")
    st.markdown("**AUC:** 0.892 · **RAG:** 4.9/5.0")
    st.markdown("[GitHub](https://github.com/yourusername/wb-project-intelligence-platform)")



try:
    silver = load_silver()
    features = load_features()
    mat, meta = load_embeddings()
except Exception as e:
    import traceback
    st.error(f"Could not load data: {e}")
    st.code(traceback.format_exc())
    st.stop()

st.markdown('<div class="hero"><h1>🌍 World Bank Project Intelligence Platform</h1><p>Live Delta Lake · RAG over ICR PDFs · ML outcome prediction · Databricks end-to-end</p></div>', unsafe_allow_html=True)

c1,c2,c3,c4,c5 = st.columns(5)
for col,(v,l) in zip([c1,c2,c3,c4,c5],[(f"{len(silver):,}","Projects"),(f"{len(features):,}","ML Rows"),("0.892","AUC"),("4.9/5","RAG"),("10","Services")]):
    col.markdown(f'<div class="metric-card"><div class="metric-val">{v}</div><div class="metric-lbl">{l}</div></div>',unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

tab1,tab2,tab3 = st.tabs(["💬 RAG — Ask the ICRs","📊 Portfolio Analytics","🔗 Hybrid Query"])

with tab1:
    st.markdown('<div class="sec">Ask questions over World Bank ICR PDFs</div>',unsafe_allow_html=True)
    examples=["What causes delays in World Bank projects?","What lessons are in Social Protection projects?","How do climate shocks affect outcomes?","What issues affect financial management?"]
    cols=st.columns(len(examples))
    for i,(c,q) in enumerate(zip(cols,examples)):
        if c.button(q[:28]+"…",key=f"e{i}"): st.session_state["q"]=q
    question=st.text_area("Your question",value=st.session_state.get("q",""),height=70)
    if st.button("🔍 Search & Answer"):
        if not question.strip(): st.warning("Enter a question.")
        else:
            with st.spinner("Retrieving and generating…"):
                try:
                    out=rag(question,mat,meta,k)
                    st.markdown(f'<div class="answer-box">{out["answer"]}</div>',unsafe_allow_html=True)
                    chips=" ".join(f'<span class="chip">{s}</span>' for s in out["sources"])
                    st.markdown(f"**Sources:** {chips}",unsafe_allow_html=True)
                    with st.expander("📄 Retrieved passages"):
                        for cm,sim in out["chunks"]:
                            st.markdown(f'<div class="chunk"><b style="color:#4da6ff">{cm["project_id"]} · {sim:.3f}</b><br>{cm["text"][:400]}…</div>',unsafe_allow_html=True)
                except Exception as e:
                    import traceback
                    st.error(f"Error: {e}")
                    st.code(traceback.format_exc())

with tab2:
    st.markdown('<div class="sec">Structured analytics — live reads from Delta</div>',unsafe_allow_html=True)
    col1,col2=st.columns(2)
    with col1:
        st.markdown("**Completion rate by region**")
        reg=(features[features["region"]!="Unknown"].groupby("region").agg(n=("label","count"),rate=("label","mean")).reset_index().sort_values("rate"))
        reg["Completion %"]=(reg["rate"]*100).round(1)
        st.dataframe(reg[["region","n","Completion %"]].rename(columns={"region":"Region","n":"Projects"}),use_container_width=True,hide_index=True)
    with col2:
        st.markdown("**Top 15 countries by commitment**")
        top=(silver.dropna(subset=["country","commitment_usd"]).groupby("country")["commitment_usd"].sum().sort_values(ascending=False).head(15).reset_index())
        top["commitment_usd"]=(top["commitment_usd"]/1e9).round(2)
        top.columns=["Country","Commitment ($B)"]
        st.dataframe(top,use_container_width=True,hide_index=True)
    if "primary_sector" in features.columns:
        sel=st.selectbox("Filter by sector",sorted(features["primary_sector"].dropna().unique()))
        sec=features[(features["primary_sector"]==sel)&(features["region"]!="Unknown")]
        if len(sec):
            agg=(sec.groupby("region").agg(n=("label","count"),rate=("label","mean"),avg=("commitment_usd","mean")).reset_index().sort_values("rate"))
            agg["Completion %"]=(agg["rate"]*100).round(1); agg["Avg $M"]=(agg["avg"]/1e6).round(1)
            st.dataframe(agg[["region","n","Completion %","Avg $M"]].rename(columns={"region":"Region","n":"Projects"}),use_container_width=True,hide_index=True)
    if "approval_date" in silver.columns:
        s2=silver.copy(); s2["year"]=pd.to_datetime(s2["approval_date"],errors="coerce").dt.year
        trend=(s2.dropna(subset=["year"]).groupby("year")["commitment_usd"].sum().reset_index())
        trend["$B"]=(trend["commitment_usd"]/1e9).round(2)
        trend=trend[(trend["year"]>=1970)&(trend["year"]<=2026)]
        st.markdown("**Portfolio growth over time**")
        st.line_chart(trend.set_index("year")[["$B"]],use_container_width=True)

with tab3:
    st.markdown('<div class="sec">Structured + Narrative — joined by project_id</div>',unsafe_allow_html=True)
    sectors=sorted(features["primary_sector"].dropna().unique()) if "primary_sector" in features.columns else []
    sel3=st.selectbox("Sector",sectors,key="h_sel")
    if st.button("🔗 Run Hybrid Analysis"):
        ca,cb=st.columns(2)
        with ca:
            st.markdown("**📊 Quantitative (SDK → Delta)**")
            s=features[(features["primary_sector"]==sel3)&(features["region"]!="Unknown")]
            if len(s):
                a=(s.groupby("region").agg(n=("label","count"),rate=("label","mean"),avg=("commitment_usd","mean")).reset_index().sort_values("rate"))
                a["Completion %"]=(a["rate"]*100).round(1); a["Avg $M"]=(a["avg"]/1e6).round(1)
                st.dataframe(a[["region","n","Completion %","Avg $M"]].rename(columns={"region":"Region","n":"Projects"}),use_container_width=True,hide_index=True)
                st.info(f"⚠️ Lowest: **{a.iloc[0]['region']}** ({a.iloc[0]['Completion %']}%)")
            else: st.info("No data for this sector.")
        with cb:
            st.markdown("**📄 Narrative (ICR RAG)**")
            with st.spinner("Generating…"):
                try:
                    out3=rag(f"Challenges and lessons in {sel3} projects",mat,meta,k)
                    st.markdown(f'<div class="answer-box">{out3["answer"]}</div>',unsafe_allow_html=True)
                    st.markdown(" ".join(f'<span class="chip">{s}</span>' for s in out3["sources"]),unsafe_allow_html=True)
                except Exception as e:
                    pass
                import traceback
                st.error(f"Error: {e}")
                st.code(traceback.format_exc())
