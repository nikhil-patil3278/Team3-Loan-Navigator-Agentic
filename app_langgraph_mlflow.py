from pathlib import Path
# src = r'''# app_langgraph_mlflow.py
"""
Loan Navigator Agent Suite — LangGraph + ChromaDB (Azure OpenAI)
With optional MLflow tracking (non-intrusive)
-----------------------------------------------------------------
- Lazy initialization of Chroma vector stores and Azure embeddings (built on-demand in nodes)
- Hardened SQL node and /chat endpoint retained
- **MLflow** logging added for /chat and /whatif (duration, success, params, JSON artifact)

Run:
  python -m uvicorn app_langgraph_mlflow:app --reload
"""

from __future__ import annotations
import os
import re
import json
import time
import tempfile
import sqlite3
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Optional MLflow (failsafe)
try:
    import mlflow  # type: ignore
    MLFLOW_ENABLED = True
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore
    MLFLOW_ENABLED = False

def _mlflow_setup() -> None:
    if not MLFLOW_ENABLED:
        return
    try:
        uri = os.getenv("MLFLOW_TRACKING_URI")
        if uri:
            mlflow.set_tracking_uri(uri)
        exp = os.getenv("MLFLOW_EXPERIMENT_NAME", "LoanNavigator")
        mlflow.set_experiment(exp)
    except Exception:
        # Do not break runtime if MLflow is misconfigured
        pass

_mlflow_setup()

def _ml_log_event(run_name: str,
                  params: Dict[str, Any] | None = None,
                  metrics: Dict[str, float] | None = None,
                  artifact_obj: Dict[str, Any] | None = None) -> None:
    if not MLFLOW_ENABLED:
        return
    try:
        with mlflow.start_run(run_name=run_name):
            if params:
                safe_params: Dict[str, Any] = {}
                for k, v in params.items():
                    safe_params[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                mlflow.log_params(safe_params)
            if metrics:
                mlflow.log_metrics(metrics)
            if artifact_obj is not None:
                try:
                    text = json.dumps(artifact_obj, indent=2, ensure_ascii=False)
                    with tempfile.NamedTemporaryFile("w", delete=False, suffix="_response.json") as f:
                        f.write(text)
                        tmp = f.name
                    mlflow.log_artifact(tmp, artifact_path="responses")
                except Exception:
                    pass
    except Exception:
        # Never interfere with app behavior
        pass

# LangChain / LangGraph
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langgraph.graph import StateGraph, END

# ---------- Environment & Paths ----------
load_dotenv()
DATA_DIR = os.getenv("DATA_DIR", ".")
SQLITE_PATH = os.getenv("SQLITE_PATH", os.path.join(DATA_DIR, "LoanDB_BlueLoans4all.sqlite"))
POLICY_DIR = os.getenv("POLICY_DIR", os.path.join(DATA_DIR, "BL4A_policy_docs"))
CHROMA_POLICY_DIR = os.getenv("CHROMA_POLICY_DIR", os.path.join(DATA_DIR, "chroma_policy"))
CHROMA_AMORT_DIR = os.getenv("CHROMA_AMORT_DIR", os.path.join(DATA_DIR, "chroma_amort"))
AMORT_DOC_HINT = os.getenv("AMORT_DOC_HINT", "Amortization_Calculation_Explained_For_WhatIf_Calculator").lower()

# Azure OpenAI config
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")  # e.g., gpt-4o-mini
AZURE_OPENAI_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")  # e.g., text-embedding-3-large

# ---------- Utilities ----------
def ensure_dirs():
    os.makedirs(POLICY_DIR, exist_ok=True)
    os.makedirs(CHROMA_POLICY_DIR, exist_ok=True)
    os.makedirs(CHROMA_AMORT_DIR, exist_ok=True)


def open_readonly_sqlite(path: str) -> sqlite3.Connection:
    """Open a read-only SQLite connection."""
    uri = f"file:{os.path.abspath(path)}?mode=ro"
    return sqlite3.connect(uri, uri=True)

# ---------- DB Introspection & Guardrails ----------
@dataclass
class DBSchema:
    tables: Dict[str, List[str]]

    @classmethod
    def from_sqlite(cls, path: str) -> "DBSchema":
        conn = open_readonly_sqlite(path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables: Dict[str, List[str]] = {}
            for (t,) in cur.fetchall():
                try:
                    cur.execute(f"PRAGMA table_info('{t}');")
                    cols = [r[1] for r in cur.fetchall()]
                    tables[t] = cols
                except Exception:
                    pass
            return cls(tables=tables)
        finally:
            conn.close()

    def to_prompt(self) -> str:
        lines = []
        for t, cols in self.tables.items():
            lines.append(f"- {t}({', '.join(cols)})")
        return "\n".join(lines)

    def find_loan_table(self) -> Optional[Tuple[str, Dict[str, str]]]:
        syns = {
            "loan_id": {"loan_id", "LoanID", "loanId", "id", "Loan_Id"},
            "loan_amount": {"loan_amount", "principal", "amount", "LoanAmount", "principal_amount"},
            "interest_rate": {"interest_rate", "rate", "annual_rate", "InterestRate", "roi"},
            "tenure_months": {"tenure_months", "months", "term_months", "TenureMonths", "tenure"},
        }
        for t, cols in self.tables.items():
            lower = {c.lower(): c for c in cols}
            mapping: Dict[str, str] = {}
            ok = True
            for k, cand in syns.items():
                match = None
                for c in cand:
                    if c.lower() in lower:
                        match = lower[c.lower()]
                        break
                if not match:
                    ok = False
                    break
                mapping[k] = match
            if ok:
                return t, mapping
        return None


def validate_sql(sql: str, allowed_tables: List[str]) -> None:
    low = sql.lower().strip()
    if not low.startswith("select"):
        raise ValueError("Only SELECT statements are allowed.")
    forbidden = ("insert","update","delete","drop","pragma","alter","attach","create","vacuum","--","/*",";\n","; ")
    if any(tok in low for tok in forbidden):
        raise ValueError("Forbidden SQL keyword or pattern detected.")
    m = re.findall(r"from\s+([a-zA-Z0-9_]+)", low)
    for tbl in m:
        if tbl not in [x.lower() for x in allowed_tables]:
            raise ValueError(f"Query references unknown table: {tbl}")

# ---------- Embeddings & Vector Stores (Lazy) ----------
POLICY_VS: Optional[Chroma] = None
AMORT_VS: Optional[Chroma] = None


def get_embeddings():
    # Validate env only when needed
    missing = []
    for k in ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_EMBED_DEPLOYMENT"]:
        if not globals().get(k) or not os.getenv(k):
            missing.append(k)
    if missing:
        raise RuntimeError(f"Azure OpenAI embedding config missing: {', '.join(missing)}")
    return AzureOpenAIEmbeddings(
        azure_deployment=AZURE_OPENAI_EMBED_DEPLOYMENT,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
    )


def build_or_load_policy_vs() -> Chroma:
    embeddings = get_embeddings()
    vs = Chroma(collection_name="bl4a_policy", embedding_function=embeddings, persist_directory=CHROMA_POLICY_DIR)
    # If it already has docs, just return
    try:
        if vs._collection.count() > 0:
            return vs
    except Exception:
        pass
    # Load PDFs and add
    docs = []
    for path in sorted(Path(POLICY_DIR).glob("*.pdf")):
        loader = PyMuPDFLoader(str(path))
        docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    splits = splitter.split_documents(docs) if docs else []
    if splits:
        vs.add_documents(splits)
        vs.persist()
    return vs


def build_or_load_amort_vs() -> Chroma:
    embeddings = get_embeddings()
    vs = Chroma(collection_name="bl4a_amort", embedding_function=embeddings, persist_directory=CHROMA_AMORT_DIR)
    try:
        if vs._collection.count() > 0:
            return vs
    except Exception:
        pass
    docs = []
    for path in sorted(Path(POLICY_DIR).glob("*.pdf")):
        if AMORT_DOC_HINT in path.stem.lower():
            loader = PyMuPDFLoader(str(path))
            docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    splits = splitter.split_documents(docs) if docs else []
    if splits:
        vs.add_documents(splits)
        vs.persist()
    return vs


def get_policy_vs() -> Chroma:
    global POLICY_VS
    if POLICY_VS is None:
        POLICY_VS = build_or_load_policy_vs()
    return POLICY_VS


def get_amort_vs() -> Chroma:
    global AMORT_VS
    if AMORT_VS is None:
        AMORT_VS = build_or_load_amort_vs()
    return AMORT_VS

# ---------- LLMs ----------

def get_chat_llm(temp: float = 0.0) -> AzureChatOpenAI:
    missing = []
    for k in ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_CHAT_DEPLOYMENT"]:
        if not globals().get(k) or not os.getenv(k):
            missing.append(k)
    if missing:
        raise RuntimeError(f"Azure OpenAI chat config missing: {', '.join(missing)}")
    return AzureChatOpenAI(
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        temperature=temp,
    )

# ---------- Deterministic What-If ----------

def emi(principal: float, annual_rate: float, months: int) -> float:
    r = annual_rate/12/100.0
    if r == 0:
        return principal / months
    return principal * r * (1+r)**months / ((1+r)**months - 1)


def amort_schedule(principal: float, annual_rate: float, months: int, extra: Optional[Dict[int, float]] = None) -> List[Dict[str,Any]]:
    extra = extra or {}
    r = annual_rate/12/100.0
    e = emi(principal, annual_rate, months)
    bal = principal
    out: List[Dict[str,Any]] = []
    for m in range(1, months+1):
        interest = bal * r
        principal_comp = e - interest
        prepay = extra.get(m, 0.0)
        bal = max(0.0, bal - principal_comp - prepay)
        out.append({"month":m,"emi":e,"interest":interest,"principal":principal_comp+prepay,"prepay":prepay,"balance":bal})
        if bal <= 1e-6:
            break
    return out


def simulate_prepayment(principal: float, rate: float, months: int, prepay_month: int, prepay_amt: float, mode: str='reduce_tenure') -> Dict[str,Any]:
    base = amort_schedule(principal, rate, months)
    base_interest = sum(x["interest"] for x in base)
    schedule = amort_schedule(principal, rate, months, extra={prepay_month: prepay_amt})
    new_interest = sum(x["interest"] for x in schedule)
    if mode == "reduce_tenure":
        new_months = len(schedule)
        new_emi = schedule[0]["emi"] if schedule else 0.0
    else:
        if prepay_month <= len(schedule):
            rem_balance = schedule[prepay_month-1]["balance"]
            rem_months = months - prepay_month
            new_emi = emi(rem_balance, rate, rem_months) if rem_months > 0 else 0.0
            new_months = months
        else:
            new_emi = schedule[0]["emi"] if schedule else 0.0
            new_months = months
    return {
        "base_months": len(base),
        "base_interest": base_interest,
        "new_months": new_months,
        "new_emi": new_emi,
        "new_interest": new_interest,
        "interest_saved": base_interest - new_interest,
        "schedule_sample": schedule[:6]
    }

# ---------- LangGraph State & Nodes ----------
class FlowState(TypedDict, total=False):
    query: str
    intent: str
    sql_result: Dict[str, Any]
    policy_answer: Dict[str, Any]
    whatif_result: Dict[str, Any]
    error: str

# Ensure directories at import (safe)
ensure_dirs()

# --- Router node ---
def router_node(state: FlowState) -> FlowState:
    try:
        # Force JSON output
        llm = get_chat_llm(0.0).bind(response_format={"type": "json_object"})
    except Exception as e:
        state["error"] = f"Router LLM config error: {e}"
        return state
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a router. Classify the user's message into one of: ['sql','policy','whatif','unknown'].\n"
         "Examples:\n- 'How many EMIs left for Loan ID LN-1001?' → sql\n- 'What is the prepayment penalty?' → policy\n- 'If I prepay ₹20,000 in month 6, what happens?' → whatif\n"
         "Return ONLY JSON: {{\"intent\":\"...\"}}."),
        ("user", "{q}")
    ])
    chain = prompt | llm
    rsp = chain.invoke({"q": state["query"]})
    content = rsp.content if hasattr(rsp, "content") else str(rsp)
    try:
        js = json.loads(content)
        state["intent"] = js.get("intent", "unknown")
        return state
    except Exception:
        ql = state["query"].lower()
        if any(k in ql for k in ["prepay", "what if", "reduce emi", "reduce tenure", "what-if"]):
            state["intent"] = "whatif"
        elif any(k in ql for k in ["policy", "rbi", "top-up", "top up", "prepayment penalty", "disclosure", "eligibility"]):
            state["intent"] = "policy"
        elif any(k in ql for k in ["loan id", "customer id", "emi", "interest rate", "tenure", "select", "from "]):
            state["intent"] = "sql"
        else:
            state["intent"] = "unknown"
        return state

# --- SQL agent node (HARDENED) ---
def sql_node(state: FlowState) -> FlowState:
    try:
        query = state["query"]  # type: ignore
    except Exception:
        state["error"] = "Internal: missing 'query' in state"
        return state

    try:
        schema = DBSchema.from_sqlite(SQLITE_PATH)
    except Exception as e:
        state["error"] = f"DB schema error: {e}"
        return state

    if not schema.tables:
        state["error"] = "Database not found or empty schema"
        return state

    try:
        llm = get_chat_llm(0.0)
    except Exception as e:
        state["error"] = f"SQL LLM config error: {e}"
        return state

    sys = (
        "You convert the user's question to ONE parameterized SQLite SELECT over the provided schema.\n"
        "Rules:\n- Use ONLY tables/columns shown.\n- SELECT only; forbid PRAGMA/INSERT/UPDATE/DELETE/ALTER/DROP/ATTACH/VACUUM.\n"
        "- Return JSON: {{\"sql\":\"...\",\"params\":[...]}}\n- Prefer exact match for identifiers like loan_id/customer_id.\n"
        "- For the column like status consider only Active and Closed\n"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys),
        ("user", "Schema:\n{schema}\n\nUser question:\n{question}")
    ])
    try:
        rsp = (prompt | llm).invoke({"schema": schema.to_prompt(), "question": query})
        content = rsp.content if hasattr(rsp, "content") else str(rsp)
        js = json.loads(content)  # type: ignore
        sql = (js.get("sql") or "").strip()
        params = js.get("params", [])
    except Exception as e:
        state["error"] = f"LLM->SQL parse error: {e}"
        return state

    try:
        validate_sql(sql, list(schema.tables.keys()))
    except Exception as e:
        state["error"] = f"SQL validation error: {e}"
        return state

    try:
        conn = open_readonly_sqlite(SQLITE_PATH)
        cur = conn.cursor()
        cur.execute(sql, tuple(params))
        cols = [c[0] for c in cur.description] if cur.description else []
        rows = cur.fetchall()
    except Exception as e:
        state["error"] = f"SQL execution error: {e}"
        return state
    finally:
        try:
            conn.close()
        except Exception:
            pass

    rows_json = [list(r) for r in rows] if rows else []
    state["sql_result"] = {"columns": cols, "rows": rows_json}
    return state

# --- Policy agent node (RAG with Chroma, lazy VS) ---
def policy_node(state: FlowState) -> FlowState:
    try:
        vs = get_policy_vs()
    except Exception as e:
        state["error"] = f"Policy VS/embeddings error: {e}"
        return state

    retriever = vs.as_retriever(search_kwargs={"k": 4})
    try:
        llm = get_chat_llm(0.1)
    except Exception as e:
        state["error"] = f"Policy LLM config error: {e}"
        return state

    docs = retriever.invoke(state["query"])  # list of Documents
    if not docs:
        state["policy_answer"] = {
            "answer": "I couldn't find a matching policy in the available PDFs.",
            "citations": []
        }
        return state
    context = []
    cits = []
    for d in docs:
        source = d.metadata.get("source", "")
        page = d.metadata.get("page", None)
        cite = f"{os.path.basename(source)} p.{page}" if page is not None else os.path.basename(source)
        context.append(f"[{cite}]\n{d.page_content}")
        cits.append({"citation": cite})
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a compliance assistant. Use ONLY the provided context. Be precise and include citations like [filename p.N]."),
        ("user", "Question: {q}\n\nContext:\n{ctx}\n\nReturn a short answer with citations.")
    ])
    ans = (prompt | llm).invoke({"q": state["query"], "ctx": "\n\n".join(context)})
    state["policy_answer"] = {"answer": ans.content if hasattr(ans, "content") else str(ans), "citations": cits}
    return state

# --- What-If agent node (lazy amort VS) ---
def whatif_node(state: FlowState) -> FlowState:
    q = state["query"]
    try:
        m = re.search(r"loan\s*id\s*(ln-\d+)", q.lower())
        loan_id = m.group(1).upper() if m else None
        m2 = re.search(r"prepay(?:ment)?\s*(?:of)?\s*rs?\.?\s*([0-9,]+)", q.lower())
        amt = float(m2.group(1).replace(",","")) if m2 else None
        m3 = re.search(r"month\s*(\d+)", q.lower())
        pm = int(m3.group(1)) if m3 else None
        mode = "reduce_tenure" if "reduce tenure" in q.lower() else ("reduce_emi" if "reduce emi" in q.lower() else "reduce_tenure")
    except Exception:
        loan_id, amt, pm, mode = None, None, None, "reduce_tenure"

    schema = DBSchema.from_sqlite(SQLITE_PATH)
    found = schema.find_loan_table()
    if not found:
        state["error"] = "Could not locate loan table with required columns in the database."
        return state
    table, mapc = found

    if loan_id:
        sql = f"SELECT {mapc['loan_amount']}, {mapc['interest_rate']}, {mapc['tenure_months']} FROM {table} WHERE {mapc['loan_id']} = ?"
        params = (loan_id,)
    else:
        sql = f"SELECT {mapc['loan_amount']}, {mapc['interest_rate']}, {mapc['tenure_months']}, {mapc['loan_id']} FROM {table} LIMIT 1"
        params = ()

    try:
        conn = open_readonly_sqlite(SQLITE_PATH)
        cur = conn.cursor()
        cur.execute(sql, params)
        row = cur.fetchone()
    except Exception as e:
        state["error"] = f"DB error: {str(e)}"
        return state
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if not row:
        state["error"] = "No matching loan found"
        return state

    if loan_id is None:
        loan_id = row[3] if len(row) == 4 else "UNKNOWN"
    P, rate, n = float(row[0]), float(row[1]), int(row[2])
    if pm is None:
        pm = max(1, min(6, n))
    if amt is None:
        amt = round(0.10 * P, 2)

    sim = simulate_prepayment(P, rate, n, pm, amt, mode)
    sim_out = {"loan_id": loan_id, **sim}

    # Retrieve amortization references lazily
    try:
        vs = get_amort_vs()
        amort_retriever = vs.as_retriever(search_kwargs={"k": 4})
        amort_docs = amort_retriever.invoke("amortization prepayment EMI interest schedule")
    except Exception:
        amort_docs = []

    ctx = []
    cits = []
    for d in amort_docs:
        source = d.metadata.get("source", "")
        page = d.metadata.get("page", None)
        cite = f"{os.path.basename(source)} p.{page}" if page is not None else os.path.basename(source)
        ctx.append(f"[{cite}]\n{d.page_content}")
        cits.append({"citation": cite})

    try:
        llm = get_chat_llm(0.3)
    except Exception as e:
        state["error"] = f"WhatIf LLM config error: {e}"
        return state

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You explain loan simulations concisely for non-experts. Include RBI-style disclosures if relevant."),
        ("user", "Simulation JSON:\n{sim}\n\nContext (amortization):\n{ctx}\n\nSummarize for the borrower with citations if context is provided.")
    ])
    explanation = (prompt | llm).invoke({"sim": json.dumps(sim_out), "ctx": "\n\n".join(ctx)})

    state["whatif_result"] = {
        "simulation": sim_out,
        "explanation": explanation.content if hasattr(explanation, "content") else str(explanation),
        "citations": cits
    }
    return state

# ---------- Build LangGraph ----------
workflow = StateGraph(FlowState)
workflow.add_node("router", router_node)
workflow.add_node("sql_agent", sql_node)
workflow.add_node("policy_agent", policy_node)
workflow.add_node("whatif_agent", whatif_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges(
    "router",
    lambda s: s.get("intent", "unknown"),
    {
        "sql": "sql_agent",
        "policy": "policy_agent",
        "whatif": "whatif_agent",
        "unknown": END,
    },
)
workflow.add_edge("sql_agent", END)
workflow.add_edge("policy_agent", END)
workflow.add_edge("whatif_agent", END)

app_graph = workflow.compile()

# ---------- FastAPI Models & App ----------
class ChatIn(BaseModel):
    query: str = Field(..., description="User question or instruction")

class WhatIfIn(BaseModel):
    loan_id: str
    prepay_amt: float
    prepay_month: int
    mode: str = Field(default="reduce_tenure", description="reduce_tenure | reduce_emi")

app = FastAPI(title="Loan Navigator — LangGraph + ChromaDB (lazy init) + MLflow")

@app.post("/chat")
def chat(inp: ChatIn) -> Dict[str, Any]:
    start = time.time()
    try:
        result = app_graph.invoke({"query": inp.query})
        intent = result.get("intent", "unknown")
        # MLflow log
        _ml_log_event(
            run_name="chat",
            params={"intent": intent, "query": inp.query},
            metrics={"duration_s": time.time() - start, "success": 0.0 if "error" in result else 1.0},
            artifact_obj=result
        )
        if "error" in result:
            return {"intent": intent, "ok": False, "error": result["error"]}
        if intent == "sql":
            return {"intent":"sql", "ok": True, **({"sql_result": result.get("sql_result")} if "sql_result" in result else {})}
        if intent == "policy":
            return {"intent":"policy", "ok": True, **({"policy_answer": result.get("policy_answer")} if "policy_answer" in result else {})}
        if intent == "whatif":
            return {"intent":"whatif", "ok": True, **({"whatif_result": result.get("whatif_result")} if "whatif_result" in result else {})}
        return {"intent":"unknown", "ok": False, "message": "I can help with SQL, policy questions (PDFs), or prepayment simulations."}
    except Exception as e:  # pragma: no cover
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/whatif")
def whatif(inp: WhatIfIn) -> Dict[str, Any]:
    start = time.time()
    try:
        state: FlowState = {"query": f"loan id {inp.loan_id} prepayment of {inp.prepay_amt} month {inp.prepay_month} {inp.mode}", "intent":"whatif"}
        state_out = whatif_node(state)
        # MLflow log
        _ml_log_event(
            run_name="whatif",
            params={"loan_id": inp.loan_id, "prepay_amt": inp.prepay_amt, "prepay_month": inp.prepay_month, "mode": inp.mode},
            metrics={"duration_s": time.time() - start, "success": 1.0 if "whatif_result" in state_out else 0.0},
            artifact_obj=state_out
        )
        if "whatif_result" not in state_out:
            return {"ok": False, "error": state_out.get("error", "Unknown error")}
        return {"ok": True, **state_out["whatif_result"]}
    except Exception as e:  # pragma: no cover
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

Path('app_langgraph_mlflow.py').write_text(src, encoding='utf-8')
print('app_langgraph_mlflow.py written with MLflow integration, size:', len(src))