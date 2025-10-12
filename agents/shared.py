# shared.py
from __future__ import annotations
import os, re, json, sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", ".")
SQLITE_PATH = os.getenv("SQLITE_PATH", os.path.join(DATA_DIR, "LoanDB_BlueLoans4all.sqlite"))
POLICY_DIR = os.getenv("POLICY_DIR", os.path.join(DATA_DIR, "BL4A_policy_docs"))
CHROMA_POLICY_DIR = os.getenv("CHROMA_POLICY_DIR", os.path.join(DATA_DIR, "chroma_policy"))
CHROMA_AMORT_DIR = os.getenv("CHROMA_AMORT_DIR", os.path.join(DATA_DIR, "chroma_amort"))
AMORT_DOC_HINT = os.getenv("AMORT_DOC_HINT", "Amortization_Calculation_Explained_For_WhatIf_Calculator").lower()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
AZURE_OPENAI_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")

os.makedirs(POLICY_DIR, exist_ok=True)
os.makedirs(CHROMA_POLICY_DIR, exist_ok=True)
os.makedirs(CHROMA_AMORT_DIR, exist_ok=True)

def open_readonly_sqlite(path: str) -> sqlite3.Connection:
    uri = f"file:{os.path.abspath(path)}?mode=ro"
    return sqlite3.connect(uri, uri=True)

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
        return "\n".join([f"- {t}({', '.join(cols)})" for t, cols in self.tables.items()])

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
    import re
    m = re.findall(r"from\s+([a-zA-Z0-9_]+)", low)
    for tbl in m:
        if tbl not in [x.lower() for x in allowed_tables]:
            raise ValueError(f"Query references unknown table: {tbl}")

# Embeddings & Vector Stores (lazy)
POLICY_VS = None
AMORT_VS = None

def get_embeddings():
    missing = []
    for k in ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_EMBED_DEPLOYMENT"]:
        if not os.getenv(k):
            missing.append(k)
    if missing:
        raise RuntimeError(f"Azure OpenAI embedding config missing: {', '.join(missing)}")
    return AzureOpenAIEmbeddings(
        azure_deployment=AZURE_OPENAI_EMBED_DEPLOYMENT,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
    )

def build_or_load_policy_vs():
    global POLICY_VS
    embeddings = get_embeddings()
    vs = Chroma(collection_name="bl4a_policy", embedding_function=embeddings, persist_directory=CHROMA_POLICY_DIR)
    try:
        if vs._collection.count() > 0:
            POLICY_VS = vs
            return vs
    except Exception:
        pass
    docs = []
    for path in sorted(Path(POLICY_DIR).glob("*.pdf")):
        loader = PyMuPDFLoader(str(path))
        docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    splits = splitter.split_documents(docs) if docs else []
    if splits:
        vs.add_documents(splits)
        vs.persist()
    POLICY_VS = vs
    return vs

def build_or_load_amort_vs():
    global AMORT_VS
    embeddings = get_embeddings()
    vs = Chroma(collection_name="bl4a_amort", embedding_function=embeddings, persist_directory=CHROMA_AMORT_DIR)
    try:
        if vs._collection.count() > 0:
            AMORT_VS = vs
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
    AMORT_VS = vs
    return vs

def get_policy_vs():
    return POLICY_VS or build_or_load_policy_vs()

def get_amort_vs():
    return AMORT_VS or build_or_load_amort_vs()

# LLM
def get_chat_llm(temp: float = 0.0) -> AzureChatOpenAI:
    missing = []
    for k in ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_CHAT_DEPLOYMENT"]:
        if not os.getenv(k):
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

# What-if math
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
