# app_pdf_rag.py
# Loan Navigator Agent Suite — LLM-Enabled (Azure OpenAI)
# --------------------------------------------------------
# Changes in this rewrite:
# - Policy RAG now loads from PDFs under ./BL4A_policy_docs (instead of .txt)
# - Citations include file name and page number
# - SQLite database path set to ./LoanDB_BlueLoans4all.sqlite (configurable)
# - Dynamic schema introspection for NL -> SQL (no hard-coded table)
# - What-If explanations can be grounded with the document
#   "Amortization_Calculation_Explained_For_WhatIf_Calculator" if present
#
from __future__ import annotations
import os
import re
import json
import math
import sqlite3
import traceback
from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Azure OpenAI typed import
try:
    from openai import AzureOpenAI  # runtime import
except Exception:
    AzureOpenAI = None  # type: ignore[assignment]
if TYPE_CHECKING:
    from openai import AzureOpenAI as AzureOpenAIType

# ---------- Environment & Paths ----------
load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "."))
SQLITE_PATH = Path(os.getenv("SQLITE_PATH", DATA_DIR / "LoanDB_BlueLoans4all.sqlite"))
POLICY_DIR = Path(os.getenv("POLICY_DIR", DATA_DIR / "BL4A_policy_docs"))
AMORT_DOC_HINT = os.getenv(
    "AMORT_DOC_HINT",
    "Amortization_Calculation_Explained_For_WhatIf_Calculator"
)
POLICY_DIR.mkdir(parents=True, exist_ok=True)

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")  # e.g., gpt-4o-mini
EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")  # e.g., text-embedding-3-large
REQUIRE_LLM = True

# ---------- Azure OpenAI Client ----------
def get_aoai() -> "AzureOpenAIType":
    if not REQUIRE_LLM:
        raise RuntimeError("LLM not required, but client requested.")
    if AzureOpenAI is None:
        raise RuntimeError("The 'openai' package is not installed. Run: pip install openai")
    for name, val in [
        ("AZURE_OPENAI_ENDPOINT", AZURE_OPENAI_ENDPOINT),
        ("AZURE_OPENAI_API_KEY", AZURE_OPENAI_API_KEY),
        ("AZURE_OPENAI_CHAT_DEPLOYMENT", CHAT_DEPLOYMENT),
        ("AZURE_OPENAI_EMBED_DEPLOYMENT", EMBED_DEPLOYMENT),
    ]:
        if not val:
            raise RuntimeError(f"Missing environment variable: {name}")
    return AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,  # type: ignore[arg-type]
        api_key=AZURE_OPENAI_API_KEY,          # type: ignore[arg-type]
        api_version=AZURE_OPENAI_API_VERSION,
    )

# ---------- DB Helpers ----------
class DBSchema:
    """Holds tables and columns for introspection."""
    def __init__(self, conn: sqlite3.Connection):
        self.tables: Dict[str, List[str]] = {}
        cur = conn.cursor()
        # list tables
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        table_names = [r[0] for r in cur.fetchall()]
        for t in table_names:
            try:
                cur.execute(f"PRAGMA table_info('{t}');")
                cols = [row[1] for row in cur.fetchall()]  # col names
                self.tables[t] = cols
            except Exception:
                # ignore tables we can't introspect
                pass

    def to_prompt(self) -> str:
        lines = ["Database schema (SQLite):\n"]
        for t, cols in self.tables.items():
            lines.append(f"- {t}({', '.join(cols)})")
        return "\n".join(lines)

    def find_loan_table(self) -> Optional[Tuple[str, Dict[str, str]]]:
        """Try to locate a table that contains the typical columns
        required for What-If: loan_amount, interest_rate, tenure_months, loan_id.
        Returns (table_name, mapping) where mapping maps canonical names
        to actual column names.
        """
        # canonical candidates and synonyms
        syns = {
            "loan_id": {"loan_id", "LoanID", "loanId", "id", "Loan_Id"},
            "loan_amount": {"loan_amount", "principal", "amount", "LoanAmount", "principal_amount"},
            "interest_rate": {"interest_rate", "rate", "annual_rate", "InterestRate", "roi"},
            "tenure_months": {"tenure_months", "months", "term_months", "TenureMonths", "tenure"},
        }
        for t, cols in self.tables.items():
            lower_cols = {c.lower(): c for c in cols}
            mapping: Dict[str, str] = {}
            ok = True
            for k, cand in syns.items():
                match = None
                for c in cand:
                    if c.lower() in lower_cols:
                        match = lower_cols[c.lower()]
                        break
                if not match:
                    ok = False
                    break
                mapping[k] = match
            if ok:
                return t, mapping
        return None


def run_sql(q: str, params: Tuple = ()) -> Tuple[List[str], List[tuple]]:
    conn = sqlite3.connect(SQLITE_PATH)
    try:
        cur = conn.cursor()
        cur.execute(q, params)
        cols = [c[0] for c in cur.description] if cur.description else []
        rows = cur.fetchall()
        return cols, rows
    finally:
        conn.close()

# ---------- PDF-based RAG Index ----------
@dataclass
class Passage:
    file: str
    page: int
    text: str

class RagPDFIndex:
    """Lightweight RAG over PDF policy documents using AOAI embeddings."""
    def __init__(self, client: "AzureOpenAIType", policy_dir: Path, chunk_chars: int = 1200, overlap: int = 200):
        self.client = client
        self.policy_dir = policy_dir
        self.chunk_chars = chunk_chars
        self.overlap = overlap
        self.passages: List[Passage] = []
        self.vecs: Optional[np.ndarray] = None
        self._load_and_embed()

    def _chunk(self, text: str) -> List[str]:
        parts: List[str] = []
        i = 0
        step = self.chunk_chars - self.overlap
        while i < len(text):
            parts.append(text[i:i+self.chunk_chars])
            i += step if step > 0 else self.chunk_chars
        return parts

    def _load_and_embed(self) -> None:
        pdfs = sorted(self.policy_dir.glob("*.pdf"))
        passages: List[Passage] = []
        for pdf in pdfs:
            try:
                doc = fitz.open(pdf)
                for pno in range(len(doc)):
                    page = doc[pno]
                    txt = page.get_text("text") or ""
                    txt = txt.strip()
                    if not txt:
                        continue
                    for ch in self._chunk(txt):
                        passages.append(Passage(file=pdf.name, page=pno+1, text=ch))
                doc.close()
            except Exception:
                # skip unreadable PDFs
                continue
        self.passages = passages
        if not passages:
            self.vecs = np.zeros((0, 1536), dtype=float)
            return
        texts = [p.text for p in passages]
        vectors = self._embed_batch(texts)
        self.vecs = np.array(vectors, dtype=float)

    def _embed_batch(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            resp = self.client.embeddings.create(
                model=os.environ["AZURE_OPENAI_EMBED_DEPLOYMENT"],
                input=batch,
            )
            out.extend([d.embedding for d in resp.data])
        return out

    def search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        if self.vecs is None or self.vecs.size == 0:
            return []
        resp = self.client.embeddings.create(
            model=os.environ["AZURE_OPENAI_EMBED_DEPLOYMENT"],
            input=query,
        )
        qv = np.array(resp.data[0].embedding, dtype=float)
        qn = np.linalg.norm(qv) or 1.0
        dn = np.linalg.norm(self.vecs, axis=1)  # (N,)
        sims = (self.vecs @ qv) / ((dn * qn) + 1e-12)
        idx = sims.argsort()[::-1][:k]
        hits = []
        for i in idx:
            p = self.passages[i]
            hits.append({
                "file": p.file,
                "page": p.page,
                "text": p.text,
                "similarity": float(sims[i]),
                "citation": f"{p.file} p.{p.page}",
            })
        return hits

# ---------- Optional Amortization Doc Index ----------
class AmortDocIndex(RagPDFIndex):
    """RAG restricted to the amortization explanation document (if present)."""
    def __init__(self, client: "AzureOpenAIType", policy_dir: Path, hint: str):
        self.hint = hint.lower()
        self.client = client
        self.policy_dir = policy_dir
        self.chunk_chars = 1200
        self.overlap = 200
        self.passages = []
        self.vecs = None
        self._load_and_embed_specific()

    def _load_and_embed_specific(self) -> None:
        candidates = [p for p in self.policy_dir.glob("*.pdf") if self.hint in p.stem.lower()]
        passages: List[Passage] = []
        for pdf in candidates:
            try:
                doc = fitz.open(pdf)
                for pno in range(len(doc)):
                    page = doc[pno]
                    txt = page.get_text("text") or ""
                    txt = txt.strip()
                    if not txt:
                        continue
                    for ch in self._chunk(txt):
                        passages.append(Passage(file=pdf.name, page=pno+1, text=ch))
                doc.close()
            except Exception:
                continue
        self.passages = passages
        if not passages:
            self.vecs = np.zeros((0, 1536), dtype=float)
            return
        texts = [p.text for p in passages]
        vectors = self._embed_batch(texts)
        self.vecs = np.array(vectors, dtype=float)

# ---------- LLM Utilities ----------
class LLM:
    def __init__(self):
        self.client = get_aoai()

    def classify_intent(self, user_text: str) -> str:
        """Returns one of: sql, policy, whatif, unknown"""
        sys = (
            "You are a router. Classify the user's message into one of: ['sql','policy','whatif','unknown'].\n"
            "Examples:\n"
            "- 'How many EMIs left for Loan ID LN-1001?' → sql\n"
            "- 'What is the prepayment penalty?' → policy\n"
            "- 'If I prepay ₹20,000 in month 6, what happens?' → whatif\n"
            "Return ONLY JSON: {\"intent\":\"...\"}."
        )
        rsp = self.client.chat.completions.create(
            model=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
            messages=[
                {"role":"system","content":sys},
                {"role":"user","content":user_text}
            ],
            temperature=0,
            response_format={"type":"json_object"}
        )
        try:
            return json.loads(rsp.choices[0].message.content)["intent"]  # type: ignore
        except Exception:
            return "unknown"

    def nlp_to_sql(self, user_text: str, db_schema: DBSchema) -> Tuple[str, Tuple]:
        """Generate parameterized SQL with strict guardrails over the introspected DB schema."""
        sys = (
            "You convert the user's question to ONE parameterized SQLite SELECT.\n"
            "Rules:\n"
            "- Use ONLY tables/columns present in the provided schema.\n"
            "- SELECT only; forbid PRAGMA/INSERT/UPDATE/DELETE/ALTER/DROP/ATTACH/VACUUM.\n"
            "- Return JSON: {\"sql\":\"...\",\"params\":[...]}\n"
            "- Prefer exact match for identifiers like loan_id/customer_id when provided.\n"
            "- If identifiers missing, return a general SELECT with LIMIT 10.\n"
            "- For column name status write only Closed and Active\n"
        )
        prompt = f"Schema provided by the system:\n\n{db_schema.to_prompt()}\n\nUser question:\n{user_text}"
        rsp = self.client.chat.completions.create(
            model=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
            messages=[{"role":"system","content":sys}, {"role":"user","content":prompt}],
            temperature=0,
            response_format={"type":"json_object"}
        )
        js = json.loads(rsp.choices[0].message.content)  # type: ignore
        sql = (js.get("sql") or "").strip()
        params = js.get("params", [])
        # Guardrails
        low = sql.lower()
        if not low.startswith("select"):
            raise ValueError("Only SELECT allowed.")
        forbidden = ("insert","update","delete","drop","pragma","alter","attach","create","vacuum")
        if any(tok in low for tok in forbidden):
            raise ValueError("Forbidden SQL keyword detected.")
        if " from " not in f" {low} ":
            raise ValueError("Missing FROM clause.")
        if ";" in low.strip()[:-1]:
            raise ValueError("Multiple statements not allowed.")
        return sql, tuple(params)

    def rag_answer(self, question: str, passages: List[Dict[str,Any]]) -> str:
        if not passages:
            return ("I couldn't find an exact matching policy in the available PDFs. "
                    "Please consult the official policy booklet or share the file name.")
        context = "Use ONLY these passages. Cite as [filename p.N].\n\n"
        for p in passages:
            context += f"[{p['citation']}]\n{p['text']}\n\n"
        messages = [
            {"role":"system","content":"You are a compliance assistant. Be precise, cite sources, don't fabricate."},
            {"role":"user","content": f"Question: {question}\n\n{context}\nReturn a short answer with citations."}
        ]
        rsp = self.client.chat.completions.create(
            model=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
            messages=messages,  # pyright: ignore[reportArgumentType]
            temperature=0.1
        )
        return rsp.choices[0].message.content  # pyright: ignore[reportReturnType]

    def explain_simulation(self, sim: Dict[str,Any], amort_hits: List[Dict[str,Any]]) -> str:
        """Explain the simulation for a borrower. If amortization doc passages are available,
        ground the explanation in them and include citations."""
        base_sys = "You explain loan simulations concisely for non-experts. Include RBI-style disclosures if relevant."
        if amort_hits:
            context = "Use these amortization references for accuracy. Cite as [filename p.N].\n\n"
            for p in amort_hits[:4]:
                context += f"[{p['citation']}]\n{p['text']}\n\n"
            messages = [
                {"role":"system","content": base_sys},
                {"role":"user","content": f"Simulation JSON:\n{json.dumps(sim)}\n\n{context}\nSummarize for the borrower with citations."}
            ]
        else:
            messages = [
                {"role":"system","content": base_sys},
                {"role":"user","content": f"Summarize this for a borrower:\n{json.dumps(sim)}"}
            ]
        rsp = self.client.chat.completions.create(
            model=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
            messages=messages,
            temperature=0.3
        )
        return rsp.choices[0].message.content

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
        # Recompute EMI for remaining term after prepayment:
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

# ---------- FastAPI Models ----------
class ChatIn(BaseModel):
    query: str = Field(..., description="User question or instruction")

class WhatIfIn(BaseModel):
    loan_id: str
    prepay_amt: float
    prepay_month: int
    mode: str = Field(default="reduce_tenure", description="reduce_tenure | reduce_emi")

# ---------- App init ----------
app = FastAPI(title="Loan Navigator Agent Suite (PDF RAG)")
try:
    llm = LLM()
    policy_index = RagPDFIndex(llm.client, POLICY_DIR)  # build embeddings at startup
    amort_index = AmortDocIndex(llm.client, POLICY_DIR, AMORT_DOC_HINT)
except Exception as e:  # pragma: no cover
    if REQUIRE_LLM:
        raise
    llm = None  # type: ignore[assignment]
    policy_index = None  # type: ignore[assignment]
    amort_index = None  # type: ignore[assignment]

# ---------- Routing ----------
def route_query(query: str) -> Dict[str,Any]:
    # Build DB schema snapshot per request to stay up-to-date
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        db_schema = DBSchema(conn)
        conn.close()
    except Exception as e:
        db_schema = None  # type: ignore

    try:
        intent = llm.classify_intent(query)  # pyright: ignore[reportOptionalMemberAccess]
    except Exception:
        intent = "unknown"

    if intent == "sql":
        if db_schema is None or not db_schema.tables:
            return {"intent":"sql","ok":False,"error":"Database not found or empty schema"}
        try:
            sql, params = llm.nlp_to_sql(query, db_schema)  # pyright: ignore[reportOptionalMemberAccess]
            cols, rows = run_sql(sql, params)
            if not rows:
                return {"intent":"sql","ok":False,"message":"No rows found"}
            return {"intent":"sql","ok":True,"columns":cols,"rows":rows}
        except Exception as e:
            return {"intent":"sql","ok":False,"error":str(e)}

    if intent == "policy":
        try:
            hits = policy_index.search(query, k=4)  # type: ignore[union-attr]
            answer = llm.rag_answer(query, hits)    # pyright: ignore[reportOptionalMemberAccess]
            return {
                "intent":"policy","ok":True,"answer":answer,
                "citations":[{"citation":h["citation"],"similarity":h["similarity"]} for h in hits]
            }
        except Exception as e:
            return {"intent":"policy","ok":False,"error":str(e)}

    if intent == "whatif":
        # extract params from the query if present
        try:
            m = re.search(r"loan\s*id\s*(ln-\d+)", query.lower())
            loan_id = m.group(1).upper() if m else None
            m2 = re.search(r"prepay(?:ment)?\s*(?:of)?\s*rs?\.?\s*([0-9,]+)", query.lower())
            amt = float(m2.group(1).replace(",","")) if m2 else None
            m3 = re.search(r"month\s*(\d+)", query.lower())
            pm = int(m3.group(1)) if m3 else None
            mode = "reduce_tenure" if "reduce tenure" in query.lower() else ("reduce_emi" if "reduce emi" in query.lower() else "reduce_tenure")
        except Exception:
            loan_id, amt, pm, mode = None, None, None, "reduce_tenure"

        # Locate a table with required columns
        try:
            conn = sqlite3.connect(SQLITE_PATH)
            schema = DBSchema(conn)
            found = schema.find_loan_table()
            conn.close()
        except Exception as e:
            found = None

        if not found:
            return {"intent":"whatif","ok":False,"error":"Could not locate loan table with required columns in the database."}
        table, mapc = found
        # If loan_id not provided, pick first row
        if loan_id:
            q = f"SELECT {mapc['loan_amount']}, {mapc['interest_rate']}, {mapc['tenure_months']} FROM {table} WHERE {mapc['loan_id']} = ?"
            params = (loan_id,)
        else:
            q = f"SELECT {mapc['loan_amount']}, {mapc['interest_rate']}, {mapc['tenure_months']}, {mapc['loan_id']} FROM {table} LIMIT 1"
            params = ()
        cols, rows = run_sql(q, params)
        if not rows:
            return {"intent":"whatif","ok":False,"error":"No matching loan found"}
        if loan_id is None:
            # set loan_id from the fetched row if we selected it
            if len(cols) == 4:
                loan_id = rows[0][3]
            else:
                loan_id = "UNKNOWN"
        P, rate, n = rows[0][0], rows[0][1], int(rows[0][2])

        # defaults for prepay
        if pm is None:
            pm = max(1, min(6, n))
        if amt is None:
            # 10% of principal as a heuristic
            amt = round(0.10 * float(P), 2)

        sim = simulate_prepayment(float(P), float(rate), int(n), int(pm), float(amt), mode)
        sim_out = {"loan_id":loan_id, **sim}
        # pull amortization references
        amort_hits = amort_index.search("amortization prepayment EMI interest schedule", k=4) if amort_index else []
        explanation = llm.explain_simulation(sim_out, amort_hits)  # pyright: ignore[reportOptionalMemberAccess]
        return {"intent":"whatif","ok":True,"simulation":sim_out,"explanation":explanation,
                "citations":[{"citation":h["citation"],"similarity":h["similarity"]} for h in amort_hits]}

    return {"intent":"unknown","ok":False,
            "message":"I can help with SQL (loan data), policy questions (PDFs), or prepayment simulations. Please rephrase or add context."}

# ---------- Endpoints ----------
@app.post("/chat")
def chat(inp: ChatIn) -> Dict[str,Any]:
    try:
        return route_query(inp.query)
    except Exception as e:  # pragma: no cover
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/whatif")
def whatif(inp: WhatIfIn) -> Dict[str,Any]:
    try:
        # Find the loan table dynamically
        conn = sqlite3.connect(SQLITE_PATH)
        schema = DBSchema(conn)
        found = schema.find_loan_table()
        conn.close()
        if not found:
            return {"ok":False,"error":"Loan table with required columns not found"}
        table, mapc = found

        q = f"SELECT {mapc['loan_amount']}, {mapc['interest_rate']}, {mapc['tenure_months']} FROM {table} WHERE {mapc['loan_id']} = ?"
        cols, rows = run_sql(q, (inp.loan_id,))
        if not rows:
            return {"ok":False,"error":f"Loan {inp.loan_id} not found"}
        P, rate, n = rows[0]
        if inp.prepay_month < 1 or inp.prepay_month > n or inp.prepay_amt < 0:
            return {"ok":False,"error":"Invalid prepayment inputs"}
        sim = simulate_prepayment(float(P), float(rate), int(n), int(inp.prepay_month), float(inp.prepay_amt), inp.mode)
        sim_out = {"loan_id":inp.loan_id, **sim}
        amort_hits = amort_index.search("amortization prepayment EMI interest schedule", k=4) if amort_index else []
        explanation = llm.explain_simulation(sim_out, amort_hits)  # pyright: ignore[reportOptionalMemberAccess]
        return {"ok":True,"simulation":sim_out,"explanation":explanation,
                "citations":[{"citation":h["citation"],"similarity":h["similarity"]} for h in amort_hits]}
    except Exception as e:  # pragma: no cover
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
