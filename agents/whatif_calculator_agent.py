# agents/whatif_calculator_agent.py
from __future__ import annotations
import re, json, os
from langchain_core.prompts import ChatPromptTemplate
from shared import DBSchema, SQLITE_PATH, simulate_prepayment, get_amort_vs, get_chat_llm


def whatif_node(state: dict) -> dict:
    q = state.get("query","")
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

    import sqlite3
    if loan_id:
        sql = f"SELECT {mapc['loan_amount']}, {mapc['interest_rate']}, {mapc['tenure_months']} FROM {table} WHERE {mapc['loan_id']} = ?"
        params = (loan_id,)
    else:
        sql = f"SELECT {mapc['loan_amount']}, {mapc['interest_rate']}, {mapc['tenure_months']}, {mapc['loan_id']} FROM {table} LIMIT 1"
        params = ()

    try:
        conn = sqlite3.connect(f"file:{os.path.abspath(SQLITE_PATH)}?mode=ro", uri=True)
        cur = conn.cursor(); cur.execute(sql, params)
        row = cur.fetchone()
    except Exception as e:
        state["error"] = f"DB error: {e}"
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
    pm = pm or max(1, min(6, n))
    amt = amt or round(0.10 * P, 2)

    sim = simulate_prepayment(P, rate, n, pm, amt, mode)
    sim_out = {"loan_id": loan_id, **sim}

    # RAG grounding
    try:
        vs = get_amort_vs()
        retriever = vs.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke("amortization prepayment EMI interest schedule")
    except Exception:
        docs = []

    ctx, cits = [], []
    for d in docs:
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
        "explanation": getattr(explanation,'content',str(explanation)),
        "citations": cits
    }
    return state
