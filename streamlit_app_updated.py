# streamlit_app.py
# Streamlit UI for Loan Navigator (LangGraph + Chroma backend)
# ------------------------------------------------------------
# Tabs:
# - Chat: routes to SQL / Policy RAG / What-If via /chat (LangGraph router)
# - What-If Simulator: direct /whatif + local charts
# - Policy Q&A: policy questions with citations (PDF RAG)
# - Loans: search/list loans (Loan ID / Customer ID) + CSV download
# - Top-Up: eligibility check (SQL) + policy criteria with citations

import os
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st  # type: ignore
import plotly.graph_objects as go  # pyright: ignore[reportMissingImports]

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
DEFAULT_API_BASE = os.getenv("BACKEND_API_BASE", "http://127.0.0.1:8000")
st.set_page_config(page_title="Loan Navigator", page_icon="🗨️", layout="wide")

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.title("🔌 Backend Settings")
api_base = st.sidebar.text_input("FastAPI Base URL", value=DEFAULT_API_BASE, help="E.g., http://127.0.0.1:8000")
if api_base.endswith("/"):
    api_base = api_base[:-1]

default_loan = st.sidebar.text_input("Default Loan ID", value="2001")
st.sidebar.caption("Ensure the backend is running: `python -m uvicorn app_langgraph:app --reload --port 8000`")

# ------------------------------------------------------------
# HTTP helpers
# ------------------------------------------------------------
def post_json(path: str, payload: dict, timeout: float = 60.0):
    url = f"{api_base}{path}"
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        if resp.status_code >= 400:
            return {"ok": False, "error": f"HTTP {resp.status_code}: {resp.text}"}
        return resp.json()
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

def chat_api(query: str):
    return post_json("/chat", {"query": query})

def whatif_api(loan_id: str, amt: float, month: int, mode: str):
    return post_json("/whatif", {
        "loan_id": loan_id,
        "prepay_amt": float(amt),
        "prepay_month": int(month),
        "mode": mode
    })

# ------------------------------------------------------------
# Deterministic amortization (for local visuals)
# ------------------------------------------------------------
def emi(principal: float, annual_rate: float, months: int) -> float:
    r = annual_rate/12/100.0
    if r == 0:
        return principal / months
    return principal * r * (1+r)**months / ((1+r)**months - 1)

def amort_schedule(principal: float, annual_rate: float, months: int, extra: dict | None = None) -> pd.DataFrame:
    extra = extra or {}
    r = annual_rate/12/100.0
    e = emi(principal, annual_rate, months)
    bal = principal
    rows = []
    for m in range(1, months+1):
        interest = bal * r
        principal_comp = e - interest
        prepay = float(extra.get(m, 0.0))
        bal = max(0.0, bal - principal_comp - prepay)
        rows.append({
            "month": m,
            "emi": e,
            "interest": interest,
            "principal": principal_comp + prepay,
            "prepay": prepay,
            "balance": bal
        })
        if bal <= 1e-6:
            break
    return pd.DataFrame(rows)

# ------------------------------------------------------------
# UI helpers
# ------------------------------------------------------------
def pretty_json(d: dict):
    try:
        return json.dumps(d, indent=2, ensure_ascii=False)
    except Exception:
        return str(d)


def metric_card_cols(metrics: list[tuple[str, str | float | int, str]]):
    cols = st.columns(len(metrics))
    for c, (label, value, help_text) in zip(cols, metrics):
        with c:
            st.metric(label, value)
            if help_text:
                st.caption(help_text)


def draw_balance_chart(df: pd.DataFrame, title: str = "Amortization Balance"):
    if df is None or df.empty:
        st.info("No schedule to plot.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["month"], y=df["balance"], mode="lines+markers",
        line=dict(color="#2E86C1", width=3), name="Balance"
    ))
    fig.update_layout(
        height=350, template="plotly_white", title=title,
        xaxis_title="Month", yaxis_title="Outstanding Balance (₹)"
    )
    st.plotly_chart(fig, use_container_width=True)

# New: robust SQL response -> DataFrame for LangGraph backend
# /chat SQL returns: {intent:"sql", ok:true, sql_result:{columns:[...], rows:[[...], ...]}}
# Legacy shape (if any): {intent:"sql", ok:true, columns:[...], rows:[...]}

def parse_sql_df(res: dict) -> pd.DataFrame:
    if not isinstance(res, dict):
        return pd.DataFrame()
    if res.get("intent") != "sql" or not res.get("ok", False):
        return pd.DataFrame()
    sr = res.get("sql_result") or {}
    cols = sr.get("columns") or res.get("columns") or []
    rows = sr.get("rows") or res.get("rows") or []
    try:
        return pd.DataFrame(rows, columns=cols)
    except Exception:
        return pd.DataFrame()

# New: getters for nested policy / what-if shapes (LangGraph)

def get_policy_answer(res: dict) -> tuple[str, list[dict]]:
    # /chat policy returns: {intent:"policy", ok:true, policy_answer:{answer:str, citations:[{citation: str}]}}
    if not isinstance(res, dict):
        return "", []
    pa = res.get("policy_answer") or {}
    ans = pa.get("answer") or res.get("answer") or ""
    cits = pa.get("citations") or res.get("citations") or []
    return ans, cits


def get_whatif_result(res: dict) -> tuple[dict, str, list[dict]]:
    # /chat whatif returns nested whatif_result; /whatif returns flat keys
    if not isinstance(res, dict):
        return {}, "", []
    wr = res.get("whatif_result") or {}
    sim = wr.get("simulation") or res.get("simulation") or {}
    expl = wr.get("explanation") or res.get("explanation") or ""
    cits = wr.get("citations") or res.get("citations") or []
    return sim, expl, cits

# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # {"role":"user/assistant","content":str}

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("🗨️ Loan Navigator — Frontend")
st.caption("Streamlit UI for the FastAPI backend (LangGraph + Chroma).")

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab_chat, tab_whatif, tab_policy, tab_loans, tab_topup = st.tabs(
    ["Chat", "What‑If Simulator", "Policy Q&A", "Loans", "Top‑Up"]
)

# ------------------------------------------------------------
# Tab: Chat
# ------------------------------------------------------------
with tab_chat:
    st.subheader("Ask anything (SQL, policy, or what‑if)")
    chat_input = st.text_area(
        "Your message",
        value="How many EMIs left for Loan ID 2001?",
        height=100,
        placeholder="e.g., What is the prepayment penalty as per policy?"
    )
    c1, c2 = st.columns([1, 4])
    with c1:
        run_chat = st.button("Send", type="primary")
    with c2:
        st.write("")

    if run_chat and chat_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": chat_input})
        with st.spinner("Thinking..."):
            res = chat_api(chat_input)
            if not res.get("ok", True) and "error" in res:
                st.error(res.get("error") or "Unknown error")
            else:
                st.session_state.chat_history.append({"role": "assistant", "content": pretty_json(res)})

    st.markdown("### Conversation")
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            # Attempt to parse structured response
            try:
                data = json.loads(msg["content"])
                intent = data.get("intent") or ("policy" if data.get("policy_answer") else "unknown")
                with st.expander(f"Assistant ({intent}) response", expanded=True):
                    st.code(pretty_json(data), language="json")

                    if intent == "policy":
                        ans, cits = get_policy_answer(data)
                        if ans:
                            st.markdown("**Answer:**")
                            st.write(ans)
                        if cits:
                            st.markdown("**Citations:**")
                            for c in cits:
                                # new shape: {citation: "file p.N"}
                                cit = c.get("citation") or c.get("file")
                                sim = c.get("similarity")
                                if sim is not None:
                                    st.write(f"- `{cit}` (similarity: {sim:.3f})")
                                else:
                                    st.write(f"- `{cit}`")

                    elif intent == "sql":
                        df = parse_sql_df(data)
                        if not df.empty:
                            st.markdown("**SQL Result:**")
                            st.dataframe(df)
                        else:
                            st.info("No data returned.")

                    elif intent == "whatif":
                        sim, expl, cits = get_whatif_result(data)
                        if sim:
                            st.markdown("**Simulation:**")
                            st.code(pretty_json(sim), language="json")
                        if expl:
                            st.markdown("**Explanation:**")
                            st.write(expl)
                        if cits:
                            st.markdown("**Citations:**")
                            for c in cits:
                                cit = c.get("citation")
                                simv = c.get("similarity")
                                if simv is not None:
                                    st.write(f"- `{cit}` (similarity: {simv:.3f})")
                                else:
                                    st.write(f"- `{cit}`")
            except Exception:
                st.markdown(f"**Assistant:** {msg['content']}")

# ------------------------------------------------------------
# Tab: What‑If Simulator
# ------------------------------------------------------------
with tab_whatif:
    st.subheader("Prepayment scenario")
    with st.form("whatif_form"):
        col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1.2])
        with col1:
            loan_id = st.text_input("Loan ID", value=default_loan)
        with col2:
            prepay_amt = st.number_input("Prepay Amount (₹)", value=10000.0, min_value=0.0, step=1000.0)
        with col3:
            prepay_month = st.number_input("Prepay in Month", value=6, min_value=1, step=1)
        with col4:
            mode = st.selectbox("Strategy", options=["reduce_tenure", "reduce_emi"], index=0)
        submitted = st.form_submit_button("Simulate", type="primary")

    if submitted:
        with st.spinner("Simulating..."):
            res = whatif_api(loan_id, prepay_amt, prepay_month, mode)
            if not res.get("ok", True) and "error" in res:
                st.error(res.get("error") or "Unknown error")
            else:
                sim, expl, cits = get_whatif_result(res)
                if sim:
                    metrics = [
                        ("Base Months", sim.get("base_months", "-"), "Original tenure"),
                        ("New Months", sim.get("new_months", "-"), "After prepayment"),
                        ("Interest Saved (₹)", f"{sim.get('interest_saved',0):,.0f}", "Base - new interest"),
                        ("New EMI (₹)", f"{sim.get('new_emi',0):,.0f}", "If 'reduce_emi' mode or first EMI value"),
                    ]
                    metric_card_cols(metrics)
                if expl:
                    st.markdown("**LLM Explanation:**")
                    st.write(expl)
                # Optional visuals using local amort calc if we can fetch loan details
                try:
                    q = f"Show loan details for Loan ID {loan_id}"
                    detail = chat_api(q)
                    df = parse_sql_df(detail)
                    if not df.empty and set(["loan_amount","interest_rate","tenure_months"]).issubset(df.columns):
                        P = float(df.loc[0, "loan_amount"])  # pyright: ignore[reportArgumentType]
                        rate = float(df.loc[0, "interest_rate"])  # pyright: ignore[reportArgumentType]
                        months = int(df.loc[0, "tenure_months"])  # pyright: ignore[reportArgumentType]
                        base_df = amort_schedule(P, rate, months, {})
                        prepay_df = amort_schedule(P, rate, months, {int(prepay_month): float(prepay_amt)})
                        st.markdown("**Amortization (Local Visualization)**")
                        colA, colB = st.columns(2)
                        with colA:
                            st.caption("Base schedule")
                            draw_balance_chart(base_df, "Base Balance")
                        with colB:
                            st.caption("After prepayment")
                            draw_balance_chart(prepay_df, "Balance after Prepayment")
                        with st.expander("Download schedules"):
                            st.download_button("Download base.csv", base_df.to_csv(index=False).encode("utf-8"), file_name="base_schedule.csv")
                            st.download_button("Download prepay.csv", prepay_df.to_csv(index=False).encode("utf-8"), file_name="prepay_schedule.csv")
                    else:
                        st.info("Could not fetch loan details for charts; showing summary only.")
                except Exception as e:
                    st.info(f"Visualization note: {e}")
                with st.expander("Raw response"):
                    st.code(pretty_json(res), language="json")

# ------------------------------------------------------------
# Tab: Policy Q&A
# ------------------------------------------------------------
with tab_policy:
    st.subheader("Ask a policy/compliance question")
    pol_q = st.text_area(
        "Your policy question",
        value="What is the prepayment penalty for microloans?",
        height=100
    )
    run_policy = st.button("Ask Policy", type="primary")
    if run_policy and pol_q.strip():
        with st.spinner("Retrieving policy & summarizing..."):
            res = chat_api(pol_q)  # backend classifies as policy & runs RAG
            if not res.get("ok", True) and "error" in res:
                st.error(res.get("error") or "Unknown error")
            else:
                ans, cits = get_policy_answer(res)
                st.markdown("**Answer:**")
                st.write(ans or pretty_json(res))
                if cits:
                    st.markdown("**Citations:**")
                    for c in cits:
                        cit = c.get("citation") or c.get("file") or ""
                        sim = c.get("similarity")
                        if sim is not None:
                            st.write(f"- `{cit}` (similarity: {sim:.3f})")
                        else:
                            st.write(f"- `{cit}`")
                with st.expander("Raw response"):
                    st.code(pretty_json(res), language="json")

# ------------------------------------------------------------
# Tab: Loans (search/list)
# ------------------------------------------------------------
with tab_loans:
    st.subheader("Search Loans")
    with st.form("loan_search_form"):
        col1, col2, col3 = st.columns([1.2, 1.2, 1])
        with col1:
            q_loan = st.text_input("Loan ID (exact)", value="")
        with col2:
            q_cust = st.text_input("Customer ID (exact)", value="")
        with col3:
            limit = st.number_input("Limit", min_value=1, value=50, step=10)
        search_btn = st.form_submit_button("Search", type="primary")

    if search_btn:
        if q_loan:
            query = f"Show loan details for Loan ID {q_loan}"
        elif q_cust:
            query = f"List all loans for customer id {q_cust}"
        else:
            query = "List all loans"
        with st.spinner("Fetching loans..."):
            res = chat_api(query)
            if not res.get("ok", True) and "error" in res:
                st.error(res.get("error") or "Unknown error")
            else:
                df = parse_sql_df(res)
                if df.empty:
                    st.info("No records found.")
                else:
                    df = df.head(limit)
                    st.dataframe(df, use_container_width=True)
                    with st.expander("Download CSV"):
                        st.download_button(
                            "Download loans.csv",
                            df.to_csv(index=False).encode("utf-8"),
                            file_name="loans.csv"
                        )
                    if "loan_amount" in df.columns:
                        st.caption("Summary")
                        metrics = [
                            ("Count", len(df), ""),
                            ("Total Loan (₹)", f"{df['loan_amount'].sum():,.0f}", ""),
                            ("Avg Rate (%)", f"{df['interest_rate'].mean():.2f}" if "interest_rate" in df.columns else "-", "")
                        ]
                        metric_card_cols(metrics)

# ------------------------------------------------------------
# Tab: Top‑Up Eligibility
# ------------------------------------------------------------
with tab_topup:
    st.subheader("Check Top‑Up Eligibility")
    with st.form("topup_form"):
        top_loan = st.text_input("Loan ID", value=default_loan)
        check_btn = st.form_submit_button("Check Eligibility", type="primary")

    if check_btn and top_loan.strip():
        # 1) SQL: ask explicitly for the flag (LLM-friendly phrasing)
        with st.spinner("Checking eligibility..."):
            res_sql = chat_api(f"Fetch topup_eligible for Loan ID {top_loan}")

            def to_df(res: dict) -> pd.DataFrame:
                try:
                    return parse_sql_df(res)
                except Exception:
                    return pd.DataFrame()

            df = to_df(res_sql)
            if df.empty or "topup_eligible" not in df.columns:
                # Second try—spell out columns very explicitly
                res_sql = chat_api(f"Return columns loan_id, topup_eligible from loans where loan_id = '{top_loan}'")
                df = to_df(res_sql)

            eligible = None
            raw_val = None
            if not df.empty and "topup_eligible" in df.columns:
                raw_val = df.iloc[0]["topup_eligible"]
                s = str(raw_val).strip().lower()
                if s in ("1", "true", "yes", "y"): 
                    eligible = 1
                elif s in ("0", "false", "no", "n"):
                    eligible = 0

            colA, colB = st.columns([1, 2])
            with colA:
                if eligible == 1:
                    st.success(f"Loan {top_loan}: Top‑Up Eligible ✅")
                elif eligible == 0:
                    st.info(f"Loan {top_loan}: Not Eligible ❌")
                else:
                    st.warning(f"Loan {top_loan}: Eligibility unknown (data not found).")
            with colB:
                st.caption("Raw SQL response")
                st.code(pretty_json(res_sql), language="json")

        # 2) Policy: show criteria with citations
        with st.spinner("Fetching policy criteria..."):
            res_pol = chat_api("What is the top‑up eligibility policy?")
            if not res_pol.get("ok", True) and "error" in res_pol:
                st.error(res_pol.get("error") or "Unknown error")
            else:
                ans, cits = get_policy_answer(res_pol)
                st.markdown("**Top‑Up Eligibility Policy (summarized):**")
                st.write(ans or pretty_json(res_pol))
                if cits:
                    st.markdown("**Citations:**")
                    for c in cits:
                        cit = c.get("citation") or c.get("file") or ""
                        sim = c.get("similarity")
                        if sim is not None:
                            st.write(f"- `{cit}` (similarity: {sim:.3f})")
                        else:
                            st.write(f"- `{cit}`")
