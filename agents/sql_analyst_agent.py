# agents/sql_analyst_agent.py
from __future__ import annotations
import json
from langchain_core.prompts import ChatPromptTemplate
from shared import DBSchema, SQLITE_PATH, validate_sql, open_readonly_sqlite, get_chat_llm


def sql_node(state: dict) -> dict:
    query = state.get("query")
    if not query:
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
        content = getattr(rsp, 'content', str(rsp))
        js = json.loads(content)
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
