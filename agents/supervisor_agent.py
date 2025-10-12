# agents/supervisor_agent.py
from __future__ import annotations
import json
from langchain_core.prompts import ChatPromptTemplate
from shared import get_chat_llm


def router_node(state: dict) -> dict:
    """Classify intent -> one of sql|policy|whatif|unknown.
    Uses Azure JSON mode for reliable JSON output.
    """
    try:
        llm = get_chat_llm(0.0).bind(response_format={"type": "json_object"})
    except Exception as e:
        state["error"] = f"Router LLM config error: {e}"
        return state

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a router. Classify the user's message into one of: ['sql','policy','whatif','unknown'].\n"
         "Return ONLY JSON: {{\"intent\":\"...\"}}."),
        ("user", "{q}")
    ])
    rsp = (prompt | llm).invoke({"q": state.get("query","")})
    content = getattr(rsp, 'content', str(rsp))
    try:
        js = json.loads(content)
        state["intent"] = js.get("intent", "unknown")
    except Exception:
        # Conservative fallback
        ql = str(state.get("query","")) .lower()
        if any(k in ql for k in ["prepay","what if","reduce emi","reduce tenure","what-if"]):
            state["intent"] = "whatif"
        elif any(k in ql for k in ["policy","rbi","top-up","prepayment penalty","disclosure","eligibility"]):
            state["intent"] = "policy"
        elif any(k in ql for k in ["loan id","customer id","emi","interest rate","tenure","select","from "]):
            state["intent"] = "sql"
        else:
            state["intent"] = "unknown"
    return state
