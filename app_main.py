# app_main.py
"""
Main FastAPI app orchestrating the agentic flow using LangGraph.
Splits agents into separate files and includes optional MLflow logging.
Run: python -m uvicorn app_main:app --reload
"""
from __future__ import annotations
import time, traceback, json
from typing import Any, Dict, TypedDict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

from agents.supervisor_agent import router_node
from agents.sql_analyst_agent import sql_node
from agents.policy_guru_agent import policy_node
from agents.whatif_calculator_agent import whatif_node

# Optional MLflow (failsafe)
try:
    import mlflow  # type: ignore
    MLFLOW_ENABLED = True
except Exception:
    mlflow = None  # type: ignore
    MLFLOW_ENABLED = False


def _ml_log(run_name: str, params: Dict[str, Any], metrics: Dict[str, float], artifact: Dict[str, Any]):
    if not MLFLOW_ENABLED:
        return
    try:
        # basic config via env: MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
        exp = __import__('os').getenv('MLFLOW_EXPERIMENT_NAME', 'LoanNavigator')
        mlflow.set_experiment(exp)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({k: (json.dumps(v, ensure_ascii=False) if isinstance(v,(dict,list)) else v) for k,v in params.items()})
            mlflow.log_metrics(metrics)
            import tempfile
            with tempfile.NamedTemporaryFile('w', delete=False, suffix='_response.json') as f:
                f.write(json.dumps(artifact, indent=2, ensure_ascii=False))
                tmp = f.name
            mlflow.log_artifact(tmp, artifact_path='responses')
    except Exception:
        pass

class FlowState(TypedDict, total=False):
    query: str
    intent: str
    sql_result: Dict[str, Any]
    policy_answer: Dict[str, Any]
    whatif_result: Dict[str, Any]
    error: str

# Build graph
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

# FastAPI models & app
class ChatIn(BaseModel):
    query: str = Field(..., description="User question or instruction")

class WhatIfIn(BaseModel):
    loan_id: str
    prepay_amt: float
    prepay_month: int
    mode: str = Field(default="reduce_tenure", description="reduce_tenure | reduce_emi")

app = FastAPI(title="Loan Navigator — Agentic (LangGraph) + MLflow")

@app.post("/chat")
def chat(inp: ChatIn) -> Dict[str, Any]:
    start = time.time()
    try:
        result = app_graph.invoke({"query": inp.query})
        intent = result.get("intent", "unknown")
        _ml_log("chat",
                params={"intent": intent, "query": inp.query},
                metrics={"duration_s": time.time()-start, "success": 0.0 if "error" in result else 1.0},
                artifact=result)
        if "error" in result:
            return {"intent": intent, "ok": False, "error": result["error"]}
        if intent == "sql":
            return {"intent":"sql", "ok": True, **({"sql_result": result.get("sql_result")} if "sql_result" in result else {})}
        if intent == "policy":
            return {"intent":"policy", "ok": True, **({"policy_answer": result.get("policy_answer")} if "policy_answer" in result else {})}
        if intent == "whatif":
            return {"intent":"whatif", "ok": True, **({"whatif_result": result.get("whatif_result")} if "whatif_result" in result else {})}
        return {"intent":"unknown", "ok": False, "message": "I can help with SQL, policy (PDFs), or prepayment simulations."}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/whatif")
def whatif(inp: WhatIfIn) -> Dict[str, Any]:
    start = time.time()
    try:
        state: FlowState = {"query": f"loan id {inp.loan_id} prepayment of {inp.prepay_amt} month {inp.prepay_month} {inp.mode}", "intent":"whatif"}
        state_out = whatif_node(state)
        _ml_log("whatif",
                params={"loan_id": inp.loan_id, "prepay_amt": inp.prepay_amt, "prepay_month": inp.prepay_month, "mode": inp.mode},
                metrics={"duration_s": time.time()-start, "success": 1.0 if "whatif_result" in state_out else 0.0},
                artifact=state_out)
        if "whatif_result" not in state_out:
            return {"ok": False, "error": state_out.get("error", "Unknown error")}
        return {"ok": True, **state_out["whatif_result"]}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
