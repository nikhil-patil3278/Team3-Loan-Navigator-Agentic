# agents/policy_guru_agent.py
from __future__ import annotations
from langchain_core.prompts import ChatPromptTemplate
from shared import get_policy_vs, get_chat_llm
import os


def policy_node(state: dict) -> dict:
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

    docs = retriever.invoke(state.get("query",""))
    if not docs:
        state["policy_answer"] = {"answer": "I couldn't find a matching policy in the available PDFs.", "citations": []}
        return state
    context, cits = [], []
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
    ans = (prompt | llm).invoke({"q": state.get("query",""), "ctx": "\n\n".join(context)})
    state["policy_answer"] = {"answer": getattr(ans,'content',str(ans)), "citations": cits}
    return state
