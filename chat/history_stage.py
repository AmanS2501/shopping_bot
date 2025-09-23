# chat/history_stage.py
from typing import List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from .prompt_loader import PromptConfig

def format_history(history: List[dict], max_chars: int = 4000) -> str:
    out, total = [], 0
    for turn in history[-20:]:
        prefix = "User:" if turn["role"] == "user" else "Assistant:"
        line = f"{prefix} {turn['content']}".strip()
        if total + len(line) > max_chars:
            break
        total += len(line)
        out.append(line)
    return "\n".join(out)

def run_history_stage(question: str, history: List[dict], model: str, prompt_path: str) -> dict:
    cfg = PromptConfig.load(prompt_path)
    system_txt = "\n".join(cfg.data["prompt_shell"]["system"])
    developer_txt = "\n".join(cfg.data["prompt_shell"]["developer"])
    tpl = cfg.data["prompt_shell"]["user_template"]
    safe_history = format_history(history)
    user_txt = tpl.format(question=question, history=safe_history)


    prompt = ChatPromptTemplate.from_messages(
        [("system", system_txt), ("system", developer_txt), ("human", "{user_message}")]
    )
    llm = ChatGroq(model=model, temperature=0.1, top_p=0.8, max_tokens=200)
    msgs = prompt.format_messages(user_message=user_txt)
    resp = llm.invoke(msgs).content.strip()

    # Parse simple key markers
    # Expected: DECISION=YES; ANSWER='...'
    # or: DECISION=NO; REASON='...'
    result = {"decision": "NO", "answer": None, "raw": resp}
    if "DECISION=YES" in resp:
        result["decision"] = "YES"
        # naive extract
        if "ANSWER='" in resp:
            try:
                result["answer"] = resp.split("ANSWER='", 1)[1].rsplit("'", 1)[0]
            except Exception:
                result["answer"] = resp
    return result
