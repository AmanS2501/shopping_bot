# chat/rag_stage.py
from typing import List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from .prompt_loader import PromptConfig

def join_context(docs: List[Document], max_chars: int = 7000) -> str:
    out, total = [], 0
    for d in docs:
        t = d.page_content
        if total + len(t) > max_chars:
            break
        total += len(t)
        out.append(t)
    return "\n\n---\n\n".join(out)

def run_rag_stage(question: str, docs: List[Document], model: str, prompt_path: str) -> str:
    cfg = PromptConfig.load(prompt_path)
    system_txt = "\n".join(cfg.data["prompt_shell"]["system"])
    developer_txt = "\n".join(cfg.data["prompt_shell"]["developer"])
    user_txt = cfg.render_user(question=question, context=join_context(docs))

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_txt), ("system", developer_txt), ("human", "{user_message}")]
    )
    llm = ChatGroq(model=model, temperature=0.2, top_p=0.8, max_tokens=256, frequency_penalty=0.2)
    msgs = prompt.format_messages(user_message=user_txt)
    return llm.invoke(msgs).content
