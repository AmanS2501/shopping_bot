from typing import List
import os
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from .prompt_loader import PromptConfig

from typing import List
from langchain_core.documents import Document

def join_context(docs: List[Document], max_chars: int = 8000) -> str:
    out, total = [], 0
    for d in docs:
        t = d.page_content
        if total + len(t) > max_chars:
            break
        total += len(t)
        out.append(t)
    return "\n\n---\n\n".join(out)

def build_prompt(cfg: PromptConfig) -> ChatPromptTemplate:
    system_txt = cfg.render_system()
    developer_txt = cfg.render_developer()
    # MessagesPlaceholder inserts prior chat turns cleanly
    return ChatPromptTemplate.from_messages(
    [
    ("system", system_txt),
    ("system", developer_txt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{user_message}")
    ]
    )

def chatgroq_answer(
    question: str,
    docs: List[Document],
    history: List[dict],
    prompt_path: str = "prompts/shopping_bot_prompt.json",
    model: str | None = None,
    temperature: float = 0.0,
    ) -> str:
    cfg = PromptConfig.load(prompt_path)
    ctx = join_context(docs)
    user_msg = cfg.render_user(question=question, context=ctx)

    chat_model = model or os.getenv("CHAT_MODEL", "openai/gpt-oss-120b")
    llm = ChatGroq(model=chat_model, temperature=temperature)

    prompt = build_prompt(cfg)
    # Convert simple history dicts to LangChain messages
    lc_history = []
    for turn in history[-12:]:
        if turn["role"] == "user":
            lc_history.append(HumanMessage(content=turn["content"]))
        else:
            lc_history.append(SystemMessage(content=f"Assistant: {turn['content']}"))

    msgs = prompt.format_messages(history=lc_history, user_message=user_msg)
    resp = llm.invoke(msgs)
    return resp.content
