# chat/chatgroq_rag.py

from typing import List
from langchain_groq import ChatGroq  # ChatGroq integration [13][14]
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

SYSTEM = (
    "You are a helpful assistant. Use the provided context to answer the user's question. "
    "If the answer is not present, say you don't know. Be concise and cite facts briefly."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:"),
    ]
)

def join_context(docs: List[Document], max_chars: int = 8000) -> str:
    out, total = [], 0
    for d in docs:
        t = d.page_content
        if total + len(t) > max_chars:
            break
        total += len(t)
        out.append(t)
    return "\n\n---\n\n".join(out)

def chatgroq_answer(question: str, docs: List[Document], model: str = "llama-3.1-70b-versatile", temperature: float = 0.0) -> str:
    ctx = join_context(docs)
    llm = ChatGroq(model=model, temperature=temperature)  # requires GROQ_API_KEY [13][14]
    msgs = PROMPT.format_messages(question=question, context=ctx)
    resp = llm.invoke(msgs)
    return resp.content
