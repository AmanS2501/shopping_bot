# chat/crc_chain.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import os, re, json

from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnableMap
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from retrieval.simple_retriever import retrieve_with_crossencoder_rerank
from .prompt_loader import PromptConfig  # uses brace-escaping in your project

CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-oss-120b")

def _format_history(history: List[dict], max_chars: int = 4000) -> str:
    out, total = [], 0
    for turn in history[-20:]:
        prefix = "User:" if turn.get("role") == "user" else "Assistant:"
        line = f"{prefix} {turn.get('content','')}".strip()
        if total + len(line) > max_chars: break
        total += len(line)
        out.append(line)
    return "\n".join(out)

def _join_context(docs: List[Document], max_chars: int = 7000) -> str:
    out, total = [], 0
    for d in docs:
        t = d.page_content
        if total + len(t) > max_chars: break
        total += len(t)
        out.append(t)
    return "\n\n---\n\n".join(out)

def _build_prompt(block: Dict[str, Any]) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [("system", "\n".join(block["system"])), ("human", "{user_message}")]
    )

def _groq_llm(temp: float = 0.2, max_tokens: int = 256) -> ChatGroq:
    return ChatGroq(
        model=CHAT_MODEL,
        temperature=temp,
        top_p=0.8,
        max_tokens=max_tokens,
        frequency_penalty=0.2,
        presence_penalty=0.0,
    )

class ConversationalRetrievalChain:
    def __init__(
        self,
        crc_prompt_path: str = "prompts/crc_prompts.json",
        crossencoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        pool_k: int = int(os.getenv("RETRIEVER_POOL_K", "60")),
        top_k: int = int(os.getenv("RETRIEVER_TOP_K", "5")),
    ):
        self.cfg = PromptConfig.load(crc_prompt_path)
        self.crossencoder_model = crossencoder_model
        self.pool_k = pool_k
        self.top_k = top_k

        # Subprompts
        self.refine_prompt = _build_prompt(self.cfg.data["refine"])
        self.history_answer_prompt = _build_prompt(self.cfg.data["answer_from_history"])
        self.context_answer_prompt = _build_prompt(self.cfg.data["answer_from_context"])

        self.llm_refine = _groq_llm(temp=0.1, max_tokens=120)
        self.llm_history = _groq_llm(temp=0.1, max_tokens=256)
        self.llm_context = _groq_llm(temp=0.2, max_tokens=256)

    def _refine_step(self, question: str, history: List[dict]) -> Dict[str, Any]:
        user_msg = self.cfg.data["refine"]["user_template"].format(
            history=_format_history(history), question=question
        )
        msgs = self.refine_prompt.format_messages(user_message=user_msg)
        resp = self.llm_refine.invoke(msgs).content.strip()

        # Parse ROUTE
        out = {"route": "RETRIEVE", "query": question, "answer": None, "raw": resp}
        if "ROUTE=HISTORY" in resp and "ANSWER='" in resp:
            try:
                ans = resp.split("ANSWER='", 1)[1].rsplit("'", 1)[0]
                out["route"] = "HISTORY"
                out["answer"] = ans
                return out
            except Exception:
                pass
        if "ROUTE=RETRIEVE" in resp and "QUERY='" in resp:
            try:
                q2 = resp.split("QUERY='", 1)[1].rsplit("'", 1)[0]
                if q2: out["query"] = q2
            except Exception:
                pass
        return out

    def _answer_from_history(self, question: str, history: List[dict]) -> str:
        user_msg = self.cfg.data["answer_from_history"]["user_template"].format(
            history=_format_history(history), question=question
        )
        msgs = self.history_answer_prompt.format_messages(user_message=user_msg)
        return self.llm_history.invoke(msgs).content

    def _answer_from_context(self, question: str, docs: List[Document]) -> str:
        user_msg = self.cfg.data["answer_from_context"]["user_template"].format(
            question=question, context=_join_context(docs)
        )
        msgs = self.context_answer_prompt.format_messages(user_message=user_msg)
        return self.llm_context.invoke(msgs).content

    def invoke(self, question: str, history: List[dict], chroma) -> Dict[str, Any]:
        # 1) History-aware refine
        refine = self._refine_step(question, history)

        if refine["route"] == "HISTORY" and refine["answer"]:
            return {"answer": refine["answer"], "docs": []}

        # 2) Retrieve with refined query
        docs = retrieve_with_crossencoder_rerank(
            query=refine["query"],
            chroma=chroma,
            crossencoder_model=self.crossencoder_model,
            pool_k=self.pool_k,
            top_k=self.top_k,
        )

        # 3) Answer from context only
        answer = self._answer_from_context(question, docs)
        return {"answer": answer, "docs": docs}
