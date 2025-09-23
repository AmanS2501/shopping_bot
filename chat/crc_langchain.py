# chat/crc_langchain.py
from __future__ import annotations
from typing import List, Dict, Any
import os, re, json

from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda, RunnableMap, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from .prompt_loader import PromptConfig
from retrieval.simple_retriever import load_chroma, retrieve_with_crossencoder_rerank

CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-oss-120b")

def _groq(temp=0.2, max_tokens=256):
    return ChatGroq(model=CHAT_MODEL, temperature=temp, top_p=0.8, max_tokens=max_tokens, frequency_penalty=0.2)

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

class CRC:
    def __init__(
        self,
        crc_prompt_path: str = "prompts/crc_prompts.json",
        chroma=None,
        crossencoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        pool_k: int = int(os.getenv("RETRIEVER_POOL_K", "60")),
        top_k: int = int(os.getenv("RETRIEVER_TOP_K", "5")),
    ):
        self.cfg = PromptConfig.load(crc_prompt_path)
        self.chroma = chroma
        self.crossencoder = crossencoder_model
        self.pool_k = pool_k
        self.top_k = top_k

        # Build refine prompt
        self.refine_prompt = ChatPromptTemplate.from_messages(
            [("system", "\n".join(self.cfg.data["refine"]["system"])), ("human", "{user_message}")]
        )
        # History-only answer prompt
        self.hist_prompt = ChatPromptTemplate.from_messages(
            [("system", "\n".join(self.cfg.data["history_answer"]["system"])), ("human", "{user_message}")]
        )
        # Context-only answer prompt
        self.ctx_prompt = ChatPromptTemplate.from_messages(
            [("system", "\n".join(self.cfg.data["context_answer"]["system"])), ("human", "{user_message}")]
        )

        # LLMs
        self.llm_refine = _groq(temp=0.1, max_tokens=120)
        self.llm_hist = _groq(temp=0.1, max_tokens=256)
        self.llm_ctx = _groq(temp=0.2, max_tokens=256)

        # ConversationalRetrievalChain for context stage
        # We will adapt it by injecting reranked docs and strict prompt for answering.
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.retriever = self.chroma.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.pool_k}
        )

        self.conv_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm_ctx,
            retriever=self.retriever,
            return_source_documents=True,
            verbose=False,
        )

        # Build runnables
        self.graph = self._build_graph()

    def _refine_step(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        question, history = inputs["question"], inputs["history"]
        user_msg = self.cfg.data["refine"]["user_template"].format(
            history=_format_history(history), question=question
        )
        msgs = self.refine_prompt.format_messages(user_message=user_msg)
        text = self.llm_refine.invoke(msgs).content.strip()

        out = {"route": "RETRIEVE", "query": question, "answer": None, "raw": text}
        if "ROUTE=HISTORY" in text and "ANSWER='" in text:
            try:
                ans = text.split("ANSWER='", 1)[1].rsplit("'", 1)[0]
                out["route"] = "HISTORY"
                out["answer"] = ans
                return {**inputs, "refine": out}
            except Exception:
                pass
        if "ROUTE=RETRIEVE" in text and "QUERY='" in text:
            try:
                q2 = text.split("QUERY='", 1)[1].rsplit("'", 1)[0]
                if q2: out["query"] = q2
            except Exception:
                pass
        return {**inputs, "refine": out}

    def _retrieve_step(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if inputs["refine"]["route"] == "HISTORY":
            return {**inputs, "docs": []}
        # get candidate docs from retriever
        docs = self.retriever.get_relevant_documents(inputs["refine"]["query"])
        # optional: apply CrossEncoder reranking
        try:
            docs = retrieve_with_crossencoder_rerank(
                query=inputs["refine"]["query"],
                chroma=self.chroma,
                crossencoder_model=self.crossencoder,
                pool_k=self.pool_k,
                top_k=self.top_k,
            )
        except Exception:
            # if reranker fails, fall back to retriever docs truncated to top_k
            docs = docs[: self.top_k]
        return {**inputs, "docs": docs}


    def _answer_step(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        question, history, docs, refine = inputs["question"], inputs["history"], inputs["docs"], inputs["refine"]

        if refine["route"] == "HISTORY" and refine["answer"]:
            # Optionally pass through hist_prompt to normalize tone
            user_msg = self.cfg.data["history_answer"]["user_template"].format(
                history=_format_history(history), question=question
            )
            msgs = self.hist_prompt.format_messages(user_message=user_msg)
            # If strict formatting desired, could ignore llm here and use refine answer directly
            final = refine["answer"]
            return {"answer": final, "docs": []}

        # Context-only answer
        user_msg = self.cfg.data["context_answer"]["user_template"].format(
            question=question, context=_join_context(docs)
        )
        msgs = self.ctx_prompt.format_messages(user_message=user_msg)
        final = self.llm_ctx.invoke(msgs).content
        return {"answer": final, "docs": docs}

    def _build_graph(self):
        return (
            RunnableMap({
                "question": lambda x: x["question"],
                "history": lambda x: x["history"],
            })
            | RunnableLambda(self._refine_step)
            | RunnableLambda(self._retrieve_step)
            | RunnableLambda(self._answer_step)
        )

    def invoke(self, question: str, history: List[dict]) -> Dict[str, Any]:
        return self.graph.invoke({"question": question, "history": history})
