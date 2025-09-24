# import os
# from dotenv import load_dotenv
# load_dotenv()
# from chat.crc_chain import ConversationalRetrievalChain
# from collectors.pdf_collector import PDFCollector
# from collectors.json_collector import JSONCollector
# from cleaning.cleaner import TextCleaner
# from chunking.chunker import Chunker
# from embeddings.embedder import Embedder
# from embeddings.chromadb_embed import ChromaDBEmbedder
# from chat.history_stage import run_history_stage
# from chat.rag_stage import run_rag_stage
# # from router.query_router import classify
# from retrieval.simple_retriever import load_chroma, retrieve_with_crossencoder_rerank
# from chat.chatgroq_rag import chatgroq_answer

# PERSIST_DIR = "chromadb_store"
# COLLECTION_NAME = "rag_collection"
# CROSSENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# POOL_K = int(os.getenv("RETRIEVER_POOL_K", "40"))
# TOP_K = int(os.getenv("RETRIEVER_TOP_K", "5"))
# CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-oss-120b")
# HISTORY_PROMPT = os.getenv("HISTORY_PROMPT", "prompts/history_prompt.json")
# RAG_PROMPT = os.getenv("RAG_PROMPT", "prompts/rag_prompt.json")


# def format_sources(docs):
#     srcs = []
#     for d in docs:
#         title = d.metadata.get("title") or d.metadata.get("source") or "doc"
#         srcs.append(title)
#     return srcs

# def format_history(history):
#     """
#     Convert chat_history list into a compact text block
#     to condition follow-up turns. Each item is a dict with keys:
#     {'role': 'user'|'assistant', 'content': '...'}
#     """
#     if not history:
#         return ""
#     lines = []
#     for turn in history:
#         role = "User" if turn["role"] == "user" else "Assistant"
#         lines.append(f"{role}: {turn['content']}")
#     return "\n".join(lines)


# def main():
#     print("[PIPELINE] Starting data collection pipeline...")

#     data_dir = os.getenv("DATA_DIR", "data")
#     # Optional first-run ingestion (uncomment to enable)
#     # pdf_documents = PDFCollector().load(data_dir)
#     # json_documents = JSONCollector().load(data_dir)
#     # all_raw_documents = pdf_documents + json_documents
#     # cleaned = TextCleaner().clean_documents(all_raw_documents)
#     # chunks = Chunker().chunk_documents(cleaned)
#     embedder = Embedder()
#     persist_abs = os.path.abspath(PERSIST_DIR)
#     # ChromaDBEmbedder(persist_directory=persist_abs).store_embeddings(
#     #     embedder, chunks, collection_name=COLLECTION_NAME
#     # )

#     vs = load_chroma(persist_abs, embedder.embedder, COLLECTION_NAME)
#     try:
#         print(f"[DEBUG] Reopened collection count: {vs._collection.count()}")
#     except Exception:
#         pass

#     crc = ConversationalRetrievalChain(
#         crc_prompt_path=os.getenv("CRC_PROMPT", "prompts/crc_prompts.json"),
#         crossencoder_model=CROSSENCODER_MODEL,
#         pool_k=POOL_K,
#         top_k=TOP_K,
#     )

#     chat_history: List[dict] = []
#     print("\n[CHAT] Ask your shopping questions. Type 'exit' to quit.\n")
#     while True:
#         user_q = input("Enter your query: ").strip()
#         if user_q.lower() in {"exit", "quit"}:
#             print("[CHAT] Bye.")
#             break

#         out = crc.invoke(question=user_q, history=chat_history, chroma=vs)
#         print("\n[ANSWER]\n", out["answer"])
#         print("\n[SOURCES]")
#         if out["docs"]:
#             for i, d in enumerate(out["docs"]):
#                 print(f"{i+1}. {d.metadata}")
#         else:
#             print("-")

#         chat_history.append({"role": "user", "content": user_q})
#         chat_history.append({"role": "assistant", "content": out["answer"]})
#     print("[PIPELINE] Done")



# if __name__ == "__main__":
#     main()


# #conversational retrieval chain with history and RAG

# # from langchain_core.runnables.history import RunnableWithMessageHistory
# # from langchain_core.chat_history import BaseChatMessageHistory
# # from langchain_community.chat_message_histories import ChatMessageHistory
# # from langchain_core.callbacks import CallbackManager

# # runnables



import os
from dotenv import load_dotenv
load_dotenv()

from embeddings.embedder import Embedder
from embeddings.chromadb_embed import ChromaDBEmbedder
from retrieval.simple_retriever import load_chroma, retrieve_with_crossencoder_rerank

from chat.crc_langchain import CRC

PERSIST_DIR = "chromadb_store"
COLLECTION_NAME = "rag_collection"
CROSSENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
POOL_K = int(os.getenv("RETRIEVER_POOL_K", "60"))
TOP_K = int(os.getenv("RETRIEVER_TOP_K", "5"))

def main():
    print("[PIPELINE] Starting data collection pipeline...")

    embedder = Embedder()
    persist_abs = os.path.abspath(PERSIST_DIR)
    vs = load_chroma(persist_abs, embedder.embedder, COLLECTION_NAME)
    try:
        print(f"[DEBUG] Reopened collection count: {vs._collection.count()}")
    except Exception:
        pass

    crc = CRC(
        crc_prompt_path=os.getenv("CRC_PROMPT", "prompts/crc_prompts.json"),
        chroma=vs,
        crossencoder_model=CROSSENCODER_MODEL,
        pool_k=POOL_K,
        top_k=TOP_K,
    )

    chat_history = []
    print("\n[CHAT] Ask your shopping questions. Type 'exit' to quit.\n")
    while True:
        user_q = input("Enter your query: ").strip()
        if user_q.lower() in {"exit", "quit"}:
            print("[CHAT] Bye.")
            break

        out = crc.invoke(question=user_q, history=chat_history)
        print("\n[ANSWER]\n", out["answer"])
        print("\n[SOURCES]")
        if out["docs"]:
            for i, d in enumerate(out["docs"]):
                print(f"{i+1}. {d.metadata}")
        else:
            print("-")

        chat_history.append({"role": "user", "content": user_q})
        chat_history.append({"role": "assistant", "content": out["answer"]})

if __name__ == "__main__":
    main()
