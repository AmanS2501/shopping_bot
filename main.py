
import os
from dotenv import load_dotenv
load_dotenv()


# Existing pipeline imports
from collectors.pdf_collector import PDFCollector
from collectors.json_collector import JSONCollector
from cleaning.cleaner import TextCleaner
from chunking.chunker import Chunker
from embeddings.embedder import Embedder
from embeddings.chromadb_embed import ChromaDBEmbedder


# New imports for retrieval + ChatGroq
from retrieval.simple_retriever import load_chroma, retrieve_with_crossencoder_rerank  # modern reopen [1]
from chat.chatgroq_rag import chatgroq_answer  # ChatGroq chain 


# ----------------------------
# Config
# ----------------------------
PERSIST_DIR = "chromadb_store"
COLLECTION_NAME = "rag_collection"
CROSSENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # small, fast reranker 
POOL_K = 40
TOP_K = 5


def format_history(history):
    """
    Convert chat_history list into a compact text block
    to condition follow-up turns. Each item is a dict with keys:
    {'role': 'user'|'assistant', 'content': '...'}
    """
    if not history:
        return ""
    lines = []
    for turn in history:
        role = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


def main():
    print("[PIPELINE] Starting data collection pipeline...")

    # 1) Collect
    data_dir = os.getenv("DATA_DIR", "data")
    # pdf_collector = PDFCollector()
    # json_collector = JSONCollector()
    # pdf_documents = pdf_collector.load(data_dir)
    # print(f"[PIPELINE] Collected {len(pdf_documents)} raw PDF documents")
    # json_documents = json_collector.load(data_dir)
    # print(f"[PIPELINE] Collected {len(json_documents)} raw JSON documents")
    # all_raw_documents = pdf_documents + json_documents
    # print(f"[PIPELINE] Total collected raw documents: {len(all_raw_documents)}")

    # # 2) Clean
    # cleaner = TextCleaner()
    # cleaned_documents = cleaner.clean_documents(all_raw_documents)
    # cleaning_stats = cleaner.get_cleaning_stats(all_raw_documents, cleaned_documents)
    # print(f"[PIPELINE] Cleaned documents: {len(cleaned_documents)}")
    # print(f"[PIPELINE] Cleaning stats: {cleaning_stats}")

    # # 3) Chunk
    # chunker = Chunker()
    # chunked_documents = chunker.chunk_documents(cleaned_documents)
    # print(f"[PIPELINE] Chunked documents: {len(chunked_documents)}")

    # # 4) Embed + Persist to Chroma (write with explicit collection name)
    embedder = Embedder()
    persist_abs = os.path.abspath(PERSIST_DIR)
    # chroma_db_embedder = ChromaDBEmbedder(persist_directory=persist_abs)
    # chroma_vectorstore = chroma_db_embedder.store_embeddings(
    #     embedder,
    #     chunked_documents,
    #     collection_name=COLLECTION_NAME
    # )
    # print(f"[CHROMADB] Embeddings stored and collection persisted at: {persist_abs}")

    # 5) Reopen same collection with same embedding instance (critical)
    vs = load_chroma(persist_abs, embedder.embedder, COLLECTION_NAME)
    try:
        # Internal count to validate vectors exist
        cnt = vs._collection.count()
        print(f"[DEBUG] Reopened collection count: {cnt}")
    except Exception:
        print("[DEBUG] Could not read collection count (non-fatal).")  # best-effort only

    # 6) Retrieve with CrossEncoder rerank + ChatGroq answer
    pool_k = int(os.getenv("RETRIEVER_POOL_K", POOL_K))
    top_k = int(os.getenv("RETRIEVER_TOP_K", TOP_K))

    # Follow-up chat loop with conversational context
    chat_history = []
    print("\n[CHAT] Ask questions about your data. Type 'exit' to quit.\n")
    while True:
        user_q = input("Enter your query: ").strip()
        if user_q.lower() in {"exit", "quit"}:
            print("[CHAT] Exiting chat loop.")
            break

        # Retrieve fresh context per turn
        docs = retrieve_with_crossencoder_rerank(
            query=user_q,
            chroma=vs,
            crossencoder_model=CROSSENCODER_MODEL,
            pool_k=pool_k,
            top_k=top_k,
        )
        print(f"[RETRIEVER] Reranked docs: {len(docs)}")
        if not docs:
            print("[WARN] No docs retrieved. Confirm collection path/name and embedding instance are consistent.")

        # Provide conversation context to the LLM along with retrieved context
        history_block = format_history(chat_history)
        question_with_history = (
            f"{user_q}\n\n"
            f"(Conversation so far, for context):\n{history_block}" if history_block else user_q
        )

        answer = chatgroq_answer(question_with_history, docs, model="openai/gpt-oss-120b", temperature=0.0)
        print("\n[ANSWER]\n", answer)
        print("\n[SOURCES]")
        for i, d in enumerate(docs):
            print(f"{i+1}. {d.metadata}")
        print("\n")  # spacer for readability

        # Update in-memory history
        chat_history.append({"role": "user", "content": user_q})
        chat_history.append({"role": "assistant", "content": answer})

    print("[PIPELINE] Pipeline run completed (all stages)")


if __name__ == "__main__":
    main()
