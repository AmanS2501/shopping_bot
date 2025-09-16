import os
from typing import List, Tuple
from langchain_chroma import Chroma  # [3][2]
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder  # [9][10]

def load_chroma(persist_directory: str, embedding: Embeddings, collection_name: str) -> Chroma:
    persist_abs = os.path.abspath(persist_directory)
    print(f"[DEBUG] Reopen Chroma @ {persist_abs} collection={collection_name}")
    return Chroma(
        persist_directory=persist_abs,
        embedding_function=embedding,
        collection_name=collection_name,  # CRITICAL: same collection [2]
    )

def _pairwise_inputs(query: str, docs: List[Document]) -> List[Tuple[str, str]]:
    return [(query, d.page_content) for d in docs]

def retrieve_with_crossencoder_rerank(
    query: str,
    chroma: Chroma,
    crossencoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    pool_k: int = 40,
    top_k: int = 5,
) -> List[Document]:
    if not query or not query.strip():
        print("[WARN] Empty query provided to retriever.")
        return []
    # Verify collection not empty
    try:
        cnt = chroma._collection.count()
        print(f"[DEBUG] Collection count: {cnt}")
        if cnt == 0:
            print("[ERROR] Chroma collection is empty. Re-run embedding or fix collection name/path.")
            return []
    except Exception:
        pass

    pool_docs = chroma.similarity_search(query, k=pool_k)  # [2]
    if not pool_docs:
        print("[WARN] similarity_search returned 0 candidates.")
        return []

    reranker = CrossEncoder(crossencoder_model)
    scores = reranker.predict(_pairwise_inputs(query, pool_docs))  # [9][10]
    ranked = sorted(zip(pool_docs, scores), key=lambda x: float(x[1]), reverse=True)
    return [d for d, s in ranked[:top_k]]
