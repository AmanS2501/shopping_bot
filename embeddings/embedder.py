from typing import List
from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings

class Embedder:


    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.embedder = self._load_embedder(model_name, device)

    def _load_embedder(self, model_name: str, device: str):
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': device}
            )
            print(f"[EMBEDDER] Loaded model {model_name} on device {device}")
            return embeddings
        except Exception as e:
            print(f"[EMBEDDER] Failed to load embedding model: {e}")
            raise


    def embed_documents(self, documents: List[Document]) -> List[dict]:

        if not documents:
            return []
        try:
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            embeddings = self.embedder.embed_documents(texts)
            results = []
            for emb, meta, text in zip(embeddings, metadatas, texts):
                results.append({
                    "embedding": emb,         # vector
                    "metadata": meta,         # original metadata per chunk
                    "page_content": text
                })
            print(f"[EMBEDDER] Embedded {len(results)} documents")
            return results
        except Exception as e:
            print(f"[EMBEDDER] Failed to embed documents: {e}")
            return []

    def embed_query(self, query: str):

        try:
            return self.embedder.embed_query(query)
        except Exception as e:
            print(f"[EMBEDDER] Failed to embed query: {e}")
            return None
