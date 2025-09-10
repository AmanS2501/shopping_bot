from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
import os

class ChromaDBEmbedder:


    def __init__(self, persist_directory: str = "chromadb_store"):
        self.persist_directory = persist_directory
        # Initialize vector store, persistent on disk
        os.makedirs(self.persist_directory, exist_ok=True)
        self.vectorstore = None

    def store_embeddings(self, embedder, documents: List[Document], collection_name: str = "rag_collection"):

        if not documents:
            print("[CHROMADB] No documents to embed/store.")
            return None

        try:
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            embeddings = embedder.embedder.embed_documents(texts)

            self.vectorstore = Chroma.from_documents(
                documents=[Document(page_content=text, metadata=meta) for text, meta in zip(texts, metadatas)],
                embedding=embedder.embedder,
                collection_name=collection_name,
                persist_directory=self.persist_directory
            )

            print(f"[CHROMADB] Stored {len(texts)} embeddings in Chroma collection '{collection_name}'")
            return self.vectorstore
        except Exception as e:
            print(f"[CHROMADB] Failed to store embeddings: {e}")
            return None

    def similarity_search(self, query: str, embedder, k: int = 5):
        if self.vectorstore is None:
            print("[CHROMADB] Vectorstore not initialized.")
            return []
        try:
            query_emb = embedder.embed_query(query)
            results = self.vectorstore.similarity_search_by_vector(query_emb, k=k)
            print(f"[CHROMADB] Found {len(results)} results for query.")
            return results
        except Exception as e:
            print(f"[CHROMADB] Similarity search failed: {e}")
            return []
