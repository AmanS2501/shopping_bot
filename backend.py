from fastapi import FastAPI, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

import os

# Import your pipeline modules
from collectors.pdf_collector import PDFCollector
from collectors.json_collector import JSONCollector
from cleaning.cleaner import TextCleaner
from chunking.chunker import Chunker
from embeddings.embedder import Embedder
from embeddings.chromadb_embed import ChromaDBEmbedder

app = FastAPI(
    title="RAG Pipeline API",
    description="Backend API for document RAG pipeline",
    version="1.0.0"
)
# Enable CORS for development/testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to frontend origin in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cache (for session efficiency)
pipeline_cache = {}

class PipelineStats(BaseModel):
    pdf_count: int
    json_count: int
    total_count: int
    cleaning_stats: dict
    chunk_count: int

class SearchResult(BaseModel):
    metadata: dict
    page_content: str

@app.post("/run_pipeline", response_model=PipelineStats)
def run_pipeline(data_dir: str = Body(..., embed=True)):
    """
    Trigger the whole pipeline on data_dir. Collects PDFs/JSONs, cleans, chunks, embeds.
    """
    if not os.path.isdir(data_dir):
        return PipelineStats(
            pdf_count=0,
            json_count=0,
            total_count=0,
            cleaning_stats={},
            chunk_count=0
        )
    pdf_collector = PDFCollector()
    json_collector = JSONCollector()
    pdf_docs = pdf_collector.load(data_dir)
    json_docs = json_collector.load(data_dir)
    all_docs = pdf_docs + json_docs

    cleaner = TextCleaner()
    cleaned_docs = cleaner.clean_documents(all_docs)
    stats = cleaner.get_cleaning_stats(all_docs, cleaned_docs)

    chunker = Chunker()
    chunked_docs = chunker.chunk_documents(cleaned_docs)

    # Cache results
    pipeline_cache[data_dir] = {
        "pdf_docs": pdf_docs,
        "json_docs": json_docs,
        "cleaned_docs": cleaned_docs,
        "chunked_docs": chunked_docs,
        "stats": stats
    }

    return PipelineStats(
        pdf_count=len(pdf_docs),
        json_count=len(json_docs),
        total_count=len(all_docs),
        cleaning_stats=stats,
        chunk_count=len(chunked_docs)
    )

@app.get("/sample_docs", response_model=List[SearchResult])
def sample_docs(data_dir: str = Query(...), n: int = Query(5)):
    """
    Returns up to n sample cleaned documents from last run of pipeline.
    """
    cache = pipeline_cache.get(data_dir)
    if not cache:
        return []
    docs = cache["cleaned_docs"]
    return [
        SearchResult(metadata=doc.metadata, page_content=doc.page_content)
        for doc in docs[:n]
    ]

@app.get("/sample_chunks", response_model=List[SearchResult])
def sample_chunks(data_dir: str = Query(...), n: int = Query(5)):
    """
    Returns up to n sample chunked documents from last run of pipeline.
    """
    cache = pipeline_cache.get(data_dir)
    if not cache:
        return []
    docs = cache["chunked_docs"]
    return [
        SearchResult(metadata=doc.metadata, page_content=doc.page_content)
        for doc in docs[:n]
    ]

@app.post("/semantic_search", response_model=List[SearchResult])
def semantic_search(
    data_dir: str = Body(..., embed=True),
    query: str = Body(..., embed=True),
    k: int = Body(5, embed=True)
):
    """
    Perform a vector DB semantic retrieval for query string.
    """
    cache = pipeline_cache.get(data_dir)
    if not cache or not cache["chunked_docs"]:
        return []

    embedder = Embedder()
    chroma_db_embedder = ChromaDBEmbedder(persist_directory="chromadb_store")
    chroma_db_embedder.store_embeddings(embedder, cache["chunked_docs"], collection_name="rag_collection")
    results = chroma_db_embedder.similarity_search(query, embedder, k=k)

    return [
        SearchResult(metadata=res.metadata, page_content=res.page_content)
        for res in results
    ]

@app.get("/")
def hello():
    return {"status": "OK", "message": "RAG Pipeline backend is running!"}
