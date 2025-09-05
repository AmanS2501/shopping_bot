import os
from dotenv import load_dotenv
load_dotenv()

from collectors.pdf_collector import PDFCollector
from collectors.json_collector import JSONCollector
from cleaning.cleaner import TextCleaner
from chunking.chunker import Chunker
from embeddings.embedder import Embedder
from embeddings.chromadb_embed import ChromaDBEmbedder


PDF_SOURCES = [
    "data/FAQs.pdf",

]

JSON_SOURCES = [
    "data/sample.json",

]

def main():
    print("[PIPELINE] Starting data collection pipeline...")

    # Step 1: PDF Data Collection
    pdf_collector = PDFCollector()
    pdf_documents = pdf_collector.load(PDF_SOURCES)
    print(f"[PIPELINE] Collected {len(pdf_documents)} raw PDF documents")

    # Step 2: JSON Data Collection
    json_collector = JSONCollector()
    json_documents = json_collector.load(JSON_SOURCES)
    print(f"[PIPELINE] Collected {len(json_documents)} raw JSON documents")

    # Step 3: Combine and Clean Documents
    all_raw_documents = pdf_documents + json_documents
    print(f"[PIPELINE] Total collected raw documents: {len(all_raw_documents)}")

    cleaner = TextCleaner()
    cleaned_documents = cleaner.clean_documents(all_raw_documents)
    cleaning_stats = cleaner.get_cleaning_stats(all_raw_documents, cleaned_documents)
    print(f"[PIPELINE] Cleaned documents: {len(cleaned_documents)}")
    print(f"[PIPELINE] Cleaning stats: {cleaning_stats}")

    # Step 4: Chunking
    chunker = Chunker()
    chunked_documents = chunker.chunk_documents(cleaned_documents)
    print(f"[PIPELINE] Chunked documents: {len(chunked_documents)}")

    # Step 5: Embedding and ChromaDB Storage
    embedder = Embedder()
    chroma_db_embedder = ChromaDBEmbedder(persist_directory="chromadb_store")
    chroma_vectorstore = chroma_db_embedder.store_embeddings(embedder, chunked_documents, collection_name="rag_collection")


    sample_query = "How do I reset my password?"
    results = chroma_db_embedder.similarity_search(sample_query, embedder, k=5)
    for i, result in enumerate(results):
        print(f"[RESULT {i}] Content: {result.page_content}, Metadata: {result.metadata}")

    print("[PIPELINE] Pipeline run completed (all stages)")

if __name__ == "__main__":
    main()
