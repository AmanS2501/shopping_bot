import os
from dotenv import load_dotenv
load_dotenv()

from collectors.pdf_collector import PDFCollector
from collectors.json_collector import JSONCollector
from cleaning.cleaner import TextCleaner
from chunking.chunker import Chunker
from embeddings.embedder import Embedder
from embeddings.chromadb_embed import ChromaDBEmbedder


def get_data_directory():
    default_dir = "data/"
    print(f"Default data directory is set to '{default_dir}'.")
    user_response = input("Is your data directory correct and contains all files? (yes/no): ").strip().lower()
    while user_response not in {"yes", "y", "no", "n"}:
        user_response = input("Please answer with 'yes' or 'no': ").strip().lower()

    if user_response in {"no", "n"}:
        print("Please enter the full or relative path to your data directory.")
        print("Example inputs: 'data', './data', '/home/user/mydata', 'C:\\Users\\DataFolder'")
        new_dir = input("Enter data directory path: ").strip()
        while not os.path.isdir(new_dir):
            print(f"Directory '{new_dir}' does not exist.")
            new_dir = input("Please re-enter a valid data directory path: ").strip()
        return new_dir
    else:
        if not os.path.isdir(default_dir):
            print(f"Warning: Default directory '{default_dir}' does not exist.")
        return default_dir


def main():
    print("[PIPELINE] Starting data collection pipeline...")

    data_dir = get_data_directory()

    pdf_collector = PDFCollector()
    json_collector = JSONCollector()

    pdf_documents = pdf_collector.load(data_dir)
    print(f"[PIPELINE] Collected {len(pdf_documents)} raw PDF documents")

    json_documents = json_collector.load(data_dir)
    print(f"[PIPELINE] Collected {len(json_documents)} raw JSON documents")

    all_raw_documents = pdf_documents + json_documents
    print(f"[PIPELINE] Total collected raw documents: {len(all_raw_documents)}")

    cleaner = TextCleaner()
    cleaned_documents = cleaner.clean_documents(all_raw_documents)
    cleaning_stats = cleaner.get_cleaning_stats(all_raw_documents, cleaned_documents)
    print(f"[PIPELINE] Cleaned documents: {len(cleaned_documents)}")
    print(f"[PIPELINE] Cleaning stats: {cleaning_stats}")

    chunker = Chunker()
    chunked_documents = chunker.chunk_documents(cleaned_documents)
    print(f"[PIPELINE] Chunked documents: {len(chunked_documents)}")

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
