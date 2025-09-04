import os
from dotenv import load_dotenv
load_dotenv()

from collectors.pdf_collector import PDFCollector
from collectors.json_collector import JSONCollector
from cleaning.cleaner import TextCleaner
from chunking.chunker import Chunker


PDF_SOURCES = [
    "data/FAQs.pdf",

]

JSON_SOURCES = [
    "data/sample.json",

]

def main():
    print("[PIPELINE] Starting data collection pipeline...")

    # Step 1a: PDF Data Collection
    pdf_collector = PDFCollector()
    pdf_documents = pdf_collector.load(PDF_SOURCES)
    print(f"[PIPELINE] Collected {len(pdf_documents)} raw PDF documents")

    # Step 1b: JSON Data Collection
    json_collector = JSONCollector()
    json_documents = json_collector.load(JSON_SOURCES)
    print(f"[PIPELINE] Collected {len(json_documents)} raw JSON documents")

    # Combine documents
    all_raw_documents = pdf_documents + json_documents
    print(f"[PIPELINE] Total collected raw documents: {len(all_raw_documents)}")

    # Step 2: Data Cleaning
    cleaner = TextCleaner()
    cleaned_documents = cleaner.clean_documents(all_raw_documents)
    cleaning_stats = cleaner.get_cleaning_stats(all_raw_documents, cleaned_documents)
    print(f"[PIPELINE] Cleaned documents: {len(cleaned_documents)}")
    print(f"[PIPELINE] Cleaning stats: {cleaning_stats}")

    # Step 3: Chunking
    chunker = Chunker()
    chunked_documents = chunker.chunk_documents(cleaned_documents)
    print(f"[PIPELINE] Chunked documents: {len(chunked_documents)}")



if __name__ == "__main__":
    main()
