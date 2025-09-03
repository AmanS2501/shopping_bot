import os
from dotenv import load_dotenv
load_dotenv()
from collectors.pdf_collector import PDFCollector



PDF_SOURCES = [
    "data/FAQs.pdf",

]

def main():
    print("[PIPELINE] Starting data collection pipeline...")

    pdf_collector = PDFCollector()
    raw_documents = pdf_collector.load(PDF_SOURCES)
    print(f"[PIPELINE] Collected {len(raw_documents)} raw PDF documents")



if __name__ == "__main__":
    main()
