from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import requests
from io import BytesIO
from langchain_core.documents import Document
import PyPDF2
import json
import uuid

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

class PDFCollector:


    def __init__(self, backup_path: str = "collectors/pdf_extracted_backup.jsonl"):
        self.backup_path = backup_path

    def read_pdf_file(self, file_path: str) -> str:
        """Extract raw text from a local PDF file."""
        try:
            print(f"[INFO] Reading PDF: {file_path}")
            text = ""
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_idx, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            print(f"[ERROR] Failed to read PDF {file_path}: {e}")
            return ""

    def read_pdf_from_url(self, url: str) -> str:

        try:
            print(f"[INFO] Downloading PDF from URL: {url}")
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            pdf_stream = BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_stream)
            text = ""
            for page_idx, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except Exception as e:
            print(f"[ERROR] Failed to download/read PDF from URL {url}: {e}")
            return ""

    def fetch_pdf_content(self, file_path_or_url: str) -> str:

        from urllib.parse import urlparse
        is_url = file_path_or_url.startswith("http://") or file_path_or_url.startswith("https://")
        file_extension = Path(urlparse(file_path_or_url).path).suffix.lower()

        if file_extension != '.pdf':
            print(f"[ERROR] Unsupported file type for PDFCollector: {file_extension}")
            return ""

        if is_url:
            return self.read_pdf_from_url(file_path_or_url)
        else:
            return self.read_pdf_file(file_path_or_url)

    def load(self, source_list):

        docs = []
        with open(self.backup_path, "w", encoding="utf-8") as backup_file:
            for src in source_list:
                text = self.fetch_pdf_content(src)
                if text:
                    doc_id = str(uuid.uuid4())
                    metadata = {
                        "source": src,
                        "file_extension": ".pdf",
                    }
                    backup_entry = {
                        "id": doc_id,
                        "page_content": text,
                        "metadata": metadata
                    }
                    json.dump(backup_entry, backup_file, ensure_ascii=False)
                    backup_file.write("\n")
                    docs.append(Document(page_content=text, metadata=metadata))
        print(f"[INFO] Loaded and backed up {len(docs)} PDF documents")
        return docs
