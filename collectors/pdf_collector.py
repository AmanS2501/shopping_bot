# from pathlib import Path
# from dotenv import load_dotenv
# load_dotenv()
# import requests
# from io import BytesIO
# from langchain_core.documents import Document
# import PyPDF2
# import json
# import uuid

# HEADERS = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
# }

# class PDFCollector:


#     def __init__(self, backup_path: str = "collectors/pdf_extracted_backup.jsonl"):
#         self.backup_path = backup_path

#     def read_pdf_file(self, file_path: str) -> str:
#         """Extract raw text from a local PDF file."""
#         try:
#             print(f"[INFO] Reading PDF: {file_path}")
#             text = ""
#             with open(file_path, "rb") as file:
#                 pdf_reader = PyPDF2.PdfReader(file)
#                 for page_idx, page in enumerate(pdf_reader.pages):
#                     page_text = page.extract_text()
#                     if page_text:
#                         text += page_text + "\n"
#             return text
#         except Exception as e:
#             print(f"[ERROR] Failed to read PDF {file_path}: {e}")
#             return ""

#     def read_pdf_from_url(self, url: str) -> str:

#         try:
#             print(f"[INFO] Downloading PDF from URL: {url}")
#             response = requests.get(url, headers=HEADERS, timeout=10)
#             response.raise_for_status()
#             pdf_stream = BytesIO(response.content)
#             pdf_reader = PyPDF2.PdfReader(pdf_stream)
#             text = ""
#             for page_idx, page in enumerate(pdf_reader.pages):
#                 page_text = page.extract_text()
#                 if page_text:
#                     text += page_text + "\n"
#             return text
#         except Exception as e:
#             print(f"[ERROR] Failed to download/read PDF from URL {url}: {e}")
#             return ""

#     def fetch_pdf_content(self, file_path_or_url: str) -> str:

#         from urllib.parse import urlparse
#         is_url = file_path_or_url.startswith("http://") or file_path_or_url.startswith("https://")
#         file_extension = Path(urlparse(file_path_or_url).path).suffix.lower()

#         if file_extension != '.pdf':
#             print(f"[ERROR] Unsupported file type for PDFCollector: {file_extension}")
#             return ""

#         if is_url:
#             return self.read_pdf_from_url(file_path_or_url)
#         else:
#             return self.read_pdf_file(file_path_or_url)

#     def load(self, source_list):

#         docs = []
#         with open(self.backup_path, "w", encoding="utf-8") as backup_file:
#             for src in source_list:
#                 text = self.fetch_pdf_content(src)
#                 if text:
#                     doc_id = str(uuid.uuid4())
#                     metadata = {
#                         "source": src,
#                         "file_extension": ".pdf",
#                     }
#                     backup_entry = {
#                         "id": doc_id,
#                         "page_content": text,
#                         "metadata": metadata
#                     }
#                     json.dump(backup_entry, backup_file, ensure_ascii=False)
#                     backup_file.write("\n")
#                     docs.append(Document(page_content=text, metadata=metadata))
#         print(f"[INFO] Loaded and backed up {len(docs)} PDF documents")
#         return docs



from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import requests
from io import BytesIO
from langchain_core.documents import Document
import PyPDF2
import json
import uuid
import os

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

class PDFCollector:
    def __init__(self, backup_path: str = "collectors/pdf_extracted_backup.jsonl"):
        self.backup_path = backup_path

    def read_pdf_file(self, file_path: str) -> str:
        try:
            print(f"[INFO] Reading PDF: {file_path}")
            text = ""
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
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
            for page in pdf_reader.pages:
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
        # Only process .pdf files or URLs ending in .pdf/.PDF
        if file_extension not in ['.pdf', '.PDF']:
            print(f"[ERROR] Unsupported file type for PDFCollector: {file_extension}")
            return ""
        if is_url:
            return self.read_pdf_from_url(file_path_or_url)
        else:
            return self.read_pdf_file(file_path_or_url)

    def _get_pdf_files_in_directory(self, directory_path: str):
        """Return all unique PDF files (.pdf or .PDF) in the given directory path or warn if not valid."""
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            print(f"[ERROR] Directory does not exist: {directory_path}")
            return []
        pdf_files = set()
        for ext in ("*.pdf", "*.PDF"):
            for f in Path(directory_path).glob(ext):
                if f.is_file():
                    pdf_files.add(str(f.resolve()).lower())  # use resolved absolute path, lower-cased
        unique_files = list(pdf_files)
        if not unique_files:
            print(f"[ERROR] No PDF files found in directory: {directory_path}")
        return unique_files


    def load(self, sources):
        """
        Loads PDFs from a directory, a single file, or a list of file paths/URLs (no duplicates).
        Returns list of Document objects and backs up each extraction to a JSONL file.
        """
        docs = []

        print("[DEBUG] Argument received:", sources)
        if isinstance(sources, str):
            print("[DEBUG] Absolute path:", os.path.abspath(sources))
            print("[DEBUG] Is dir?", os.path.isdir(sources))

        file_list = []

        # Directory input (extract all pdf/pdf files)
        if isinstance(sources, str) and os.path.isdir(sources):
            file_list = self._get_pdf_files_in_directory(sources)
            if not file_list:
                print(f"[VALIDATION] No valid PDF files to process in folder '{sources}'.")
                return []

        # Single PDF file (not a dir)
        elif isinstance(sources, str) and sources.lower().endswith('.pdf') and not os.path.isdir(sources):
            file_list = [sources]

        # List input (only unique files)
        elif isinstance(sources, list):
            seen = set()
            file_list = []
            for item in sources:
                if isinstance(item, str) and (item.lower().endswith(".pdf") or item.lower().endswith(".PDF") or item.startswith("http")):
                    if item not in seen:
                        file_list.append(item)
                        seen.add(item)
            if not file_list:
                print("[ERROR] No valid PDF files or URLs in the list to process.")
                return []

        else:
            print("[ERROR] 'sources' must be a directory path, a PDF file path, or a list of PDF paths/URLs.")
            return []

        if not file_list:
            print("[VALIDATION] No PDF files found. Aborting PDF extraction.")
            return []

        print(f"[INFO] PDF files to process: {file_list}")
        with open(self.backup_path, "w", encoding="utf-8") as backup_file:
            for src in file_list:
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
