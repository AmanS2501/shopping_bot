# from pathlib import Path
# from dotenv import load_dotenv
# load_dotenv()
# import requests
# import json
# import uuid
# from langchain_core.documents import Document

# HEADERS = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
# }

# class JSONCollector:


#     def __init__(self, backup_path: str = "collectors/json_extracted_backup.jsonl"):
#         self.backup_path = backup_path

#     def read_json_file(self, file_path: str) -> list:

#         try:
#             print(f"[INFO] Reading JSON: {file_path}")
#             with open(file_path, "r", encoding="utf-8") as file:
#                 data = json.load(file)
#             return self._extract_texts(data)
#         except Exception as e:
#             print(f"[ERROR] Failed to read JSON {file_path}: {e}")
#             return []

#     def read_json_from_url(self, url: str) -> list:

#         try:
#             print(f"[INFO] Downloading JSON from URL: {url}")
#             response = requests.get(url, headers=HEADERS, timeout=10)
#             response.raise_for_status()
#             data = response.json()
#             return self._extract_texts(data)
#         except Exception as e:
#             print(f"[ERROR] Failed to download/read JSON from URL {url}: {e}")
#             return []

#     def fetch_json_content(self, file_path_or_url: str) -> list:

#         from urllib.parse import urlparse
#         is_url = file_path_or_url.startswith("http://") or file_path_or_url.startswith("https://")
#         file_extension = Path(urlparse(file_path_or_url).path).suffix.lower()

#         if file_extension != '.json':
#             print(f"[ERROR] Unsupported file type for JSONCollector: {file_extension}")
#             return []

#         if is_url:
#             return self.read_json_from_url(file_path_or_url)
#         else:
#             return self.read_json_file(file_path_or_url)

#     def _extract_texts(self, data) -> list:

#         texts = []
#         if isinstance(data, dict):
#             # If flat dict, collect all string values
#             for k, v in data.items():
#                 if isinstance(v, str):
#                     texts.append((v, {"field": k}))
#                 elif isinstance(v, list):
#                     for item in v:
#                         if isinstance(item, str):
#                             texts.append((item, {"field": k}))
#                         elif isinstance(item, dict):
#                             for kk, vv in item.items():
#                                 if isinstance(vv, str):
#                                     texts.append((vv, {"field": k, "subfield": kk}))
#         elif isinstance(data, list):
#             for item in data:
#                 if isinstance(item, str):
#                     texts.append((item, {}))
#                 elif isinstance(item, dict):
#                     for k, v in item.items():
#                         if isinstance(v, str):
#                             texts.append((v, {"field": k}))
#         return texts

#     def load(self, source_list):

#         docs = []
#         with open(self.backup_path, "w", encoding="utf-8") as backup_file:
#             for src in source_list:
#                 entries = self.fetch_json_content(src)
#                 for (text, submeta) in entries:
#                     if text:
#                         doc_id = str(uuid.uuid4())
#                         metadata = {
#                             "source": src,
#                             "file_extension": ".json",
#                         }
#                         metadata.update(submeta)
#                         backup_entry = {
#                             "id": doc_id,
#                             "page_content": text,
#                             "metadata": metadata
#                         }
#                         json.dump(backup_entry, backup_file, ensure_ascii=False)
#                         backup_file.write("\n")
#                         docs.append(Document(page_content=text, metadata=metadata))
#         print(f"[INFO] Loaded and backed up {len(docs)} JSON documents")
#         return docs




from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import requests
import json
import uuid
from langchain_core.documents import Document
import os

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

class JSONCollector:
    def __init__(self, backup_path: str = "collectors/json_extracted_backup.jsonl"):
        self.backup_path = backup_path

    def read_json_file(self, file_path: str):
        try:
            print(f"[INFO] Reading JSON: {file_path}")
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            return self._extract_texts(data)
        except Exception as e:
            print(f"[ERROR] Failed to read JSON {file_path}: {e}")
            return []

    def read_json_from_url(self, url: str):
        try:
            print(f"[INFO] Downloading JSON from URL: {url}")
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            data = response.json()
            return self._extract_texts(data)
        except Exception as e:
            print(f"[ERROR] Failed to download/read JSON from URL {url}: {e}")
            return []

    def _extract_texts(self, data):
        """
        Extract text content for docs from JSON data structures.
        Customize this method as needed for your JSON schemas.
        Returns list of (text, metadata) tuples.
        """
        texts = []
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, str):
                    texts.append((v, {"field": k}))
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, str):
                            texts.append((item, {"field": k}))
                        elif isinstance(item, dict):
                            for kk, vv in item.items():
                                if isinstance(vv, str):
                                    texts.append((vv, {"field": k, "subfield": kk}))
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    texts.append((item, {}))
                elif isinstance(item, dict):
                    for k, v in item.items():
                        if isinstance(v, str):
                            texts.append((v, {"field": k}))
        return texts

    def _get_json_files_in_directory(self, directory_path: str):
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            print(f"[ERROR] Directory does not exist: {directory_path}")
            return []
        json_files = set()
        for ext in ("*.json", "*.JSON"):
            for f in Path(directory_path).glob(ext):
                if f.is_file():
                    json_files.add(str(f.resolve()).lower())
        files = list(json_files)
        if not files:
            print(f"[ERROR] No JSON files found in directory: {directory_path}")
        return files

    def fetch_json_content(self, file_path_or_url: str):
        from urllib.parse import urlparse
        is_url = file_path_or_url.startswith("http://") or file_path_or_url.startswith("https://")
        file_extension = Path(urlparse(file_path_or_url).path).suffix.lower()

        if file_extension not in ['.json']:
            print(f"[ERROR] Unsupported file type for JSONCollector: {file_extension}")
            return []

        if is_url:
            return self.read_json_from_url(file_path_or_url)
        else:
            return self.read_json_file(file_path_or_url)

    def load(self, sources):
        docs = []

        print("[DEBUG] Argument received:", sources)
        if isinstance(sources, str):
            print("[DEBUG] Absolute path:", os.path.abspath(sources))
            print("[DEBUG] Is dir?", os.path.isdir(sources))

        file_list = []

        if isinstance(sources, str) and os.path.isdir(sources):
            file_list = self._get_json_files_in_directory(sources)
            if not file_list:
                print(f"[VALIDATION] No valid JSON files to process in folder '{sources}'.")
                return []

        elif isinstance(sources, str) and sources.lower().endswith('.json') and not os.path.isdir(sources):
            file_list = [sources]

        elif isinstance(sources, list):
            seen = set()
            file_list = []
            for item in sources:
                if isinstance(item, str) and (item.lower().endswith(".json") or item.lower().endswith(".JSON") or item.startswith("http")):
                    if item not in seen:
                        file_list.append(item)
                        seen.add(item)
            if not file_list:
                print("[ERROR] No valid JSON files or URLs in the list to process.")
                return []

        else:
            print("[ERROR] 'sources' must be a directory path, a JSON file path, or a list of JSON paths/URLs.")
            return []

        if not file_list:
            print("[VALIDATION] No JSON files found. Aborting JSON extraction.")
            return []

        print(f"[INFO] JSON files to process: {file_list}")
        with open(self.backup_path, "w", encoding="utf-8") as backup_file:
            for src in file_list:
                entries = self.fetch_json_content(src)
                for text, meta in entries:
                    if text:
                        doc_id = str(uuid.uuid4())
                        metadata = {"source": src, "file_extension": ".json"}
                        metadata.update(meta)
                        backup_entry = {
                            "id": doc_id,
                            "page_content": text,
                            "metadata": metadata
                        }
                        json.dump(backup_entry, backup_file, ensure_ascii=False)
                        backup_file.write("\n")
                        docs.append(Document(page_content=text, metadata=metadata))

        print(f"[INFO] Loaded and backed up {len(docs)} JSON documents")
        return docs
