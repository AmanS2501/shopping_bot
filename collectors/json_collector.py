from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import requests
import json
import uuid
from langchain_core.documents import Document

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

class JSONCollector:


    def __init__(self, backup_path: str = "collectors/json_extracted_backup.jsonl"):
        self.backup_path = backup_path

    def read_json_file(self, file_path: str) -> list:

        try:
            print(f"[INFO] Reading JSON: {file_path}")
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            return self._extract_texts(data)
        except Exception as e:
            print(f"[ERROR] Failed to read JSON {file_path}: {e}")
            return []

    def read_json_from_url(self, url: str) -> list:

        try:
            print(f"[INFO] Downloading JSON from URL: {url}")
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            data = response.json()
            return self._extract_texts(data)
        except Exception as e:
            print(f"[ERROR] Failed to download/read JSON from URL {url}: {e}")
            return []

    def fetch_json_content(self, file_path_or_url: str) -> list:

        from urllib.parse import urlparse
        is_url = file_path_or_url.startswith("http://") or file_path_or_url.startswith("https://")
        file_extension = Path(urlparse(file_path_or_url).path).suffix.lower()

        if file_extension != '.json':
            print(f"[ERROR] Unsupported file type for JSONCollector: {file_extension}")
            return []

        if is_url:
            return self.read_json_from_url(file_path_or_url)
        else:
            return self.read_json_file(file_path_or_url)

    def _extract_texts(self, data) -> list:

        texts = []
        if isinstance(data, dict):
            # If flat dict, collect all string values
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

    def load(self, source_list):

        docs = []
        with open(self.backup_path, "w", encoding="utf-8") as backup_file:
            for src in source_list:
                entries = self.fetch_json_content(src)
                for (text, submeta) in entries:
                    if text:
                        doc_id = str(uuid.uuid4())
                        metadata = {
                            "source": src,
                            "file_extension": ".json",
                        }
                        metadata.update(submeta)
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
