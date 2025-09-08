import json
import uuid
from typing import List
from langchain_core.documents import Document


from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    SentenceTransformersTokenTextSplitter,
    CharacterTextSplitter
)

class Chunker:


    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        token_chunk_size: int = 256,
        token_chunk_overlap: int = 32,
        sentence_token_chunk_size: int = 256,
        sentence_token_overlap: int = 32,
        word_chunk_size: int = 100,
        word_chunk_overlap: int = 10,
        backup_path: str = "chunking/chunked_docs_backup.jsonl"
    ):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.token_chunk_size = token_chunk_size
        self.token_chunk_overlap = token_chunk_overlap
        self.sentence_token_chunk_size = sentence_token_chunk_size
        self.sentence_token_overlap = sentence_token_overlap
        self.word_chunk_size = word_chunk_size
        self.word_chunk_overlap = word_chunk_overlap
        self.backup_path = backup_path

    def context_split(self, documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_documents(documents)

    def token_split(self, documents: List[Document]) -> List[Document]:

        splitter = TokenTextSplitter(
            encoding_name="cl100k_base",
            chunk_size=self.token_chunk_size,
            chunk_overlap=self.token_chunk_overlap
        )
        return splitter.split_documents(documents)

    def sentence_split(self, documents: List[Document]) -> List[Document]:
        splitter = SentenceTransformersTokenTextSplitter(
            # Default model: 'sentence-transformers/all-mpnet-base-v2'
            tokens_per_chunk=self.sentence_token_chunk_size,
            chunk_overlap=self.sentence_token_overlap
        )
        return splitter.split_documents(documents)

    def word_split(self, documents: List[Document]) -> List[Document]:
        splitter = CharacterTextSplitter(
            separator=" ",
            chunk_size=self.word_chunk_size,
            chunk_overlap=self.word_chunk_overlap
        )
        return splitter.split_documents(documents)

    def chunk_documents(self, documents: List[Document]) -> List[Document]:

        for chunk_fn in [
            self.context_split,
            self.token_split,
            self.sentence_split,
            self.word_split
        ]:
            chunks = chunk_fn(documents)
            print(f"[CHUNKER] Used {chunk_fn.__name__} splitting, produced {len(chunks)} chunks.")
            if len(chunks) > 300:
                print(f"[CHUNKER] Used {chunk_fn.__name__} splitting, produced {len(chunks)} chunks.")
                self.backup_jsonl(chunks)
                return chunks
            else:
                print(f"[CHUNKER] {chunk_fn.__name__} did not produce enough chunks, trying next...")

        print(f"[CHUNKER] Falling back to final chunking method with {len(chunks)} chunks.")
        self.backup_jsonl(chunks)
        return chunks

    def backup_jsonl(self, chunked_docs: List[Document]):

        with open(self.backup_path, "w", encoding="utf-8") as f:
            for doc in chunked_docs:
                entry = {
                    "id": str(uuid.uuid4()),
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")
        print(f"[CHUNKER] Backup of {len(chunked_docs)} chunks saved to {self.backup_path}")

