# import re
# from typing import List
# from langchain_core.documents import Document


# class TextCleaner:


#     def __init__(self):
#         # Add any initialization parameters or resources if needed
#         pass

#     def clean_text(self, text: str) -> str:

#         # Lowercase text
#         text = text.lower()

#         # Remove non-printable/control characters
#         text = ''.join(ch for ch in text if ch.isprintable())

#         # Remove unwanted special characters but keep basic punctuation .,!?
#         text = re.sub(r"[^a-z0-9\s\.\,\!\?']", " ", text)

#         # Normalize whitespace (collapse multiple spaces and newlines)
#         text = re.sub(r'\s+', ' ', text).strip()

#         return text

#     def clean_documents(self, documents: List[Document]) -> List[Document]:

#         cleaned_docs = []
#         for doc in documents:
#             cleaned_text = self.clean_text(doc.page_content)
#             cleaned_docs.append(Document(page_content=cleaned_text, metadata=doc.metadata))
#         return cleaned_docs



import re
import unicodedata
from typing import List, Optional
from langchain_core.documents import Document


class TextCleaner:


    def __init__(self, 
                 preserve_structure: bool = True,
                 min_sentence_length: int = 10,
                 max_sentence_length: int = 1000,
                 remove_urls: bool = True,
                 remove_emails: bool = True,
                 normalize_unicode: bool = True):

        self.preserve_structure = preserve_structure
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_unicode = normalize_unicode
        
        # Compile regex patterns for better performance
        self._compile_patterns()

    def _compile_patterns(self):

        # URL pattern (matches http, https, ftp, www)
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|'
            r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            re.IGNORECASE
        )
        
        # Email pattern
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        
        # Multiple whitespace pattern
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Non-alphanumeric pattern (keeping essential punctuation)
        self.special_chars_pattern = re.compile(r"[^a-zA-Z0-9\s\.\,\!\?\;\:\'\"\-\(\)\[\]]")
        
        # Pattern for excessive punctuation
        self.excessive_punct_pattern = re.compile(r'([.!?]){3,}')
        
        # Pattern for bullet points and list markers
        self.bullet_pattern = re.compile(r'^[\s]*[•·▪▫◦‣⁃\*\-\+]\s*', re.MULTILINE)
        
        # Pattern for numbered lists
        self.numbered_list_pattern = re.compile(r'^[\s]*\d+[\.\)]\s*', re.MULTILINE)
        
        # Pattern for headers/titles (lines with all caps or title case)
        self.header_pattern = re.compile(r'^[A-Z\s]{3,}$', re.MULTILINE)

    def normalize_unicode_text(self, text: str) -> str:

        # Normalize unicode characters to their closest ASCII equivalents.

        if not self.normalize_unicode:
            return text
            
        # Normalize unicode to NFD (decomposed form) then filter out combining characters
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        
        # Additional common unicode replacements
        unicode_replacements = {
            '"': '"', '"': '"',  # Smart quotes
            ''': "'", ''': "'",  # Smart apostrophes
            '–': '-', '—': '-',  # En dash, em dash
            '…': '...',          # Ellipsis
            '«': '"', '»': '"',  # Guillemets
            '°': ' degrees ',    # Degree symbol
            '©': '(c)',          # Copyright
            '®': '(r)',          # Registered trademark
            '™': '(tm)',         # Trademark
        }
        
        for unicode_char, replacement in unicode_replacements.items():
            text = text.replace(unicode_char, replacement)
            
        return text

    def remove_unwanted_patterns(self, text: str) -> str:

        # Remove URLs, emails, and other unwanted patterns.

        if self.remove_urls:
            text = self.url_pattern.sub(' ', text)
            
        if self.remove_emails:
            text = self.email_pattern.sub(' ', text)
            
        # Remove excessive punctuation (3 or more consecutive)
        text = self.excessive_punct_pattern.sub(r'\1', text)
        
        return text

    def clean_structural_elements(self, text: str) -> str:

        # Clean structural elements like bullet points, headers, etc.

        # Convert bullet points to regular text
        text = self.bullet_pattern.sub('', text)
        
        # Convert numbered lists to regular text
        text = self.numbered_list_pattern.sub('', text)
        
        # Clean up headers - convert all caps to title case
        def header_replacer(match):
            header_text = match.group().strip()
            if len(header_text) > 0:
                return header_text.title() + '. '
            return ''
        
        text = self.header_pattern.sub(header_replacer, text)
        
        return text

    def normalize_spacing(self, text: str) -> str:

        # Normalize spacing while preserving structure if needed.

        if self.preserve_structure:
            # Preserve paragraph breaks (double newlines)
            text = re.sub(r'\n{3,}', '\n\n', text)  # Limit to double newlines max
            text = re.sub(r'[ \t]+', ' ', text)      # Normalize horizontal whitespace
            text = re.sub(r'\n ', '\n', text)        # Remove space after newlines
            text = re.sub(r' \n', '\n', text)        # Remove space before newlines
        else:
            # Collapse all whitespace
            text = self.whitespace_pattern.sub(' ', text)
            
        return text.strip()

    def filter_sentences(self, text: str) -> str:

        # Filter sentences based on length criteria.

        if not self.preserve_structure:
            return text
            
        sentences = re.split(r'(?<=[.!?])\s+', text)
        filtered_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Skip very short sentences
            if len(sentence) < self.min_sentence_length:
                continue
                
            # Split very long sentences
            if len(sentence) > self.max_sentence_length:
                # Try to split on common conjunctions
                sub_sentences = re.split(r'\s+(?:and|but|or|however|moreover|furthermore|therefore|thus|consequently)\s+', sentence)
                for sub_sentence in sub_sentences:
                    sub_sentence = sub_sentence.strip()
                    if len(sub_sentence) >= self.min_sentence_length:
                        filtered_sentences.append(sub_sentence)
            else:
                filtered_sentences.append(sentence)
                
        return ' '.join(filtered_sentences)

    def clean_text(self, text: str) -> str:

        if not text or not isinstance(text, str):
            return ""
        
        # Step 1: Normalize unicode
        text = self.normalize_unicode_text(text)
        
        # Step 2: Remove control characters but keep printable ones
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
        
        # Step 3: Remove unwanted patterns (URLs, emails, etc.)
        text = self.remove_unwanted_patterns(text)
        
        # Step 4: Clean structural elements
        text = self.clean_structural_elements(text)
        
        # Step 5: Lowercase conversion
        text = text.lower()
        
        # Step 6: Remove unwanted special characters
        text = self.special_chars_pattern.sub(' ', text)
        
        # Step 7: Normalize spacing
        text = self.normalize_spacing(text)
        
        # Step 8: Filter sentences by length
        text = self.filter_sentences(text)
        
        # Step 9: Final cleanup
        text = text.strip()
        
        # Ensure we don't return empty strings
        if not text:
            return ""
            
        return text

    def clean_documents(self, documents: List[Document]) -> List[Document]:

        if not documents:
            return []
            
        cleaned_docs = []
        
        for i, doc in enumerate(documents):
            try:
                if not isinstance(doc, Document):
                    print(f"Warning: Item {i} is not a Document object, skipping...")
                    continue
                    
                cleaned_text = self.clean_text(doc.page_content)
                
                # Only keep documents with meaningful content
                if cleaned_text and len(cleaned_text.strip()) >= self.min_sentence_length:
                    # Preserve original metadata and add cleaning info
                    new_metadata = doc.metadata.copy() if doc.metadata else {}
                    new_metadata['cleaned'] = True
                    new_metadata['original_length'] = len(doc.page_content)
                    new_metadata['cleaned_length'] = len(cleaned_text)
                    
                    cleaned_docs.append(Document(
                        page_content=cleaned_text,
                        metadata=new_metadata
                    ))
                else:
                    print(f"Warning: Document {i} was too short after cleaning, skipping...")
                    
            except Exception as e:
                print(f"Error cleaning document {i}: {str(e)}")
                continue
                
        return cleaned_docs

    def get_cleaning_stats(self, original_docs: List[Document], cleaned_docs: List[Document]) -> dict:

        original_count = len(original_docs)
        cleaned_count = len(cleaned_docs)
        
        original_total_chars = sum(len(doc.page_content) for doc in original_docs)
        cleaned_total_chars = sum(len(doc.page_content) for doc in cleaned_docs)
        
        return {
            'original_document_count': original_count,
            'cleaned_document_count': cleaned_count,
            'documents_removed': original_count - cleaned_count,
            'removal_rate': (original_count - cleaned_count) / original_count if original_count > 0 else 0,
            'original_total_characters': original_total_chars,
            'cleaned_total_characters': cleaned_total_chars,
            'character_reduction_rate': (original_total_chars - cleaned_total_chars) / original_total_chars if original_total_chars > 0 else 0
        }