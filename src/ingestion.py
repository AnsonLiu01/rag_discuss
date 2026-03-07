from typing import List

import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter


class IngestionEngine:
    """
    Handles PDF extraction and preparation for the Vector DB
    """

    def __init__(self, chunk_size=1000, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    @staticmethod
    def extract_text_from_pdf(
        pdf_file
    ) -> str:
        """
        Reads text from an uploaded PDF file object
        """
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def create_chunks(
        self,
        text: str,
        metadata=None
    ) -> List[str]:
        """
        Splits text into manageable chunks for embedding
        """
        return self.text_splitter.split_text(text)

    def process_all_files(self, uploaded_files):
        """Orchestrates extraction and chunking for multiple files."""
        all_chunks = []
        for file in uploaded_files:
            raw_text = self.extract_text_from_pdf(file)
            chunks = self.create_chunks(raw_text)
            all_chunks.extend(chunks)
        return all_chunks