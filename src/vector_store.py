from typing import List

import chromadb
from sentence_transformers import SentenceTransformer


class VectorStore:
    """
    Handles local storage and semantic search of (lecture) notes
    """
    def __init__(
        self,
        db_path="./data/vector_store"
    ):
        self.client = chromadb.PersistentClient(path=db_path)

        self.model = SentenceTransformer('BAAI/bge-small-en-v1.5')

        self.collection = self.client.get_or_create_collection(name="temp_notes")

    def add_documents(
        self,
        chunks: List[str],
        metadata_list=None
    ):
        """
        Converts text chunks to vectors and stores them
        """
        # TODO: store as different collection per lecture and slide
        # TODO: how to chunk the docs - character/word limit, token limit, paragraph chunking, semantic chunking
        #  (shift in meanings, measure similarity between all sentences), preserving natural semnatic breaks
        #  overlapping chunking/sliding window to preserve context across chunks from previous row. Could use agentic
        #  chunking i.e. LangChain (RecursiveCharacterTextSplitter), or SpaCy.
        ids = [f"id_{i}" for i in range(len(chunks))]

        embeddings = self.model.encode(chunks).tolist()

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadata_list if metadata_list else None
        )

    def search(
        self,
        query: str,
        n_results=3
    ) -> List[str]:
        """
        Searches for the most relevant lecture note segments
        """
        query_vector = self.model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_vector,
            n_results=n_results
        )
        return results['documents'][0]