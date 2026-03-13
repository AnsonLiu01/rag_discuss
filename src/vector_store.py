from typing import List
import uuid

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
        # self.client = chromadb.PersistentClient(path=db_path)
        self.client = chromadb.EphemeralClient()

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
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]

        # Generate embeddings
        embeddings = self.model.encode(chunks).tolist()

        # CRITICAL: We must include metadatas=metadata_list here
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

    def get_all_sources(self):
        """
        Refreshes and returns unique filenames currently in the brain.
        """
        # We force a 'peek' at the data
        results = self.collection.get(include=['metadatas'])

        if not results or not results['metadatas']:
            return []

        # Extract 'source' from each metadata dict and turn into a unique set
        sources = set()
        for meta in results['metadatas']:
            if meta and 'source' in meta:
                sources.add(meta['source'])

        return sorted(list(sources))

    def delete_all(self):
        """
        Wipes the entire collection
        """
        ids = self.collection.get()['ids']
        if ids:
            self.collection.delete(ids=ids)
            return True
        return False

    def delete_by_source(self, source_name):
        """
        Deletes all chunks associated with a specific filename
        """
        self.collection.delete(where={"source": source_name})
        return True