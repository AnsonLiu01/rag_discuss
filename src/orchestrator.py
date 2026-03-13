from typing import Any, Generator

from src.vector_store import VectorStore
from src.llm_discuss import LLMDiscuss



class TutorOrchestrator:
    def __init__(self):
        """
        Initialises the orchestrator with our existing brain and voice.
        """
        self.vector_db = VectorStore()
        self.llm = LLMDiscuss()

    def ask(self, user_query: str):
        """
        Retrieves context and returns both the LLM stream and the sources.
        """
        context_chunks = self.vector_db.search(query=user_query, n_results=3)

        if not context_chunks:
            # If no notes are found, we return a "dummy" generator and empty sources
            def empty_gen():
                yield "I'm sorry, I couldn't find any relevant information in your notes."

            return empty_gen(), []

        stream_generator = self.llm.chat_stream(
            user_query=user_query,
            context_chunks=context_chunks
        )

        return stream_generator, context_chunks