import json
import os
from typing import List, Any, Generator

import requests
import yaml
from networkx.classes import common_neighbors


class LLMDiscuss:
    def __init__(
        self,
        model_name = 'gemma3:4b'
    ):
        self.model_name = model_name

        self.ollama_url = 'http://localhost:11434/api/generate'

    @staticmethod
    def format_prompt(
        user_query: str,
        context_chunks: List[str]
    ) -> str:
        """
        Formats the prompt for the LLM by combining the user query with relevant context chunks.
        This is a simple concatenation, but can be enhanced with templates or more sophisticated formatting.
        :param user_query: user's query string
        :param context_chunks: chunks of text retrieved from the vector store that are relevant to the user's query
        :return: prompt string
        """
        formatted_context = "\n\n".join(context_chunks)

        prompt_path = os.path.join(os.path.dirname(__file__), 'prompts', 'prompts_inputs.yaml')
        with open(prompt_path, 'r') as f:
            prompt_config = yaml.load(f, Loader=yaml.SafeLoader)

        prompt_template = prompt_config['introductions']['lectures']

        prompt = prompt_template.format(
            formatted_context=formatted_context,
            user_query=user_query
        )

        return prompt

    def chat_stream(
        self,
        user_query: str,
        context_chunks: List[str] = None
    ) -> Generator[Any, Any, None]:
        """
        A generator that yields chunks of text from Ollama
        """
        # TODO: caching feature CAG (Cache Augmented Generation), only good when docs don't change much
        # TODO: experiment with Agentic RAG, slower and costly but higher quality
        # TODO: multi-modal RAG e.g. images etc
        if context_chunks:
            prompt = self.format_prompt(
                user_query=user_query,
                context_chunks=context_chunks
            )
        else:
            prompt = self.format_prompt(user_query=user_query, context_chunks=[])

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True
        }

        try:
            response = requests.post(self.ollama_url, json=payload, stream=True)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    # Ollama sends back multiple JSON objects, one for each word
                    chunk = json.loads(line)
                    content = chunk.get("response", "")
                    yield content

                    if chunk.get("done"):
                        break

        except requests.exceptions.ConnectionError:
            yield "Error: Is Ollama running? Please start the Ollama app."