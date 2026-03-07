import os
from typing import List

import requests
import yaml


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

        prompt_path = os.path.join(os.path.dirname(__file__), 'prompts_input.yml')
        with open(prompt_path, 'r') as f:
            prompt_config = yaml.load(f, Loader=yaml.SafeLoader)

        prompt_template = prompt_config['introductions']['lectures']

        prompt = prompt_template.format(
            formatted_context=formatted_context,
            user_query=user_query
        )

        return prompt

    def chat(
        self,
        user_query: str,
        context_chunks: List[str]
    ) -> str:
        """
        Main method to handle the chat interaction. It formats the prompt and sends it to the LLM.
        :param user_query: user's query string
        :param context_chunks: chunks of text retrieved from the vector store that are relevant to the user's query
        :return: response from the LLM
        """
        prompt = self.format_prompt(
            user_query=user_query,
            context_chunks=context_chunks
        )

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()  # Checks if Ollama is actually running
            return response.json().get("response", "Error: Empty response")
        except requests.exceptions.ConnectionError:
            return "Error: Is Ollama running? Please start the Ollama app."
