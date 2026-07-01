# base_agent.py - Shared base class for all agents
from llm.gemini_client import GeminiClient

class BaseAgent:
    def __init__(self, system_prompt: str):
        self.client = GeminiClient()
        self.system_prompt = system_prompt

    def run(self, user_message: str, context: str = "") -> str:
        full_prompt = f"{self.system_prompt}\n\nContext:\n{context}\n\nUser: {user_message}"
        return self.client.generate(full_prompt)
