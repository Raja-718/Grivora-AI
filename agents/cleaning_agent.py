# cleaning_agent.py - Suggests data cleaning steps
from agents.base_agent import BaseAgent
from llm.prompt_templates import CLEANING_SYSTEM_PROMPT

class CleaningAgent(BaseAgent):
    def __init__(self):
        super().__init__(CLEANING_SYSTEM_PROMPT)
