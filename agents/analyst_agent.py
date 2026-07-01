# analyst_agent.py - Answers questions about uploaded data
from agents.base_agent import BaseAgent
from llm.prompt_templates import ANALYST_SYSTEM_PROMPT

class AnalystAgent(BaseAgent):
    def __init__(self):
        super().__init__(ANALYST_SYSTEM_PROMPT)
