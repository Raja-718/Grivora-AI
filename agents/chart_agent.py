# chart_agent.py - Auto-generates chart suggestions and code
from agents.base_agent import BaseAgent
from llm.prompt_templates import CHART_SYSTEM_PROMPT

class ChartAgent(BaseAgent):
    def __init__(self):
        super().__init__(CHART_SYSTEM_PROMPT)
