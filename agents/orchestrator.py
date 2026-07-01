# orchestrator.py - Routes user tasks to the correct agent
from agents.analyst_agent import AnalystAgent
from agents.chart_agent import ChartAgent
from agents.ml_agent import MLAgent
from agents.cleaning_agent import CleaningAgent
from llm.gemini_client import GeminiClient
from src.data_loader import load_file

class Orchestrator:
    def __init__(self):
        self.client = GeminiClient()
        self.agents = {
            "analyst": AnalystAgent(),
            "chart": ChartAgent(),
            "ml": MLAgent(),
            "cleaning": CleaningAgent(),
        }

    def decide_agent(self, message: str) -> str:
        prompt = f"""You are a routing assistant. Based on the user message below, decide which agent to use.
Choose one of: analyst, chart, ml, cleaning.
- Use 'analyst' for questions about data, columns, statistics, summaries
- Use 'chart' for visualization, charts, graphs, plots
- Use 'ml' for predictions, machine learning, models
- Use 'cleaning' for data quality, missing values, duplicates, fixing data
Only reply with the agent name, nothing else.

User message: {message}"""
        return self.client.generate(prompt).strip().lower()

    def get_file_context(self, file_path: str) -> str:
        """Load the file and return a rich context string for the agent."""
        try:
            df = load_file(file_path)
            context = f"""
DATASET INFORMATION:
- Shape: {df.shape[0]} rows x {df.shape[1]} columns
- Columns: {list(df.columns)}
- Data Types:
{df.dtypes.to_string()}

- Missing Values:
{df.isnull().sum().to_string()}

- First 10 rows:
{df.head(10).to_string()}

- Statistical Summary:
{df.describe(include='all').to_string()}
"""
            return context
        except Exception as e:
            return f"Error loading file: {str(e)}"

    def run(self, message: str, file_path: str = None) -> str:
        # Decide which agent to use
        agent_name = self.decide_agent(message)
        # Clean up agent name (sometimes Gemini adds extra words)
        for name in ["analyst", "chart", "ml", "cleaning"]:
            if name in agent_name:
                agent_name = name
                break
        else:
            agent_name = "analyst"

        agent = self.agents[agent_name]

        # Build context from actual file
        context = ""
        if file_path:
            context = self.get_file_context(file_path)

        return agent.run(message, context)
