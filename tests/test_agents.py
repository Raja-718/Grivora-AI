# test_agents.py - Test each AI agent response
import pytest
from unittest.mock import patch, MagicMock

@patch("llm.gemini_client.GeminiClient.generate")
def test_analyst_agent(mock_generate):
    mock_generate.return_value = "The average sales is 500."
    from agents.analyst_agent import AnalystAgent
    agent = AnalystAgent()
    result = agent.run("What is the average sales?", "sales,region\n500,North\n300,South")
    assert isinstance(result, str)
    assert len(result) > 0

@patch("llm.gemini_client.GeminiClient.generate")
def test_cleaning_agent(mock_generate):
    mock_generate.return_value = "Column 'age' has 3 missing values. Use df['age'].fillna(df['age'].median())"
    from agents.cleaning_agent import CleaningAgent
    agent = CleaningAgent()
    result = agent.run("Find data issues", "name,age\nAli,25\nSara,\nJohn,30")
    assert isinstance(result, str)

@patch("llm.gemini_client.GeminiClient.generate")
def test_chart_agent(mock_generate):
    mock_generate.return_value = "Use a bar chart to compare sales by region."
    from agents.chart_agent import ChartAgent
    agent = ChartAgent()
    result = agent.run("What chart should I use?", "sales,region\n500,North")
    assert isinstance(result, str)

@patch("llm.gemini_client.GeminiClient.generate")
def test_ml_agent(mock_generate):
    mock_generate.return_value = "Use Random Forest. Accuracy: 92%."
    from agents.ml_agent import MLAgent
    agent = MLAgent()
    result = agent.run("Which model should I use to predict churn?", "")
    assert isinstance(result, str)
