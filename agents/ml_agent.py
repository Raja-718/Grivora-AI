# ml_agent.py - PMA-aware ML agent with Gemini LLM intelligence
from agents.base_agent import BaseAgent
from llm.prompt_templates import ML_SYSTEM_PROMPT

PMA_SYSTEM_PROMPT = """You are an expert machine learning engineer and data scientist working inside Grivora AI.

You help users through the Predictive Modeling & Analysis (PMA) workflow.

Your capabilities:
- Explain model selection rationale for the user's specific dataset
- Interpret model metrics (accuracy, RMSE, R², F1, AUC) in plain English
- Suggest feature engineering ideas based on column names and data types
- Explain what confusion matrices, feature importance charts mean
- Recommend hyperparameter tuning directions
- Generate insights from prediction results

Rules:
- Always refer to ACTUAL column names from the dataset context
- Explain technical concepts in simple, business-friendly language
- Be specific — avoid generic advice
- Keep answers concise and actionable (under 250 words)
- If the user asks about model metrics, explain what they mean AND if they are good or bad
"""

class MLAgent(BaseAgent):
    def __init__(self):
        super().__init__(PMA_SYSTEM_PROMPT)

    def explain_model_selection(self, data_info: dict, recommendations: list) -> str:
        context = f"""
Dataset: {data_info.get('n_rows')} rows × {data_info.get('n_cols')} columns
Data type: {data_info.get('data_type')}
Problem type: {data_info.get('problem_type')}
Target column: {data_info.get('target_col')}
Numeric columns: {data_info.get('num_cols')}
Categorical columns: {data_info.get('cat_cols')}
Recommended models: {[r['key'] for r in recommendations]}
"""
        message = "Explain why these models are recommended for this dataset and what the user should expect."
        return self.run(message, context)

    def explain_metrics(self, metrics: dict, problem_type: str, model_name: str) -> str:
        context = f"""
Model: {model_name}
Problem type: {problem_type}
Metrics: {metrics}
"""
        message = "Explain these evaluation metrics in plain English. Tell the user if these results are good, average, or need improvement. Be specific."
        return self.run(message, context)

    def suggest_improvements(self, metrics: dict, model_key: str, data_info: dict) -> str:
        context = f"""
Model used: {model_key}
Metrics: {metrics}
Dataset info: {data_info}
"""
        message = "Based on these metrics, suggest 3 specific improvements the user can make: feature engineering, data cleaning, or hyperparameter tuning. Be concrete and actionable."
        return self.run(message, context)
