# Grivora AI

A Flask-based web application for data analysts and data scientists, powered by Gemini 2.5 Flash AI agents.

## Features
- Upload CSV / Excel files
- AI-powered data analysis (ask questions in plain English)
- Auto-generate charts and reports
- ML model training and predictions
- Data cleaning suggestions

## Setup

### 1. Clone & install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your Gemini API key
Edit `.env` and add your key:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Run the app
```bash
python run.py
```

Visit: http://localhost:5000

## Project Structure
```
my-data-project/
├── app/            → Flask web layer (routes, templates, static)
├── agents/         → AI agents (orchestrator + 4 specialists)
├── llm/            → Gemini client & prompt templates
├── src/            → Core data & ML modules
├── data/           → Raw, processed, sample data
├── notebooks/      → Jupyter EDA & experiments
├── models/         → Saved ML models
├── uploads/        → User uploaded files
├── tests/          → Unit tests
├── config.py       → App configuration
└── run.py          → Entry point
```

## AI Agents
| Agent | Role |
|---|---|
| Orchestrator | Routes user intent to the right agent |
| Analyst Agent | Answers questions about your data |
| Chart Agent | Generates charts and reports |
| ML Agent | Runs predictions and explains results |
| Cleaning Agent | Suggests data quality fixes |
