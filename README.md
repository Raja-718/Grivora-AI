<p align="center">
  <img src="logo/robot image.jpeg" alt="Grivora AI Logo" width="140" />
</p>

<h1 align="center">Grivora AI</h1>

<p align="center">
  <strong>AI-Powered Data Analytics & Machine Learning Platform</strong><br/>
  <em>Upload data. Ask questions in plain English. Get instant insights, charts, and predictions.</em>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%20|%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://flask.palletsprojects.com/"><img src="https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"></a>
  <a href="https://ai.google.dev/"><img src="https://img.shields.io/badge/Gemini_2.5_Flash-AI-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Gemini AI"></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn"></a>
  <a href="https://github.com/Raja-718/Grivora-AI/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"></a>
</p>

---

## 🚀 What is Grivora AI?

**Grivora AI** is a full-stack, AI-powered data analytics platform that lets data analysts and data scientists go from raw data to actionable insights — without writing a single line of code.

Upload your CSV, Excel, or Parquet files and interact with your data through natural language. Grivora AI's multi-agent system — powered by **Google Gemini 2.5 Flash** — automatically understands your intent and routes it to the right specialist agent.

> 💡 *"What are the top 5 cities by revenue?"* → The Analyst Agent answers.
>
> 📊 *"Show me a bar chart of monthly sales"* → The Chart Agent builds it.
>
> 🤖 *"Predict next quarter's revenue"* → The ML Agent trains & predicts.
>
> 🧹 *"Are there any data quality issues?"* → The Cleaning Agent flags them.

---

## ✨ Key Features

### 🧠 AI Multi-Agent System
| Agent | Responsibility |
|:------|:---------------|
| **🎯 Orchestrator** | Analyzes user intent and routes to the correct specialist agent |
| **📊 Analyst Agent** | Answers analytical questions about your data in plain English |
| **📈 Chart Agent** | Generates 25+ chart types (bar, line, scatter, heatmap, radar, etc.) |
| **🤖 ML Agent** | Trains models, runs predictions, and explains results |
| **🧹 Cleaning Agent** | Detects data quality issues and suggests fixes |

### 📂 Universal Data Loader
- **Tabular**: CSV, TSV, TXT, DAT
- **Spreadsheets**: XLSX, XLS, XLSB, XLSM, ODS
- **Modern formats**: Parquet, JSON, XML
- Auto-detects encoding, delimiters, and data types
- Handles files up to **16 MB**

### 🤖 Predictive Modeling Automation (PMA Engine)
- **40+ ML algorithms** across 7 task types:
  - Regression, Classification, Time Series, Clustering, Anomaly Detection, NLP, Dimensionality Reduction
- **Production-grade pipeline**: No data leakage — encoders/scalers fit on training data only
- **AutoML**: Trains top-N suggested models in parallel, ranks them on a leaderboard
- **LLM-first algorithm suggestion**: Gemini profiles your dataset (30+ signals) and recommends the best algorithms
- **Hyperparameter tuning** via GridSearchCV / RandomizedSearchCV
- **Permutation importance** for honest feature ranking
- Supports **XGBoost**, **LightGBM**, **CatBoost**, **Prophet**, **ARIMA**, and more

### 📊 Business Intelligence & Analytics (BIA Engine)
- Auto-generates interactive dashboards from your data profile
- LLM-planned chart selection — Gemini decides which charts tell the best story
- Rule-based fallback when the LLM is unavailable
- **MySQL connector** for live database queries
- Chunk processing for large datasets (10K+ rows)
- State management and query caching

### 🔐 Authentication & Security
- Email + Mobile OTP login (Gmail SMTP, SendGrid, Mailgun, Twilio)
- **Google OAuth** (Sign in with Google)
- Session-based auth with cookie hardening (`HttpOnly`, `SameSite=Lax`)
- CSRF protection (Flask-WTF)
- Rate limiting on LLM endpoints (Flask-Limiter)

### 📈 25+ Chart Types
Bar, Stacked Bar, Line, Multi-Line, Area, Scatter, Bubble, Pie, Doughnut, Radar, Polar Area, Heatmap, Histogram, Box Plot, Violin, Time Series, Candlestick, Waterfall, Funnel, Treemap, Gauge, Sankey, and more.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FLASK WEB LAYER                        │
│              (Routes, Templates, Static Assets)             │
├──────────┬──────────┬───────────┬───────────┬───────────────┤
│  Upload  │   Chat   │ Dashboard │  Predict  │  BI Analytics │
└────┬─────┴────┬─────┴─────┬─────┴─────┬─────┴───────┬───────┘
     │          │           │           │             │
     ▼          ▼           ▼           ▼             ▼
┌─────────────────────────────────────────────────────────────┐
│                    AI AGENT LAYER                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              🎯 Orchestrator Agent                    │   │
│  │         (Intent Detection + Smart Routing)            │   │
│  └──────┬──────────┬──────────┬──────────┬──────────────┘   │
│         │          │          │          │                   │
│    ┌────▼───┐ ┌────▼───┐ ┌───▼────┐ ┌───▼──────┐           │
│    │Analyst │ │ Chart  │ │   ML   │ │ Cleaning │           │
│    │ Agent  │ │ Agent  │ │ Agent  │ │  Agent   │           │
│    └────┬───┘ └────┬───┘ └───┬────┘ └───┬──────┘           │
└─────────┼──────────┼─────────┼──────────┼───────────────────┘
          │          │         │          │
          ▼          ▼         ▼          ▼
┌─────────────────────────────────────────────────────────────┐
│                     CORE ENGINES                             │
│                                                              │
│  📊 Chart Library    🤖 PMA Engine     📈 BIA Engine         │
│  (25+ chart types)  (40+ algorithms)  (BI dashboards)       │
│                                                              │
│  🔍 ML Suggester     ⚡ AutoML         🎨 Dashboard Planner  │
│  (LLM-first picks)  (parallel train)  (LLM chart planning)  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  🧠 Gemini 2.5 Flash  │  📁 Data Loader  │  💾 Model Store  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technologies |
|:------|:-------------|
| **Backend** | Python 3.10+, Flask 3.0, Gunicorn |
| **AI/LLM** | Google Gemini 2.5 Flash |
| **ML** | scikit-learn, XGBoost, LightGBM, CatBoost, Prophet, ARIMA |
| **Data** | Pandas, NumPy, PyArrow, OpenPyXL |
| **Visualization** | Chart.js (frontend), Matplotlib, Plotly |
| **Auth** | OTP (Email/SMS), Google OAuth, Flask-WTF CSRF |
| **Database** | SQLite (auth), MySQL (optional BIA) |
| **Security** | Flask-Limiter, CSRF protection, session hardening |

---

## ⚡ Quick Start

### Prerequisites

- **Python 3.10** or **3.11**
- **pip** (Python package manager)
- A **Google Gemini API key** ([Get one free](https://aistudio.google.com/apikey))

### 1. Clone the Repository

```bash
git clone https://github.com/Raja-718/Grivora-AI.git
cd Grivora-AI
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your Gemini API key
# (required) GEMINI_API_KEY=your_gemini_api_key_here
# (required) SECRET_KEY=your_secret_key_here
```

### 5. Run the Application

```bash
python run.py
```

🎉 **Visit**: [http://localhost:5000](http://localhost:5000)

---

## 📁 Project Structure

```
Grivora-AI/
│
├── app/                        # Flask web layer
│   ├── __init__.py             # App factory (limiter, CSRF, blueprints)
│   ├── routes.py               # All API & page routes
│   ├── auto_dashboard_route.py # Auto-dashboard generation endpoint
│   ├── auth.py                 # Auth helpers
│   ├── static/                 # CSS, JavaScript (Chart.js integration)
│   └── templates/              # Jinja2 HTML templates
│       ├── index.html          # Landing page
│       ├── dashboard.html      # Main dashboard
│       ├── chat.html           # AI chat interface
│       ├── upload.html         # File upload
│       ├── predict.html        # Prediction interface
│       ├── bi.html             # Business intelligence view
│       └── auth/               # Login & registration pages
│
├── agents/                     # AI multi-agent system
│   ├── orchestrator.py         # Intent routing (Gemini-powered)
│   ├── analyst_agent.py        # Data analysis specialist
│   ├── chart_agent.py          # Visualization specialist
│   ├── ml_agent.py             # Machine learning specialist
│   ├── cleaning_agent.py       # Data quality specialist
│   └── base_agent.py           # Agent base class
│
├── src/                        # Core data & ML modules
│   ├── pma_engine.py           # Predictive Modeling Automation (912 lines)
│   ├── algorithms.py           # 40+ algorithm registry (628 lines)
│   ├── ml_suggester.py         # LLM-first algorithm suggestion
│   ├── automl.py               # Parallel AutoML training
│   ├── chart_library.py        # 25+ chart type builders
│   ├── dashboard_planner.py    # LLM-powered chart selection
│   ├── data_loader.py          # Universal file loader
│   ├── experiments.py          # Experiment tracking
│   ├── bia/                    # Business Intelligence engine
│   │   └── bia_engine.py       # BI analytics + MySQL connector
│   └── ...                     # Preprocessor, visualizer, etc.
│
├── llm/                        # LLM integration layer
│   ├── gemini_client.py        # Gemini API client
│   ├── prompt_templates.py     # Structured prompts for each agent
│   └── tools.py                # Function calling tools
│
├── auth_system/                # Authentication & security
│   ├── auth_routes.py          # Login, register, OTP, Google OAuth
│   ├── auth_middleware.py      # Route protection guard
│   ├── user_store.py           # User database (SQLite)
│   ├── otp_store.py            # OTP generation & validation
│   └── otp_sender.py           # Email/SMS OTP dispatch
│
├── models/                     # Saved ML model artifacts (.pkl)
├── data/                       # Raw, processed, and sample data
├── notebooks/                  # Jupyter notebooks (EDA, experiments)
├── tests/                      # Unit tests (pytest)
├── logo/                       # Branding assets
│
├── config.py                   # App configuration (env-driven)
├── run.py                      # Application entry point
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
└── .gitignore                  # Git exclusions
```

---

## 🔧 Configuration

All configuration is driven by environment variables in `.env`. See [`.env.example`](.env.example) for the full list.

| Variable | Required | Description |
|:---------|:--------:|:------------|
| `GEMINI_API_KEY` | ✅ | Google Gemini API key |
| `SECRET_KEY` | ✅ | Flask session secret |
| `DEBUG` | ❌ | Enable debug mode (`True`/`False`) |
| `SMTP_USER` | ❌ | Gmail address for email OTP |
| `SMTP_PASSWORD` | ❌ | Gmail app password |
| `GOOGLE_CLIENT_ID` | ❌ | Google OAuth client ID |
| `GOOGLE_CLIENT_SECRET` | ❌ | Google OAuth secret |

> 📖 Full email/SMS OTP and OAuth setup instructions are documented inside `.env.example`.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🗺️ Roadmap

- [ ] Migrate from `google-generativeai` to `google-genai` SDK
- [ ] Add collaborative workspaces (multi-user)
- [ ] Export dashboards as PDF reports
- [ ] Deploy to cloud (AWS / GCP)
- [ ] Add real-time data streaming support
- [ ] Plugin system for custom ML algorithms

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Raja** — [@Raja-718](https://github.com/Raja-718)

---

<p align="center">
  <strong>⭐ If you found this project useful, consider giving it a star!</strong>
</p>
