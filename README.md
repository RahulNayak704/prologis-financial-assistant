# Prologis Financial Assistant

End-to-end AI-powered financial assistant for Prologis (NYSE: PLD), an industrial REIT. Combines structured data querying, classic ML models, and generative AI for natural language insights.

## Architecture

```
Streamlit UI ──┬── Gemini Agent (function-calling)
               │     ├─ tool: query_postgres
               │     ├─ tool: query_sec_edgar
               │     └─ tool: query_press_releases
               │
               ├── SageMaker Endpoint (Random Forest, housing)
               ├── SageMaker Endpoint (Logistic Regression, bank)
               └── AWS Bedrock (summarization fallback)
```

## Data sources
1. **SEC EDGAR** — Prologis 10-K and 10-Q financial metrics (revenue, net income, etc.) via the SEC company facts API.
2. **Postgres** — `properties` and `financials` tables seeded with 20 industrial properties across major US metros.
3. **Press releases** — JSON store with 10 Prologis press releases (acquisitions, expansions, earnings).

## Local setup

```bash
# 1. Clone + venv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Postgres (Docker)
docker run --name fin-postgres -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=financial_assistant -p 5432:5432 -d postgres:15

# 3. Schema + seed
psql -h localhost -U postgres -d financial_assistant -f db/schema.sql
psql -h localhost -U postgres -d financial_assistant -f db/seed.sql

# 4. SEC data
python scripts/fetch_sec.py

# 5. Env
cp .env.example .env   # then fill in keys

# 6. Run app
streamlit run app/streamlit_app.py
```

## Notes on Vertex AI vs Gemini API
The chatbot uses Gemini 2.5 Flash via the `google-generativeai` SDK (AI Studio). The agent pattern (tool declaration, function calling, multi-turn orchestration) is identical to Vertex AI ADK and the code is portable to Vertex AI by swapping the client initialization.

## ML models
- **Regression**: Random Forest on California Housing — predicts median house value.
- **Classification**: Logistic Regression on UCI Bank Marketing — predicts subscription.
- Both deployed to Amazon SageMaker as hosted endpoints.

See `ml/regression/` and `ml/classification/` for training notebooks and inference scripts.
