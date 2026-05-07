"""
Financial Assistant Web Application - Streamlit Frontend
Tabs: Chat | Data Browser | ML Predictions
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

st.set_page_config(
    page_title="Prologis Financial Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------
# Custom CSS for polish
# --------------------------------------------------------------
st.markdown("""
<style>
    /* Tighten main padding */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }

    /* Sidebar gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] h1 { color: #60a5fa; font-size: 1.5rem; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1); }

    /* Tabs styling — bigger, more breathing room */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        padding: 0 20px;
        background-color: rgba(96, 165, 250, 0.05);
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(96, 165, 250, 0.15);
        font-weight: 600;
    }

    /* Chat message styling */
    [data-testid="stChatMessage"] {
        background-color: rgba(30, 41, 59, 0.4);
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: none;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        border-radius: 8px;
        transition: transform 0.15s;
    }
    .stButton button:hover { transform: translateY(-1px); }

    /* Section headers a touch larger */
    h1 { font-size: 2.2rem; }
    h2 { font-size: 1.6rem; }

    /* Code blocks (sidebar endpoint names) */
    [data-testid="stSidebar"] code {
        font-size: 0.7rem;
        word-break: break-all;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------
@st.cache_resource
def get_db_engine():
    user = os.getenv("POSTGRES_USER", "postgres")
    pw = os.getenv("POSTGRES_PASSWORD", "postgres")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "financial_assistant")
    return create_engine(f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}")


def invoke_sagemaker(endpoint_name, payload):
    """Call a SageMaker endpoint with a JSON payload."""
    import boto3
    region = os.getenv("AWS_REGION", "us-east-1")
    runtime = boto3.client("sagemaker-runtime", region_name=region)
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode())


def safe_md(text):
    """Escape $ to prevent LaTeX rendering inside chat answers."""
    if not text:
        return text
    return text.replace("$", "\\$")


# --------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------
with st.sidebar:
    st.title("🏢 Prologis FinAssist")
    st.caption("AI-powered financial & property insights for an industrial REIT")
    st.divider()
    st.markdown("### 📂 Data Sources")
    st.markdown("""
    - **SEC EDGAR** — 10-K / 10-Q filings
    - **Postgres** — properties + financials
    - **Press Releases** — JSON store
    """)
    st.divider()
    st.markdown("### ☁️ Cloud Services")
    st.markdown("""
    - 🤖 **Gemini 2.5 Flash** — agent
    - 🔮 **AWS SageMaker** — ML endpoints
    - 📝 **AWS Bedrock** — summarization
    """)
    st.divider()
    st.markdown("### 🚀 ML Endpoints")
    reg_ep = os.getenv("SAGEMAKER_REGRESSION_ENDPOINT", "(not deployed)")
    clf_ep = os.getenv("SAGEMAKER_CLASSIFICATION_ENDPOINT", "(not deployed)")
    st.code(f"reg: {reg_ep}\nclf: {clf_ep}", language=None)
    st.divider()
    st.caption("Built for CS5500 Financial Assistant assignment.")

# --------------------------------------------------------------
# Main header
# --------------------------------------------------------------
st.title("🏢 Prologis Financial Assistant")
st.caption("End-to-end AI system: structured data + classic ML + generative AI on Postgres, AWS SageMaker, AWS Bedrock, and Google Gemini.")

# --------------------------------------------------------------
# Tabs
# --------------------------------------------------------------
tab_chat, tab_data, tab_ml = st.tabs(["💬 Chat", "📊 Data", "🤖 ML Predictions"])

# ============================================================
# TAB 1: CHAT
# ============================================================
with tab_chat:
    st.subheader("Conversational Assistant")
    st.caption("Ask about financials, properties, or recent press releases. The agent uses Gemini function calling to route your question to the right data source.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Example queries
    with st.expander("💡 Example queries", expanded=len(st.session_state.messages) == 0):
        cols = st.columns(2)
        examples = [
            "What was Prologis' net income last year?",
            "Show industrial properties in Chicago with revenue.",
            "Did Prologis announce any acquisitions recently?",
            "Summarize the most recent earnings press release.",
            "Compare property revenues between Dallas and Phoenix.",
            "Which metro has the highest average revenue per property?",
        ]
        for i, ex in enumerate(examples):
            with cols[i % 2]:
                st.markdown(f"- _{ex}_")

    # Render past messages (newest at bottom, like a real chat)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(safe_md(msg["content"]))
            if msg.get("tool_calls"):
                with st.expander(f"🔧 {len(msg['tool_calls'])} tool call(s)"):
                    for c in msg["tool_calls"]:
                        st.markdown(f"**{c['name']}**(`{c['args']}`)")
                        st.json(c["result"], expanded=False)

    # Chat input — Streamlit pins this at the bottom of the screen
    if prompt := st.chat_input("Ask a question about Prologis..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(safe_md(prompt))

        with st.chat_message("assistant"):
            try:
                from agent.agent import run_agent
                with st.spinner("Thinking..."):
                    result = run_agent(prompt)
                st.markdown(safe_md(result["answer"]))
                if result["tool_calls"]:
                    with st.expander(f"🔧 {len(result['tool_calls'])} tool call(s)"):
                        for c in result["tool_calls"]:
                            st.markdown(f"**{c['name']}**(`{c['args']}`)")
                            st.json(c["result"], expanded=False)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "tool_calls": result["tool_calls"],
                })
            except Exception as e:
                import traceback
                err = f"⚠️ Agent error: {e}"
                st.error(err)
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())
                st.session_state.messages.append({"role": "assistant", "content": err})


# ============================================================
# TAB 2: DATA BROWSER
# ============================================================
with tab_data:
    st.subheader("Data Browser")
    sub1, sub2, sub3 = st.tabs(["🏢 Properties", "📑 SEC Filings", "📰 Press Releases"])

    with sub1:
        st.markdown("#### Properties & Financials (Postgres)")
        try:
            engine = get_db_engine()
            sql = """
                SELECT p.property_id, p.address, p.metro_area, p.sq_footage,
                       p.property_type, f.revenue, f.net_income, f.expenses
                FROM properties p
                LEFT JOIN financials f ON p.property_id = f.property_id
                ORDER BY p.property_id
            """
            df = pd.read_sql(sql, engine)
            col1, col2 = st.columns(2)
            with col1:
                metro_filter = st.multiselect(
                    "Filter by metro",
                    options=sorted(df["metro_area"].unique()),
                )
            with col2:
                type_filter = st.multiselect(
                    "Filter by type",
                    options=sorted(df["property_type"].unique()),
                )
            view = df.copy()
            if metro_filter:
                view = view[view["metro_area"].isin(metro_filter)]
            if type_filter:
                view = view[view["property_type"].isin(type_filter)]
            # Summary stats
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Properties", len(view))
            mcol2.metric("Total revenue", f"${view['revenue'].sum()/1e6:.1f}M")
            mcol3.metric("Total net income", f"${view['net_income'].sum()/1e6:.1f}M")
            mcol4.metric("Avg revenue", f"${view['revenue'].mean()/1e6:.1f}M" if len(view) else "—")
            st.dataframe(view, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"DB connection failed: {e}")

    with sub2:
        st.markdown("#### SEC EDGAR — Prologis (NYSE: PLD)")
        sec_path = Path(__file__).parent.parent / "data" / "sec" / "prologis_financials.json"
        if sec_path.exists():
            data = json.loads(sec_path.read_text())
            rows = []
            for name, m in data.get("metrics", {}).items():
                latest = m.get("latest_annual") or {}
                rows.append({
                    "Metric": name,
                    "Value (USD)": f"${latest.get('val', 0):,}",
                    "FY End": latest.get("end", "—"),
                    "Form": latest.get("form", "—"),
                })
            st.table(rows)
            with st.expander("View raw JSON"):
                st.json(data, expanded=False)
        else:
            st.warning("Run `python scripts/fetch_sec.py` to populate SEC data.")

    with sub3:
        st.markdown("#### Recent Press Releases")
        pr_path = Path(__file__).parent.parent / "data" / "press_releases.json"
        if pr_path.exists():
            releases = json.loads(pr_path.read_text())
            categories = sorted(set(r["category"] for r in releases))
            cat_filter = st.multiselect("Filter by category", options=categories)
            filtered = [r for r in releases if not cat_filter or r["category"] in cat_filter]
            st.caption(f"Showing {len(filtered)} of {len(releases)} releases")
            for pr in filtered:
                with st.expander(f"📰 {pr['date']} — {pr['title']}"):
                    st.markdown(f"**Category:** `{pr['category']}`")
                    st.markdown(pr["content"])


# ============================================================
# TAB 3: ML PREDICTIONS
# ============================================================
with tab_ml:
    st.subheader("ML Model Predictions")
    st.caption("Both models are deployed as live SageMaker endpoints and called over HTTPS.")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("### 🏠 Housing Price")
        st.caption("**Random Forest** on California Housing — predicts median house value")

        med_inc = st.slider("Median Income (10k USD)", 0.5, 15.0, 5.0)
        house_age = st.slider("House Age (years)", 1, 52, 25)
        avg_rooms = st.slider("Avg Rooms", 1.0, 10.0, 5.0)
        avg_bedrms = st.slider("Avg Bedrooms", 0.5, 3.0, 1.0)
        population = st.slider("Population", 100, 5000, 1500)
        avg_occup = st.slider("Avg Occupancy", 1.0, 6.0, 3.0)
        latitude = st.slider("Latitude", 32.0, 42.0, 34.0)
        longitude = st.slider("Longitude", -125.0, -114.0, -118.0)

        if st.button("🎯 Predict House Value", key="reg_btn", use_container_width=True):
            payload = {
                "MedInc": med_inc, "HouseAge": house_age,
                "AveRooms": avg_rooms, "AveBedrms": avg_bedrms,
                "Population": population, "AveOccup": avg_occup,
                "Latitude": latitude, "Longitude": longitude,
            }
            ep = os.getenv("SAGEMAKER_REGRESSION_ENDPOINT")
            if not ep:
                st.error("SAGEMAKER_REGRESSION_ENDPOINT not set")
            else:
                try:
                    with st.spinner("Calling SageMaker..."):
                        result = invoke_sagemaker(ep, payload)
                    val = result[0]["predicted_value_usd"]
                    st.success(f"### 💰 ${val:,.0f}")
                    st.caption("Predicted median house value")
                    with st.expander("Raw response"):
                        st.json(result)
                except Exception as e:
                    st.error(f"Endpoint call failed: {e}")

    with col2:
        st.markdown("### 🏦 Subscription Likelihood")
        st.caption("**Logistic Regression** on UCI Bank Marketing — predicts subscription")

        age = st.number_input("Age", 18, 95, 35)
        job = st.selectbox("Job", [
            "admin.", "blue-collar", "technician", "services", "management",
            "retired", "self-employed", "entrepreneur", "unemployed",
            "housemaid", "student", "unknown"
        ])
        marital = st.selectbox("Marital", ["married", "single", "divorced"])
        education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
        default = st.selectbox("Has credit in default?", ["no", "yes"])
        housing = st.selectbox("Has housing loan?", ["no", "yes"])
        loan = st.selectbox("Has personal loan?", ["no", "yes"])
        contact = st.selectbox("Contact type", ["cellular", "telephone", "unknown"])
        month = st.selectbox("Last contact month", [
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec"
        ])
        poutcome = st.selectbox("Previous outcome", ["unknown", "failure", "other", "success"])
        balance = st.number_input("Balance (EUR)", -5000, 100000, 1500)
        duration = st.number_input("Last Contact Duration (s)", 0, 5000, 200)
        campaign = st.number_input("# contacts this campaign", 1, 50, 1)
        pdays = st.number_input("Days since last contact (-1 = never)", -1, 1000, -1)
        previous = st.number_input("# contacts before this campaign", 0, 50, 0)

        if st.button("🎯 Predict Subscription", key="clf_btn", use_container_width=True):
            payload = {
                "age": age, "job": job, "marital": marital, "education": education,
                "default": default, "housing": housing, "loan": loan,
                "contact": contact, "month": month, "poutcome": poutcome,
                "balance": balance, "duration": duration,
                "campaign": campaign, "pdays": pdays, "previous": previous,
            }
            ep = os.getenv("SAGEMAKER_CLASSIFICATION_ENDPOINT")
            if not ep:
                st.error("SAGEMAKER_CLASSIFICATION_ENDPOINT not set")
            else:
                try:
                    with st.spinner("Calling SageMaker..."):
                        result = invoke_sagemaker(ep, payload)
                    pred = result[0]
                    label = pred["label"]
                    prob = pred["probability"]
                    if label == "yes":
                        st.success(f"### ✅ Will subscribe")
                        st.caption(f"Confidence: {prob:.1%}")
                    else:
                        st.warning(f"### ❌ Will NOT subscribe")
                        st.caption(f"Probability of yes: {prob:.1%}")
                    with st.expander("Raw response"):
                        st.json(result)
                except Exception as e:
                    st.error(f"Endpoint call failed: {e}")
