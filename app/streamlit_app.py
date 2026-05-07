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
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

st.set_page_config(
    page_title="Prologis Financial Assistant",
    page_icon="🏢",
    layout="wide",
)

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


# --------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------
with st.sidebar:
    st.title("🏢 Prologis FinAssist")
    st.caption("AI-powered financial & property insights")
    st.divider()
    st.markdown("**Data Sources**")
    st.markdown("• SEC EDGAR (10-K/10-Q)")
    st.markdown("• Postgres (properties + financials)")
    st.markdown("• Press releases (JSON)")
    st.divider()
    st.markdown("**Cloud Services**")
    st.markdown("• Gemini 2.5 Flash (agent)")
    st.markdown("• AWS SageMaker (ML endpoints)")
    st.markdown("• AWS Bedrock (summarization)")
    st.divider()
    st.markdown("**ML Endpoints**")
    reg_ep = os.getenv("SAGEMAKER_REGRESSION_ENDPOINT", "(not deployed)")
    clf_ep = os.getenv("SAGEMAKER_CLASSIFICATION_ENDPOINT", "(not deployed)")
    st.code(f"reg: {reg_ep}\nclf: {clf_ep}", language=None)

# --------------------------------------------------------------
# Tabs
# --------------------------------------------------------------
tab_chat, tab_data, tab_ml = st.tabs(["💬 Chat", "📊 Data", "🤖 ML Predictions"])

# ============================================================
# TAB 1: CHAT (with Gemini agent)
# ============================================================
with tab_chat:
    st.header("Conversational Assistant")
    st.caption("Ask about financials, properties, or recent press releases. Powered by Gemini + function calling.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.expander("Example queries"):
        st.markdown("""
        - What was Prologis' net income last year?
        - Show industrial properties in the Chicago metro area with revenue.
        - Did Prologis announce any acquisitions recently?
        - Summarize the most recent earnings press release.
        - Compare property revenues between Dallas and Phoenix.
        """)

    # Render past messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tool_calls"):
                with st.expander(f"🔧 Tools used: {', '.join(c['name'] for c in msg['tool_calls'])}"):
                    for c in msg["tool_calls"]:
                        st.markdown(f"**{c['name']}**(`{c['args']}`)")
                        st.json(c["result"], expanded=False)

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                # Lazy import so the app starts even if agent has issues
                from agent.agent import run_agent
                with st.spinner("Thinking..."):
                    result = run_agent(prompt)
                st.markdown(result["answer"])
                if result["tool_calls"]:
                    with st.expander(f"🔧 Tools used: {', '.join(c['name'] for c in result['tool_calls'])}"):
                        for c in result["tool_calls"]:
                            st.markdown(f"**{c['name']}**(`{c['args']}`)")
                            st.json(c["result"], expanded=False)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "tool_calls": result["tool_calls"],
                })
            except Exception as e:
                err = f"⚠️ Agent error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})


# ============================================================
# TAB 2: DATA BROWSER
# ============================================================
with tab_data:
    st.header("Data Browser")
    sub1, sub2, sub3 = st.tabs(["Properties", "SEC Filings", "Press Releases"])

    with sub1:
        st.subheader("Properties & Financials (Postgres)")
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
            st.dataframe(view, use_container_width=True, hide_index=True)
            st.caption(f"Showing {len(view)} of {len(df)} properties")
        except Exception as e:
            st.error(f"DB connection failed: {e}")

    with sub2:
        st.subheader("SEC EDGAR — Prologis")
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
        st.subheader("Press Releases")
        pr_path = Path(__file__).parent.parent / "data" / "press_releases.json"
        if pr_path.exists():
            releases = json.loads(pr_path.read_text())
            for pr in releases:
                with st.expander(f"{pr['date']} — {pr['title']}"):
                    st.markdown(f"**Category:** {pr['category']}")
                    st.markdown(pr["content"])


# ============================================================
# TAB 3: ML PREDICTIONS
# ============================================================
with tab_ml:
    st.header("ML Model Predictions")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏠 Housing Price (Regression)")
        st.caption("Random Forest on California Housing — deployed to SageMaker")

        med_inc = st.slider("Median Income (10k USD)", 0.5, 15.0, 5.0)
        house_age = st.slider("House Age", 1, 52, 25)
        avg_rooms = st.slider("Avg Rooms", 1.0, 10.0, 5.0)
        avg_bedrms = st.slider("Avg Bedrooms", 0.5, 3.0, 1.0)
        population = st.slider("Population", 100, 5000, 1500)
        avg_occup = st.slider("Avg Occupancy", 1.0, 6.0, 3.0)
        latitude = st.slider("Latitude", 32.0, 42.0, 34.0)
        longitude = st.slider("Longitude", -125.0, -114.0, -118.0)

        if st.button("Predict House Value", key="reg_btn"):
            payload = {
                "MedInc": med_inc, "HouseAge": house_age,
                "AveRooms": avg_rooms, "AveBedrms": avg_bedrms,
                "Population": population, "AveOccup": avg_occup,
                "Latitude": latitude, "Longitude": longitude,
            }
            ep = os.getenv("SAGEMAKER_REGRESSION_ENDPOINT")
            if not ep:
                st.error("SAGEMAKER_REGRESSION_ENDPOINT not set in .env")
            else:
                try:
                    with st.spinner("Calling SageMaker..."):
                        result = invoke_sagemaker(ep, payload)
                    val = result[0]["predicted_value_usd"]
                    st.success(f"**Predicted value: ${val:,.0f}**")
                    with st.expander("Raw response"):
                        st.json(result)
                except Exception as e:
                    st.error(f"Endpoint call failed: {e}")

    with col2:
        st.subheader("🏦 Subscription Likelihood (Classification)")
        st.caption("Logistic Regression on Bank Marketing — deployed to SageMaker")

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

        if st.button("Predict Subscription", key="clf_btn"):
            payload = {
                "age": age, "job": job, "marital": marital, "education": education,
                "default": default, "housing": housing, "loan": loan,
                "contact": contact, "month": month, "poutcome": poutcome,
                "balance": balance, "duration": duration,
                "campaign": campaign, "pdays": pdays, "previous": previous,
            }
            ep = os.getenv("SAGEMAKER_CLASSIFICATION_ENDPOINT")
            if not ep:
                st.error("SAGEMAKER_CLASSIFICATION_ENDPOINT not set in .env")
            else:
                try:
                    with st.spinner("Calling SageMaker..."):
                        result = invoke_sagemaker(ep, payload)
                    pred = result[0]
                    label = pred["label"]
                    prob = pred["probability"]
                    if label == "yes":
                        st.success(f"**Will subscribe** (probability: {prob:.1%})")
                    else:
                        st.warning(f"**Will NOT subscribe** (probability of yes: {prob:.1%})")
                    with st.expander("Raw response"):
                        st.json(result)
                except Exception as e:
                    st.error(f"Endpoint call failed: {e}")
