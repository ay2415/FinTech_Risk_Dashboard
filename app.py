import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from analysis import calculate_risk_metrics, get_fraud_stats, prepare_model_data, predict_single_transaction
from agents.workflow import run_investigation_pipeline
from behavioral import calculate_user_history, evaluate_behavioral_anomaly

st.set_page_config(page_title="FinTech Risk & Multi-Agent Investigation", layout="wide")

@st.cache_data
def load_data(nrows):
    data = pd.read_csv('data/paysim.csv', nrows=nrows)
    return data

st.title("🛡️ FinTech Risk & Multi-Agent Investigation System")
st.markdown("Real-time monitoring of PaySim financial transactions with Behavioural Anomaly Detection & Multi-Agent AI.")

raw_data = load_data(100000)
df = calculate_risk_metrics(raw_data)

le = LabelEncoder()
df['type_encoded'] = le.fit_transform(df['type'])

# ----------------- SIDEBAR -----------------
st.sidebar.header("Global Filters & Settings")
t_type = st.sidebar.multiselect(
    "Select Transaction Type:",
    options=df["type"].unique(),
    default=df["type"].unique()
)

model_choice = st.sidebar.selectbox(
    "Agent LLM Model:",
    ["llama3.1:8b", "mistral", "gpt-4o-mini"],
    index=0,
    help="Default queries local Ollama on localhost:11434. Uses deterministic fallback if offline."
)

filtered_df = df[df["type"].isin(t_type)]
fraud_count, fraud_val = get_fraud_stats(filtered_df)

# ----------------- METRICS -----------------
c1, c2, c3 = st.columns(3)
c1.metric("Transactions Scanned", f"{len(filtered_df):,}")
c2.metric("Fraud Cases Identified", f"{fraud_count}")
c3.metric("Total Risk Value", f"${fraud_val:,.2f}")

# ----------------- VISUALIZATION -----------------
st.subheader("The Balance Error Analysis")

if not filtered_df.empty:
    sample_size = min(len(filtered_df), 2000)
    fig = px.scatter(
        filtered_df.sample(sample_size), 
        x="amount", 
        y="balance_error", 
        color="isFraud",
        title="Amount vs. Balance Discrepancy",
        labels={"balance_error": "Math Error in Balance ($)"},
        hover_data=['type']
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ----------------- MACHINE LEARNING INTELLIGENCE -----------------
st.header("Machine Learning Intelligence")

if st.button("Run AI Training & Analysis"):
    with st.spinner("Training Random Forest Classifier on historical transactions..."):
        model, report, X_test, y_test = prepare_model_data(df)
        
        st.session_state['trained_model'] = model
        st.session_state['encoder'] = le
        st.session_state['report'] = report
        
        st.success("AI Training Complete")
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.subheader("Model Accuracy Report")
            st.code(report)
        with res_col2:
            st.subheader("Top Fraud Indicators")
            importance = pd.DataFrame({
                'Feature': ['Step', 'Type', 'Amount', 'Old Balance', 'New Balance', 'Balance Error'],
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            st.bar_chart(importance.set_index('Feature'))

st.divider()

# ----------------- MULTI-AGENT FRAUD INVESTIGATION & SIMULATOR -----------------
st.header("🤖 Multi-Agent Fraud Investigation & Simulator")
st.markdown("""
Evaluate transactions through a 4-stage Multi-Agent pipeline supporting **both Outflows (Sent/Withdrawn)** and **Inflows (CASH_IN / Received)**:
**1. Detection Agent (Math + Behavioural Rules)** $\\rightarrow$ 
**2. Analysis Agent (Hypothesis Generation)** $\\rightarrow$ 
**3. Verification Agent (Audit & Risk Tiering)** $\\rightarrow$ 
**4. Reporting Agent (Executive Brief)**
""")

preset = st.selectbox(
    "Choose a Scenario Preset or Custom:",
    [
        "Preset 1: Extreme Behavioural Surge ($30,000 for a $28/tx user)",
        "Preset 2: Impossible Math ($0 Balance Outflow Anomaly)",
        "Preset 3: Normal Spending (Math & History Match)",
        "Preset 4: CASH_IN Deposit (Money Received into Account)",
        "Preset 5: High-Net-Worth Account ($45,000 for a $50k/tx user)",
        "Preset 6: Brand New Account (Cold Start / No History)",
        "Custom Input"
    ]
)

# Preset Configuration
if preset == "Preset 1: Extreme Behavioural Surge ($30,000 for a $28/tx user)":
    default_type = "CASH_OUT"
    default_amt = 30000.0
    default_old_bal = 30000.0
    default_new_bal = 0.0
    sample_history = [
        {"amount": 20.0, "type": "PAYMENT", "step": 10},
        {"amount": 30.0, "type": "PAYMENT", "step": 15},
        {"amount": 15.0, "type": "PAYMENT", "step": 20},
        {"amount": 40.0, "type": "PAYMENT", "step": 25},
        {"amount": 25.0, "type": "PAYMENT", "step": 30},
        {"amount": 35.0, "type": "PAYMENT", "step": 35},
    ]
elif preset == "Preset 2: Impossible Math ($0 Balance Outflow Anomaly)":
    default_type = "CASH_OUT"
    default_amt = 50000.0
    default_old_bal = 0.0
    default_new_bal = 0.0
    sample_history = [
        {"amount": 50.0, "type": "PAYMENT", "step": 10},
        {"amount": 80.0, "type": "PAYMENT", "step": 20}
    ]
elif preset == "Preset 3: Normal Spending (Math & History Match)":
    default_type = "TRANSFER"
    default_amt = 30.0
    default_old_bal = 500.0
    default_new_bal = 470.0
    sample_history = [
        {"amount": 20.0, "type": "PAYMENT", "step": 10},
        {"amount": 30.0, "type": "TRANSFER", "step": 15},
        {"amount": 25.0, "type": "PAYMENT", "step": 20}
    ]
elif preset == "Preset 4: CASH_IN Deposit (Money Received into Account)":
    default_type = "CASH_IN"
    default_amt = 2500.0
    default_old_bal = 1000.0
    default_new_bal = 3500.0  # Inflow: 1000 + 2500 = 3500
    sample_history = [
        {"amount": 2000.0, "type": "CASH_IN", "step": 10},
        {"amount": 500.0, "type": "PAYMENT", "step": 15},
        {"amount": 2200.0, "type": "CASH_IN", "step": 20}
    ]
elif preset == "Preset 5: High-Net-Worth Account ($45,000 for a $50k/tx user)":
    default_type = "TRANSFER"
    default_amt = 45000.0
    default_old_bal = 100000.0
    default_new_bal = 55000.0
    sample_history = [
        {"amount": 40000.0, "type": "TRANSFER", "step": 10},
        {"amount": 60000.0, "type": "TRANSFER", "step": 20},
        {"amount": 50000.0, "type": "TRANSFER", "step": 30}
    ]
elif preset == "Preset 6: Brand New Account (Cold Start / No History)":
    default_type = "PAYMENT"
    default_amt = 150.0
    default_old_bal = 500.0
    default_new_bal = 350.0
    sample_history = []
else:
    default_type = "TRANSFER"
    default_amt = 1000.0
    default_old_bal = 5000.0
    default_new_bal = 4000.0
    sample_history = [
        {"amount": 800.0, "type": "PAYMENT", "step": 10},
        {"amount": 1200.0, "type": "TRANSFER", "step": 20}
    ]

all_types = ["TRANSFER", "CASH_OUT", "CASH_IN", "PAYMENT", "DEBIT"]
col_a, col_b = st.columns(2)
with col_a:
    input_type = st.selectbox(
        "Transaction Type", 
        all_types, 
        index=all_types.index(default_type),
        help="CASH_IN = Money Received / Deposited (Inflow). TRANSFER/CASH_OUT/PAYMENT/DEBIT = Money Sent / Withdrawn (Outflow)."
    )
    input_amt = st.number_input("Transaction Amount ($)", min_value=0.0, value=default_amt, step=100.0)
with col_b:
    input_old_bal = st.number_input("Starting Account Balance ($)", min_value=0.0, value=default_old_bal, step=100.0)
    
    # Provide helpful hint for expected ending balance
    if input_type == "CASH_IN":
        expected_calc = input_old_bal + input_amt
        hint_text = f"Expected Ending Balance for CASH_IN: ${expected_calc:,.2f} (Old Bal + Amount)"
    else:
        expected_calc = max(0.0, input_old_bal - input_amt)
        hint_text = f"Expected Ending Balance for Outflow: ${expected_calc:,.2f} (Old Bal - Amount)"
        
    input_new_bal = st.number_input(
        "Ending Account Balance ($)", 
        min_value=0.0, 
        value=default_new_bal, 
        step=100.0,
        help=hint_text
    )

# Calculate and display user baseline preview
hist_stats = calculate_user_history(sample_history, current_step=40)
beh_eval = evaluate_behavioral_anomaly(
    {"amount": input_amt, "type": input_type, "nameOrig": "SIM_USER"}, 
    hist_stats
)

st.markdown("##### 👤 User Historical Baseline Summary")
bc1, bc2, bc3, bc4, bc5 = st.columns(5)
bc1.metric("Historical Tx Count", f"{hist_stats['count']}")
bc2.metric("Historical Median", f"${hist_stats['median']:,.2f}")
bc3.metric("Historical Max", f"${hist_stats['max']:,.2f}")
bc4.metric("Amount Ratio", f"{beh_eval['amount_ratio']:.1f}x" if not beh_eval['is_cold_start'] else "N/A (Cold Start)")
if beh_eval['behavioural_anomaly']:
    bc5.error("⚠️ Anomaly Detected")
else:
    bc5.success("✅ Baseline Normal")

txn_payload = {
    'step': 40,
    'type': input_type,
    'amount': input_amt,
    'oldbalanceOrg': input_old_bal,
    'newbalanceOrig': input_new_bal,
    'nameOrig': "SIM_USER"
}

btn_col1, btn_col2 = st.columns(2)

with btn_col1:
    run_agents = st.button("🚀 Execute Multi-Agent Investigation", type="primary")

with btn_col2:
    run_ml = st.button("📊 Run Baseline ML Model Check")

if run_agents:
    with st.status("Executing Multi-Agent Fraud Investigation Pipeline...", expanded=True) as status:
        st.write("🔍 **Agent 1 (Detection):** Evaluating financial math invariants & user behavioural deviations...")
        pipeline_result = run_investigation_pipeline(txn_payload, user_history=sample_history, model_name=model_choice)
        
        st.write("🧠 **Agent 2 (Analysis):** Formulating contextual hypotheses...")
        st.write("⚖️ **Agent 3 (Verification):** Auditing evidence & assigning risk tier...")
        st.write("📄 **Agent 4 (Reporting):** Synthesizing executive investigation brief...")
        
        risk_tier = pipeline_result["risk_tier"]
        status.update(label=f"Investigation Complete: Risk Tier {risk_tier}", state="complete", expanded=True)

    # Display Results in Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Executive Report (Agent 4)",
        "🔍 Invariant & Behavioural Findings (Agent 1)",
        "🧠 Contextual Analysis (Agent 2)",
        "⚖️ Verification Audit (Agent 3)"
    ])

    with tab1:
        st.markdown(pipeline_result["report"]["report_markdown"])

    with tab2:
        det = pipeline_result["detection"]
        beh = pipeline_result.get("behavioral")
        
        st.markdown("#### ⚙️ Deterministic Detection Engine Findings")
        st.info(f"**Transaction Flow Direction:** {det['metrics']['direction']}")
        if det["is_suspicious"]:
            st.error(f"Flagged {len(det['anomalies'])} Discrepancies / Anomalies:")
            for anom in det["anomalies"]:
                st.write(f"- ⚠️ {anom}")
        else:
            st.success("✅ All balance invariants satisfied & within historical behavioral baseline.")
        
        st.markdown("##### Detailed Metrics")
        st.json({
            "financial_invariants": det["metrics"],
            "user_behavioral_baseline": beh
        })

    with tab3:
        st.markdown("#### 🧠 AI Risk Analyst Insights")
        st.markdown(pipeline_result["analysis"]["analysis_text"])

    with tab4:
        st.markdown("#### ⚖️ Independent Verification & Evidence Audit")
        ver = pipeline_result["verification"]
        tier = ver["risk_tier"]
        if tier == "HIGH":
            st.error(f"**Verified Risk Tier: HIGH** (Risk Score: {ver.get('risk_score_points', 0)}/100)")
        elif tier == "MEDIUM":
            st.warning(f"**Verified Risk Tier: MEDIUM** (Risk Score: {ver.get('risk_score_points', 0)}/100)")
        else:
            st.success(f"**Verified Risk Tier: LOW** (Risk Score: {ver.get('risk_score_points', 0)}/100)")
        st.markdown(ver["audit_notes"])

if run_ml:
    if 'trained_model' in st.session_state:
        risk_score = predict_single_transaction(
            st.session_state['trained_model'], 
            st.session_state['encoder'], 
            txn_payload
        )
        if risk_score > 0.5:
            st.error(f"HIGH RISK: {risk_score:.2%} Statistical Probability of Fraud")
        else:
            st.success(f"LOW RISK: {risk_score:.2%} Statistical Probability of Fraud")
    else:
        st.warning("Please click 'Run AI Training & Analysis' above first to train the baseline ML model.")
