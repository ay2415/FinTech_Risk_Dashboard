import streamlit as st
import pandas as pd
import plotly.express as px
from data_loader import get_clean_data
from analysis import calculate_risk_metrics, get_fraud_stats, prepare_model_data, predict_single_transaction
from npu_inference import NPUInspector

st.set_page_config(page_title="FinTech Risk Monitor", layout="wide")

if "model" not in st.session_state:
    st.session_state.model = None
if "le" not in st.session_state:
    st.session_state.le = None

@st.cache_data
def load_data(nrows):
    return pd.read_csv('data/paysim.csv', nrows=nrows)

st.title(" FinTech Risk & Fraud Dashboard")
st.markdown("Real-time monitoring of PaySim synthetic transactions.")

raw_data = load_data(100000)
df = calculate_risk_metrics(raw_data)

# ── Sidebar filters ──────────────────────────────────────────────────────────
st.sidebar.header("Global Filters")
t_type = st.sidebar.multiselect(
    "Select Transaction Type:",
    options=df["type"].unique(),
    default=df["type"].unique()
)
filtered_df = df[df["type"].isin(t_type)]

fraud_count, fraud_val = get_fraud_stats(filtered_df)

# ── KPI cards ────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Transactions Scanned", f"{len(filtered_df):,}")
c2.metric("Fraud Cases Identified", f"{fraud_count}")
c3.metric("Total Risk Value", f"${fraud_val:,.2f}")

# ── Balance Error scatter ────────────────────────────────────────────────────
st.subheader("The 'Balance Error' Analysis")
if not filtered_df.empty:
    sample_size = min(len(filtered_df), 2000)
    fig = px.scatter(
        filtered_df.sample(sample_size),
        x="amount", y="balance_error",
        color="isFraud",
        title="Amount vs. Balance Discrepancy",
        labels={"balance_error": "Math Error in Balance"},
        hover_data=['type']
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Please select at least one transaction type in the sidebar.")

# ── ML Training ──────────────────────────────────────────────────────────────
st.divider()
st.header("Machine Learning Intelligence")

# Show a green badge if model is already trained
if st.session_state.model is not None:
    st.success(" Model is trained and ready. You can run the simulator below.")
else:
    st.info("Click the button below to train the AI on 100,000 transactions.")

if st.button("Run AI Training & Analysis"):
    with st.spinner("AI is analyzing transaction patterns..."):
        model, le, report, X_test, y_test = prepare_model_data(df)

        st.session_state.model = model
        st.session_state.le    = le

    st.success("AI Training Complete!")

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.subheader("Model Accuracy Report")
        st.code(report)
    with res_col2:
        st.subheader(" Top Fraud Indicators")
        importance = pd.DataFrame({
            'Feature': ['Step', 'Type', 'Amount', 'Old Balance', 'New Balance', 'Balance Error'],
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        st.bar_chart(importance.set_index('Feature'))

st.divider()
st.header("Interactive Fraud Simulator")

col_a, col_b = st.columns(2)
with col_a:
    input_type    = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"])
    input_amt     = st.number_input("Transaction Amount ($)", min_value=0.0, value=1000.0)
with col_b:
    input_old_bal = st.number_input("Current Account Balance ($)", min_value=0.0, value=5000.0)
    input_new_bal = input_old_bal - input_amt

if st.button("⚡ Run Investigation"):
    if st.session_state.model is None:
        st.warning(" Please run AI Training above first!")
    else:
        test_data = {
            'step':           1,
            'type':           input_type,
            'amount':         input_amt,
            'oldbalanceOrg':  input_old_bal,
            'newbalanceOrig': input_new_bal,
        }

        risk_score = predict_single_transaction(
            st.session_state.model,
            st.session_state.le,
            test_data
        )

        if risk_score > 0.5:
            st.error(f" HIGH RISK: {risk_score:.2%} Probability of Fraud!")
        else:
            st.success(f" LOW RISK: {risk_score:.2%} Probability of Fraud.")