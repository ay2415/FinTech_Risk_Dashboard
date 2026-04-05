import streamlit as st
import pandas as pd
import plotly.express as px
from data_loader import get_clean_data
from analysis import calculate_risk_metrics, get_fraud_stats

st.set_page_config(page_title="FinTech Risk Monitor", layout="wide")

@st.cache_data

def load_data(nrows):
    data = pd.read_csv('data/paysim.csv', nrows=nrows)
    return data

st.title(" FinTech Risk & Fraud Dashboard")
st.markdown("Real-time monitoring of PaySim synthetic transactions.")

raw_data = load_data(100000)
df = calculate_risk_metrics(raw_data)
st.sidebar.header("Global Filters")
t_type = st.sidebar.multiselect(
    "Select Transaction Type:",
    options=df["type"].unique(),
    default=df["type"].unique()
)

filtered_df = df[df["type"].isin(t_type)]

fraud_count, fraud_val = get_fraud_stats(filtered_df)


c1, c2, c3 = st.columns(3)
c1.metric("Transactions Scanned", f"{len(filtered_df):,}")
c2.metric("Fraud Cases Identified", f"{fraud_count}")
c3.metric("Total Risk Value", f"${fraud_val:,.2f}")


# 5. Visualizing the 'Balance Error'
st.subheader("The 'Balance Error' Analysis")

if not filtered_df.empty:
    # Logic: Only sample if we have enough rows, otherwise use what we have
    sample_size = min(len(filtered_df), 2000)
    
    if sample_size > 0:
        fig = px.scatter(filtered_df.sample(sample_size), 
                         x="amount", y="balance_error", 
                         color="isFraud",
                         title="Amount vs. Balance Discrepancy",
                         labels={"balance_error": "Math Error in Balance"},
                         hover_data=['type'])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data matches the selected filters.")
else:
    st.warning("Please select at least one transaction type in the sidebar.")