import pandas as pd
import streamlit as st

@st.cache_data
def get_clean_data(file_path, nrows=200000):
    """
    Loads and performs initial cleaning.
    Concept: Memory Mapping. We only load what we need to prevent crashes.
    """
    df = pd.read_csv(file_path, nrows=nrows)
    
    # Drop columns that don't help with initial risk 
    # df = df.drop(['oldbalanceDest', 'newbalanceDest'], axis=1)
    
    return df