import pandas as pd

def calculate_risk_metrics(df):
    """
    Logic: Detect 'Impossible Transactions'.
    If (Old Balance - Amount) != New Balance, something is wrong.
    """
    # Create the 'Balance Error' feature
    # Formula: Expected New Balance vs Actual New Balance
    df['balance_error'] = (df['oldbalanceOrg'] - df['amount']) - df['newbalanceOrig']
    
    # Flag transactions where the error is significant
    df['is_suspicious'] = df['balance_error'].apply(lambda x: 1 if abs(x) > 0.1 else 0)
    
    return df

def get_fraud_stats(df):
    """Calculates high-level stats for the UI cards."""
    total_fraud = df['isFraud'].sum()
    total_amt_fraud = df[df['isFraud'] == 1]['amount'].sum()
    return total_fraud, total_amt_fraud