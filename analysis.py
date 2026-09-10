import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def calculate_risk_metrics(df):
    """
    Logic: Detect 'Impossible Transactions'.
    For CASH_IN (Inflows): Expected = Old Balance + Amount
    For Outflows (TRANSFER, CASH_OUT, PAYMENT, DEBIT): Expected = Old Balance - Amount
    """
    is_cash_in = df['type'] == 'CASH_IN'
    df['balance_error'] = 0.0
    
    # Vectorized calculation for performance
    df.loc[is_cash_in, 'balance_error'] = (df.loc[is_cash_in, 'oldbalanceOrg'] + df.loc[is_cash_in, 'amount']) - df.loc[is_cash_in, 'newbalanceOrig']
    df.loc[~is_cash_in, 'balance_error'] = (df.loc[~is_cash_in, 'oldbalanceOrg'] - df.loc[~is_cash_in, 'amount']) - df.loc[~is_cash_in, 'newbalanceOrig']
    
    # Flag transactions where the error is significant
    df['is_suspicious'] = df['balance_error'].apply(lambda x: 1 if abs(x) > 0.1 else 0)
    
    return df

def prepare_model_data(df):
    le = LabelEncoder()
    df['type_encoded'] = le.fit_transform(df['type'])

    X = df[['step', 'type_encoded', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'balance_error']]
    y = df['isFraud']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)

    return model, report, X_test, y_test

def get_fraud_stats(df):
    """Calculates high-level stats for the UI cards."""
    total_fraud = int(df['isFraud'].sum())
    total_amt_fraud = float(df[df['isFraud'] == 1]['amount'].sum())
    return total_fraud, total_amt_fraud

def predict_single_transaction(model, le, input_data):
    input_data_copy = dict(input_data)
    if input_data_copy.get('type') == 'CASH_IN':
        input_data_copy['balance_error'] = (input_data_copy['oldbalanceOrg'] + input_data_copy['amount']) - input_data_copy['newbalanceOrig']
    else:
        input_data_copy['balance_error'] = (input_data_copy['oldbalanceOrg'] - input_data_copy['amount']) - input_data_copy['newbalanceOrig']
    
    input_data_copy['type_encoded'] = le.transform([input_data_copy['type']])[0]
    
    features = ['step', 'type_encoded', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'balance_error']
    X_single = pd.DataFrame([input_data_copy])[features]
    
    probability = float(model.predict_proba(X_single)[0][1])
    return probability
