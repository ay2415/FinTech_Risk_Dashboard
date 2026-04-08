import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

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

def prepare_model_data(df):
    le= LabelEncoder()
    df['type_encoded'] = le.fit_transform(df['type'])

    X=df[['step', 'type_encoded', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'balance_error']]
    y=df['isFraud']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)

    return model, report, X_test, y_test

def get_fraud_stats(df):
    """Calculates high-level stats for the UI cards."""
    total_fraud = df['isFraud'].sum()
    total_amt_fraud = df[df['isFraud'] == 1]['amount'].sum()
    return total_fraud, total_amt_fraud


def predict_single_transaction(model, le, input_data):
    input_data['balance_error'] = (input_data['oldbalanceOrg'] - input_data['amount']) - input_data['newbalanceOrig']
    
    input_data['type_encoded'] = le.transform([input_data['type']])[0]
    
    features = ['step', 'type_encoded', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'balance_error']
    X_single = pd.DataFrame([input_data])[features]
    
    probability = model.predict_proba(X_single)[0][1]
    return probability