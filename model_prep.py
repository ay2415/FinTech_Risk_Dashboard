import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

def prepare_and_train(df):
    le = LabelEncoder()
    df['type_encoded'] = le.fit_transform(df['type'])
    

    features = ['step', 'type_encoded', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'balance_error']
    X = df[features]
    y = df['isFraud']
    
    # 3. Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Initialize XGBoost (Industry standard for FinTech)
    model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
    
    st.write("Training the brain... please wait.")
    model.fit(X_train, y_train)
    
    return model, X_test, y_test