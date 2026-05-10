# FinTech Risk & Fraud Dashboard

A real-time, interactive dashboard built with Streamlit to monitor and detect fraudulent financial transactions. This application utilizes Machine Learning (Random Forest) and data visualization techniques to identify suspicious patterns in synthetic financial datasets like PaySim.

## Features

- **Real-time Monitoring**: Load and visualize transaction data to monitor risk in real-time.
- **Global Filters**: Filter transactions by type (e.g., TRANSFER, CASH_OUT, PAYMENT, DEBIT) to analyze specific segments.
- **Balance Error Analysis**: Detects "Impossible Transactions" where the math doesn't add up `(Old Balance - Amount != New Balance)` and visually highlights discrepancies.
- **Machine Learning Intelligence**: 
  - Trains a Random Forest Classifier on the fly to detect fraud.
  - Generates an interactive Model Accuracy Report and Feature Importance chart to provide transparency on what indicators drive the model's predictions.
- **Interactive Fraud Simulator**: Allows users to input hypothetical transaction details and receive a real-time risk score and probability of fraud.

## Tech Stack

- **Frontend/UI**: [Streamlit](https://streamlit.io/)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/)
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/) (Random Forest Classifier) & [XGBoost](https://xgboost.readthedocs.io/) (available in `model_prep.py`)
- **Data Visualization**: [Plotly Express](https://plotly.com/python/plotly-express/)

## Project Structure

```text
FinTech_Risk_Dashboard/
├── app.py               # Main Streamlit application
├── analysis.py          # Core logic for risk metrics, model preparation, and prediction
├── data_loader.py       # Helper script for optimized data loading
├── model_prep.py        # Alternative XGBoost model training script
├── data/
│   └── paysim.csv       # PaySim synthetic transaction dataset (Not included in repo by default)
└── Screenshots/         # Project screenshots and documentation
```

## Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd FinTech_Risk_Dashboard
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   You can install the required libraries by running:
   ```bash
   pip install streamlit pandas plotly scikit-learn xgboost
   ```

4. **Add the Dataset**:
   Download the [PaySim dataset from Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) and place the `paysim.csv` file inside the `data/` directory.

## Usage

Start the Streamlit server to view the dashboard:

```bash
streamlit run app.py
```

The application will launch in your default web browser (usually at `http://localhost:8501`). Once running, you can:
1. View global metrics and the balance error scatter plot.
2. Click **"Run AI Training & Analysis"** to train the risk model and view performance metrics.
3. Use the **"Interactive Fraud Simulator"** at the bottom to test hypothetical transactions.

## Dataset Reference

This project is built around the **PaySim** dataset, which simulates mobile money transactions based on a sample of real transactions extracted from one month of financial logs from a mobile money service implemented in an African country.


