"""
Comprehensive Test Suite for User Behavioural Anomaly Detection & Inflow/Outflow Math.
Covers: normal, moderate, extreme, cold-start, sparse history,
high-net-worth baselines, data leakage, and CASH_IN deposits.
"""

from behavioral import calculate_user_history, evaluate_behavioral_anomaly
from agents.detector import detect_anomalies
from agents.workflow import run_investigation_pipeline

def test_case_1_normal_transaction():
    """Normal transaction: within historical spending norms."""
    history_txns = [
        {"amount": 20.0, "type": "PAYMENT", "step": 10},
        {"amount": 30.0, "type": "PAYMENT", "step": 15},
        {"amount": 15.0, "type": "PAYMENT", "step": 20},
        {"amount": 40.0, "type": "PAYMENT", "step": 25},
        {"amount": 25.0, "type": "PAYMENT", "step": 30},
        {"amount": 35.0, "type": "PAYMENT", "step": 35},
    ]
    current_txn = {
        "user_id": "USER_NORMAL",
        "amount": 30.0,
        "type": "PAYMENT",
        "oldbalanceOrg": 500.0,
        "newbalanceOrig": 470.0,
        "step": 40
    }
    hist_stats = calculate_user_history(history_txns, current_step=40)
    res = evaluate_behavioral_anomaly(current_txn, hist_stats)

    assert res["historical_count"] == 6
    assert abs(res["historical_average"] - 27.5) < 0.1
    assert res["historical_median"] == 27.5
    assert res["historical_max"] == 40.0
    assert not res["behavioural_anomaly"], "Normal txn should not be flagged as behavioural anomaly"
    assert res["risk_level"] == "LOW"
    print("[PASS] Test Case 1: Normal transaction correctly evaluated as LOW risk.")

def test_case_2_moderately_unusual_transaction():
    """Moderately unusual transaction ($350 vs $28 median -> ~12x surge)."""
    history_txns = [
        {"amount": 20.0, "type": "PAYMENT", "step": 10},
        {"amount": 30.0, "type": "PAYMENT", "step": 15},
        {"amount": 15.0, "type": "PAYMENT", "step": 20},
        {"amount": 40.0, "type": "PAYMENT", "step": 25},
        {"amount": 25.0, "type": "PAYMENT", "step": 30},
        {"amount": 35.0, "type": "PAYMENT", "step": 35},
    ]
    current_txn = {
        "user_id": "USER_MODERATE",
        "amount": 350.0,
        "type": "PAYMENT",
        "oldbalanceOrg": 1000.0,
        "newbalanceOrig": 650.0,
        "step": 40
    }
    hist_stats = calculate_user_history(history_txns, current_step=40)
    res = evaluate_behavioral_anomaly(current_txn, hist_stats)

    assert res["behavioural_anomaly"] is True
    assert res["amount_ratio"] > 10.0
    assert res["risk_level"] in ["MEDIUM", "HIGH"]
    print(f"[PASS] Test Case 2: Moderate surge detected ({res['amount_ratio']}x ratio, Risk: {res['risk_level']}).")

def test_case_3_extremely_unusual_transaction():
    """Extremely unusual transaction ($30,000 vs $28 median -> >1000x surge)."""
    history_txns = [
        {"amount": 20.0, "type": "PAYMENT", "step": 10},
        {"amount": 30.0, "type": "PAYMENT", "step": 15},
        {"amount": 15.0, "type": "PAYMENT", "step": 20},
        {"amount": 40.0, "type": "PAYMENT", "step": 25},
        {"amount": 25.0, "type": "PAYMENT", "step": 30},
        {"amount": 35.0, "type": "PAYMENT", "step": 35},
    ]
    current_txn = {
        "user_id": "USER_EXTREME",
        "amount": 30000.0,
        "type": "CASH_OUT",
        "oldbalanceOrg": 30000.0,
        "newbalanceOrig": 0.0,
        "step": 40
    }
    hist_stats = calculate_user_history(history_txns, current_step=40)
    res = evaluate_behavioral_anomaly(current_txn, hist_stats)

    assert res["behavioural_anomaly"] is True
    assert res["amount_ratio"] > 1000.0
    assert res["risk_level"] == "HIGH"
    print(f"[PASS] Test Case 3: Extreme surge detected (Ratio: {res['amount_ratio']}x, Risk: {res['risk_level']}).")

def test_case_4_sparse_history():
    """User with very little historical data (1-2 transactions)."""
    history_txns = [
        {"amount": 50.0, "type": "PAYMENT", "step": 10}
    ]
    current_txn = {
        "user_id": "USER_SPARSE",
        "amount": 60.0,
        "type": "PAYMENT",
        "oldbalanceOrg": 200.0,
        "newbalanceOrig": 140.0,
        "step": 15
    }
    hist_stats = calculate_user_history(history_txns, current_step=15)
    res = evaluate_behavioral_anomaly(current_txn, hist_stats)

    assert res["historical_count"] == 1
    assert res["historical_std"] == 0.0  # Cannot compute sample std on n=1
    assert not res["behavioural_anomaly"]
    assert res["risk_level"] == "LOW"
    print("[PASS] Test Case 4: Sparse history (n=1) handled safely without math errors.")

def test_case_5_cold_start_no_history():
    """User with no historical data (brand new account)."""
    history_txns = []
    current_txn = {
        "user_id": "USER_NEW",
        "amount": 100.0,
        "type": "PAYMENT",
        "oldbalanceOrg": 500.0,
        "newbalanceOrig": 400.0,
        "step": 1
    }
    hist_stats = calculate_user_history(history_txns, current_step=1)
    res = evaluate_behavioral_anomaly(current_txn, hist_stats)

    assert res["is_cold_start"] is True
    assert res["historical_count"] == 0
    assert not res["behavioural_anomaly"]
    assert res["risk_level"] == "LOW"
    print("[PASS] Test Case 5: Cold start handled gracefully.")

def test_case_6_high_net_worth_user():
    """User whose historical transactions already contain large values."""
    history_txns = [
        {"amount": 40000.0, "type": "TRANSFER", "step": 10},
        {"amount": 60000.0, "type": "TRANSFER", "step": 20},
        {"amount": 50000.0, "type": "TRANSFER", "step": 30},
    ]
    current_txn = {
        "user_id": "USER_HNW",
        "amount": 45000.0,
        "type": "TRANSFER",
        "oldbalanceOrg": 100000.0,
        "newbalanceOrig": 55000.0,
        "step": 40
    }
    hist_stats = calculate_user_history(history_txns, current_step=40)
    res = evaluate_behavioral_anomaly(current_txn, hist_stats)

    assert res["historical_median"] == 50000.0
    assert abs(res["amount_ratio"] - 0.9) < 0.1
    assert not res["behavioural_anomaly"], "$45,000 for a user with $50k median is NOT anomalous"
    print("[PASS] Test Case 6: High-value account not falsely flagged.")

def test_case_7_data_leakage_check():
    """Verify that the current transaction is strictly excluded from historical statistics."""
    past_txns = [
        {"amount": 10.0, "type": "PAYMENT", "step": 1},
        {"amount": 20.0, "type": "PAYMENT", "step": 2},
    ]
    current_txn = {
        "amount": 10000.0,  # Current huge txn
        "type": "CASH_OUT",
        "oldbalanceOrg": 10000.0,
        "newbalanceOrig": 0.0,
        "step": 3
    }
    # Calculate history using strictly past_txns
    hist_stats = calculate_user_history(past_txns, current_step=3)
    assert 10000.0 not in hist_stats["amounts"]
    assert hist_stats["max"] == 20.0, "Current txn must not leak into historical max!"
    print("[PASS] Test Case 7: No data leakage confirmed.")

def test_case_8_end_to_end_pipeline():
    """Verify full multi-agent pipeline with behavioural context."""
    history_txns = [
        {"amount": 20.0, "type": "PAYMENT", "step": 10},
        {"amount": 30.0, "type": "PAYMENT", "step": 15},
    ]
    current_txn = {
        "user_id": "USER_E2E",
        "type": "CASH_OUT",
        "amount": 30000.0,
        "oldbalanceOrg": 30000.0,
        "newbalanceOrig": 0.0,
        "step": 20
    }
    result = run_investigation_pipeline(current_txn, user_history=history_txns)
    assert result["risk_tier"] == "HIGH"
    assert result["behavioral"]["behavioural_anomaly"] is True
    assert "behavioral" in result
    print(f"[PASS] Test Case 8: Full Multi-Agent pipeline completed with Risk Tier {result['risk_tier']}.")

def test_case_9_cash_in_inflow():
    """Verify CASH_IN (Money Received/Deposited) uses addition math: Old Balance + Amount == New Balance."""
    deposit_txn = {
        "user_id": "USER_RECEIVER",
        "type": "CASH_IN",
        "amount": 2500.0,
        "oldbalanceOrg": 1000.0,
        "newbalanceOrig": 3500.0,  # 1000 + 2500 = 3500
        "step": 50
    }
    det = detect_anomalies(deposit_txn)
    assert det["metrics"]["orig_balance_error"] == 0.0
    assert not det["is_suspicious"], "Valid deposit should not have balance errors"
    assert "INFLOW" in det["metrics"]["direction"]
    print("[PASS] Test Case 9: CASH_IN (money received) addition math verified successfully.")

if __name__ == "__main__":
    print("--- RUNNING COMPREHENSIVE SUITE ---")
    test_case_1_normal_transaction()
    test_case_2_moderately_unusual_transaction()
    test_case_3_extremely_unusual_transaction()
    test_case_4_sparse_history()
    test_case_5_cold_start_no_history()
    test_case_6_high_net_worth_user()
    test_case_7_data_leakage_check()
    test_case_8_end_to_end_pipeline()
    test_case_9_cash_in_inflow()
    print("ALL 9 TEST CASES PASSED SUCCESSFULLY!")
