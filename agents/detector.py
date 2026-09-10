"""
Agent 1: Detection Agent
Deterministic Python-only rule engine for financial invariants AND user behavioural anomalies.
Handles both Outflows (TRANSFER, CASH_OUT, PAYMENT, DEBIT) and Inflows (CASH_IN).
"""

from typing import Optional, Dict, Any, List
from behavioral import calculate_user_history, evaluate_behavioral_anomaly

def detect_anomalies(
    transaction: dict, 
    user_history: Optional[Any] = None
) -> dict:
    """
    Evaluates financial, balance, and user behavioural invariants on a single transaction.
    
    Args:
        transaction: dict containing:
            - 'type': str ('CASH_IN', 'TRANSFER', 'CASH_OUT', 'PAYMENT', 'DEBIT')
            - 'amount': float
            - 'oldbalanceOrg': float
            - 'newbalanceOrig': float
            - (optional) 'nameOrig' / 'user_id': str
            - (optional) 'step': int
            - (optional) 'oldbalanceDest': float
            - (optional) 'newbalanceDest': float
            
    Returns:
        Structured dictionary with detection results.
    """
    tx_type = str(transaction.get('type', 'UNKNOWN')).upper()
    amount = float(transaction.get('amount', 0.0))
    old_bal_orig = float(transaction.get('oldbalanceOrg', 0.0))
    new_bal_orig = float(transaction.get('newbalanceOrig', 0.0))
    user_id = transaction.get('nameOrig', transaction.get('user_id', 'UNKNOWN_USER'))
    step = transaction.get('step', 1)
    
    anomalies = []
    
    # ---------------- 1. FINANCIAL & BALANCE INVARIANTS ----------------
    
    # Invariant A: Balance Math Invariant
    # For CASH_IN (Money Received / Inflow): Expected New = Old Balance + Amount
    # For Outflows (TRANSFER, CASH_OUT, PAYMENT, DEBIT): Expected New = Old Balance - Amount
    if tx_type == "CASH_IN":
        expected_new_orig = old_bal_orig + amount
    else:
        expected_new_orig = old_bal_orig - amount
        
    orig_balance_error = round(expected_new_orig - new_bal_orig, 2)
    
    if abs(orig_balance_error) > 0.01:
        anomalies.append(
            f"Origin balance invariant violated ({'Inflow' if tx_type == 'CASH_IN' else 'Outflow'}): "
            f"Expected ${expected_new_orig:,.2f}, got ${new_bal_orig:,.2f} (Discrepancy: ${orig_balance_error:,.2f})"
        )
        
    # Invariant B: Overdraft / Draining beyond available balance (Outflows only)
    if tx_type != "CASH_IN" and amount > old_bal_orig and old_bal_orig > 0:
        anomalies.append(
            f"Transaction amount (${amount:,.2f}) exceeds available origin balance (${old_bal_orig:,.2f})"
        )
        
    # Invariant C: Zero Starting Balance Outflow (Outflows only)
    if tx_type != "CASH_IN" and old_bal_orig == 0 and amount > 0:
        anomalies.append(
            f"Zero starting balance anomaly: Outflow of ${amount:,.2f} initiated from account with $0.00 balance"
        )
        
    # Invariant D: High-Risk FinTech Channels (Liquidation types)
    high_risk_types = ["TRANSFER", "CASH_OUT"]
    is_high_risk_type = tx_type in high_risk_types
    if is_high_risk_type:
        anomalies.append(
            f"High-risk transaction category: '{tx_type}' is historically correlated with liquidation channels"
        )
        
    # Invariant E: High Dollar Value Threshold
    if amount >= 200000.0:
        anomalies.append(
            f"High-value threshold exceeded: Amount (${amount:,.2f}) exceeds $200,000 monitoring limit"
        )

    # Invariant F: Destination Invariant Check (if destination balances provided)
    dest_balance_error = None
    if 'oldbalanceDest' in transaction and 'newbalanceDest' in transaction:
        old_bal_dest = float(transaction.get('oldbalanceDest', 0.0))
        new_bal_dest = float(transaction.get('newbalanceDest', 0.0))
        # Destination balance should increase by amount
        expected_new_dest = old_bal_dest + amount
        dest_balance_error = round(new_bal_dest - expected_new_dest, 2)
        if abs(dest_balance_error) > 0.01:
            anomalies.append(
                f"Destination balance invariant violated: Expected ${expected_new_dest:,.2f}, "
                f"got ${new_bal_dest:,.2f} (Discrepancy: ${dest_balance_error:,.2f})"
            )

    # ---------------- 2. USER BEHAVIOURAL INVARIANTS ----------------
    behavioral_summary = None
    if user_history is not None:
        if isinstance(user_history, list):
            history_stats = calculate_user_history(user_history, current_step=step)
        elif isinstance(user_history, dict) and "mean" in user_history:
            history_stats = user_history
        else:
            history_stats = calculate_user_history([])

        behavioral_summary = evaluate_behavioral_anomaly(transaction, history_stats)
        
        # Add behavioral anomaly reasons to the general anomalies list
        if behavioral_summary.get("behavioural_anomaly", False):
            for r in behavioral_summary.get("reasons", []):
                anomalies.append(f"Behavioural Anomaly: {r}")

    is_suspicious = len(anomalies) > 0

    return {
        "user_id": user_id,
        "is_suspicious": is_suspicious,
        "anomalies_count": len(anomalies),
        "anomalies": anomalies,
        "metrics": {
            "type": tx_type,
            "direction": "INFLOW (Deposit/Received)" if tx_type == "CASH_IN" else "OUTFLOW (Sent/Withdrawn)",
            "amount": amount,
            "old_balance_orig": old_bal_orig,
            "new_balance_orig": new_bal_orig,
            "expected_new_orig": expected_new_orig,
            "orig_balance_error": orig_balance_error,
            "is_high_risk_type": is_high_risk_type,
            "dest_balance_error": dest_balance_error
        },
        "behavioral": behavioral_summary,
        "raw_transaction": transaction
    }
