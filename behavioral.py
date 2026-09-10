"""
User Behavioural Anomaly Detection Module.
Calculates historical baselines (mean, median, max, std, rolling windows)
strictly prior to the current transaction to prevent data leakage.
100% deterministic, explainable Python logic without black-box ML dependencies.
"""

import math
from typing import List, Dict, Any, Optional

def calculate_user_history(
    past_transactions: List[Dict[str, Any]], 
    current_step: Optional[int] = None
) -> Dict[str, Any]:
    """
    Computes historical statistics for a specific user from past transactions.
    
    Args:
        past_transactions: List of historical transaction dicts strictly BEFORE the current transaction.
            Each dict should have 'amount', 'type', and optionally 'step'.
        current_step: Optional simulation time step (1 step = 1 hour).
            Used for rolling time-window calculations (7d, 30d, 90d).
            
    Returns:
        Dictionary containing historical baseline metrics.
    """
    if not past_transactions:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "max": 0.0,
            "min": 0.0,
            "std": 0.0,
            "total_volume": 0.0,
            "rolling_7d_mean": None,
            "rolling_30d_mean": None,
            "rolling_90d_mean": None,
            "type_counts": {},
            "amounts": []
        }

    amounts = [float(tx["amount"]) for tx in past_transactions if "amount" in tx]
    count = len(amounts)
    
    if count == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "max": 0.0,
            "min": 0.0,
            "std": 0.0,
            "total_volume": 0.0,
            "rolling_7d_mean": None,
            "rolling_30d_mean": None,
            "rolling_90d_mean": None,
            "type_counts": {},
            "amounts": []
        }

    # 1. Basic Summary Statistics
    total_volume = sum(amounts)
    mean_val = total_volume / count
    sorted_amounts = sorted(amounts)
    
    # Median calculation (robust to outliers)
    if count % 2 == 1:
        median_val = sorted_amounts[count // 2]
    else:
        median_val = (sorted_amounts[(count // 2) - 1] + sorted_amounts[count // 2]) / 2.0

    max_val = sorted_amounts[-1]
    min_val = sorted_amounts[0]

    # Sample standard deviation (requires >= 2 samples)
    if count >= 2:
        variance = sum((x - mean_val) ** 2 for x in amounts) / (count - 1)
        std_val = math.sqrt(variance)
    else:
        std_val = 0.0

    # 2. Transaction Type Frequency
    type_counts = {}
    for tx in past_transactions:
        t = tx.get("type", "UNKNOWN")
        type_counts[t] = type_counts.get(t, 0) + 1

    # 3. Rolling Time Windows (where step is available; 1 step = 1 hour)
    # 7 days = 168 hours, 30 days = 720 hours, 90 days = 2160 hours
    rolling_7d_mean = None
    rolling_30d_mean = None
    rolling_90d_mean = None

    if current_step is not None:
        amt_7d = [float(tx["amount"]) for tx in past_transactions if "step" in tx and current_step - 168 <= tx["step"] < current_step]
        amt_30d = [float(tx["amount"]) for tx in past_transactions if "step" in tx and current_step - 720 <= tx["step"] < current_step]
        amt_90d = [float(tx["amount"]) for tx in past_transactions if "step" in tx and current_step - 2160 <= tx["step"] < current_step]

        if amt_7d:
            rolling_7d_mean = sum(amt_7d) / len(amt_7d)
        if amt_30d:
            rolling_30d_mean = sum(amt_30d) / len(amt_30d)
        if amt_90d:
            rolling_90d_mean = sum(amt_90d) / len(amt_90d)

    return {
        "count": count,
        "mean": round(mean_val, 2),
        "median": round(median_val, 2),
        "max": round(max_val, 2),
        "min": round(min_val, 2),
        "std": round(std_val, 2),
        "total_volume": round(total_volume, 2),
        "rolling_7d_mean": round(rolling_7d_mean, 2) if rolling_7d_mean is not None else None,
        "rolling_30d_mean": round(rolling_30d_mean, 2) if rolling_30d_mean is not None else None,
        "rolling_90d_mean": round(rolling_90d_mean, 2) if rolling_90d_mean is not None else None,
        "type_counts": type_counts,
        "amounts": amounts
    }


def evaluate_behavioral_anomaly(
    current_transaction: Dict[str, Any],
    history: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compares the current transaction against the user's historical baseline.
    
    Returns structured anomaly assessment.
    """
    user_id = current_transaction.get("nameOrig", current_transaction.get("user_id", "UNKNOWN_USER"))
    current_amt = float(current_transaction.get("amount", 0.0))
    current_type = current_transaction.get("type", "UNKNOWN")

    count = history.get("count", 0)
    mean_val = history.get("mean", 0.0)
    median_val = history.get("median", 0.0)
    max_val = history.get("max", 0.0)
    std_val = history.get("std", 0.0)
    type_counts = history.get("type_counts", {})

    reasons = []
    is_anomaly = False
    anomaly_score = 0  # 0 to 100

    # 1. Cold Start Case: No historical transactions
    if count == 0:
        return {
            "user_id": user_id,
            "amount": current_amt,
            "historical_count": 0,
            "historical_average": 0.0,
            "historical_median": 0.0,
            "historical_max": 0.0,
            "historical_std": 0.0,
            "amount_ratio": 1.0,
            "z_score": None,
            "rolling_7d_mean": None,
            "rolling_30d_mean": None,
            "rolling_90d_mean": None,
            "is_cold_start": True,
            "behavioural_anomaly": False,
            "behavioural_score": 0,
            "risk_level": "LOW",
            "reasons": ["New account with no prior transaction history (Cold Start)."]
        }

    # 2. Ratio Analysis (Prefer median to avoid distortion from past outliers)
    baseline_ref = median_val if median_val > 0 else (mean_val if mean_val > 0 else 1.0)
    amount_ratio = current_amt / baseline_ref

    # 3. Z-Score / Standard Deviation Check (where std > 0 and count >= 3)
    z_score = None
    if count >= 3 and std_val > 0:
        z_score = (current_amt - mean_val) / std_val

    # 4. Behavioral Anomaly Rule Evaluation
    
    # Severe Spike: >= 10x median and current amount is substantial (> $500)
    if amount_ratio >= 10.0 and current_amt > 500:
        is_anomaly = True
        anomaly_score += 50
        reasons.append(
            f"Extreme amount surge: Current amount (${current_amt:,.2f}) is {amount_ratio:.1f}x higher "
            f"than user's historical median (${median_val:,.2f})."
        )
    # Moderate Spike: 4x to 10x median and amount > $300
    elif amount_ratio >= 4.0 and current_amt > 300:
        is_anomaly = True
        anomaly_score += 30
        reasons.append(
            f"Moderate amount surge: Current amount (${current_amt:,.2f}) is {amount_ratio:.1f}x higher "
            f"than user's historical median (${median_val:,.2f})."
        )

    # Exceeding Historical Maximum by a wide margin
    if current_amt > 2.5 * max_val and current_amt > 500:
        is_anomaly = True
        anomaly_score += 30
        reasons.append(
            f"New spending record: Amount (${current_amt:,.2f}) exceeds historical maximum "
            f"(${max_val:,.2f}) by {(current_amt / max_val if max_val > 0 else 1.0):.1f}x."
        )

    # Statistical 3-Sigma Anomaly (Z-score >= 3.0)
    if z_score is not None and z_score >= 3.0:
        is_anomaly = True
        anomaly_score += 25
        reasons.append(
            f"Statistical deviation: Z-score is {z_score:.2f} (>3.0 std deviations above user's mean)."
        )

    # Unusual Transaction Type for this user
    if current_type in ["TRANSFER", "CASH_OUT"] and current_type not in type_counts and current_amt > 1000:
        anomaly_score += 20
        reasons.append(
            f"Unprecedented channel usage: User has never previously executed a '{current_type}' operation."
        )

    # Cap anomaly score at 100
    anomaly_score = min(anomaly_score, 100)

    # Risk level classification based on behavioural indicators alone
    if anomaly_score >= 60:
        risk_level = "HIGH"
    elif anomaly_score >= 25:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "user_id": user_id,
        "amount": current_amt,
        "historical_count": count,
        "historical_average": round(mean_val, 2),
        "historical_median": round(median_val, 2),
        "historical_max": round(max_val, 2),
        "historical_std": round(std_val, 2),
        "amount_ratio": round(amount_ratio, 2),
        "z_score": round(z_score, 2) if z_score is not None else None,
        "rolling_7d_mean": history.get("rolling_7d_mean"),
        "rolling_30d_mean": history.get("rolling_30d_mean"),
        "rolling_90d_mean": history.get("rolling_90d_mean"),
        "is_cold_start": False,
        "behavioural_anomaly": is_anomaly,
        "behavioural_score": anomaly_score,
        "risk_level": risk_level,
        "reasons": reasons if reasons else ["Transaction amount and type align with user's historical profile."]
    }
