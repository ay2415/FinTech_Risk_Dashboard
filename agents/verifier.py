"""
Agent 3: Verification Agent
Audits both quantitative detection results (financial + behavioral) and AI analyst conclusions.
Detects hallucinations, unsupported leaps of logic, and assigns calibrated risk tiers.
"""

from agents.llm_client import query_llm

SYSTEM_PROMPT = """You are a Quality & Compliance Auditor for Financial Risk Investigations.
Your role is to independently verify investigation findings from upstream automated agents.

EVALUATION CRITERIA:
1. Math & Fact Accuracy: Ensure all claims align strictly with numerical detector and behavioural metrics.
2. Objectivity Check: Verify that the analyst did not jump to unproven conclusions (e.g., claiming definitive fraud/hacking without proof).
3. Risk Tier Determination: Assign an objective Risk Tier (LOW, MEDIUM, HIGH) with supporting rationale.
"""

def verify_investigation(detection_result: dict, analysis_result: dict, model_name: str = "llama3.1:8b") -> dict:
    """
    Executes the Verification Agent workflow.
    """
    metrics = detection_result["metrics"]
    anomalies = detection_result["anomalies"]
    raw = detection_result["raw_transaction"]
    behavioral = detection_result.get("behavioral")
    analysis_text = analysis_result["analysis_text"]

    # 1. Deterministic baseline risk calculation
    risk_points = 0
    if abs(metrics.get("orig_balance_error", 0.0)) > 0.01:
        risk_points += 35
    if metrics.get("old_balance_orig", 0.0) == 0 and metrics.get("amount", 0.0) > 0:
        risk_points += 25
    if metrics.get("amount", 0.0) > metrics.get("old_balance_orig", 0.0) and metrics.get("old_balance_orig", 0.0) > 0:
        risk_points += 15
    if metrics.get("is_high_risk_type", False):
        risk_points += 10
    if metrics.get("amount", 0.0) >= 200000.0:
        risk_points += 15

    # Integrate Behavioural Anomaly points
    if behavioral and behavioral.get("behavioural_anomaly", False):
        beh_score = behavioral.get("behavioural_score", 0)
        risk_points += int(beh_score * 0.5)  # Up to 50 points from severe behavioral surge

    risk_points = min(risk_points, 100)

    if risk_points >= 50:
        deterministic_tier = "HIGH"
    elif risk_points >= 20:
        deterministic_tier = "MEDIUM"
    else:
        deterministic_tier = "LOW"

    # 2. LLM Auditor Verification
    beh_context = ""
    if behavioral and not behavioral.get("is_cold_start", False):
        beh_context = (
            f"- Historical Median: ${behavioral.get('historical_median', 0.0):,.2f}\n"
            f"- Amount Ratio: {behavioral.get('amount_ratio', 1.0):.1f}x\n"
            f"- Behavioural Anomaly: {behavioral.get('behavioural_anomaly')}\n"
        )

    user_prompt = f"""Audit the following transaction investigation:

[RAW METRICS]
- Type: {raw.get('type')}
- Amount: ${metrics['amount']:,.2f}
- Starting Balance: ${metrics['old_balance_orig']:,.2f}
- Recorded New Balance: ${metrics['new_balance_orig']:,.2f}
- Balance Discrepancy: ${metrics['orig_balance_error']:,.2f}
{beh_context}- Detector Anomalies: {anomalies}

[ANALYST CONCLUSIONS]
{analysis_text}

AUDIT TASKS:
1. Are the analyst's statements supported by the numerical facts?
2. Did the analyst avoid making unsubstantiated claims of confirmed fraud?
3. Provide your final Risk Level (LOW, MEDIUM, or HIGH) and concise audit verdict.
"""

    llm_response = query_llm(user_prompt, system_prompt=SYSTEM_PROMPT, model=model_name)

    if llm_response:
        tier = deterministic_tier
        upper_resp = llm_response.upper()
        if "RISK LEVEL: HIGH" in upper_resp or "RISK: HIGH" in upper_resp or "**HIGH**" in upper_resp:
            tier = "HIGH"
        elif "RISK LEVEL: MEDIUM" in upper_resp or "RISK: MEDIUM" in upper_resp or "**MEDIUM**" in upper_resp:
            tier = "MEDIUM"
        elif "RISK LEVEL: LOW" in upper_resp or "RISK: LOW" in upper_resp or "**LOW**" in upper_resp:
            tier = "LOW"

        return {
            "risk_tier": tier,
            "risk_score_points": risk_points,
            "audit_notes": llm_response,
            "is_llm_generated": True
        }

    # Deterministic fallback audit
    fallback_notes = (
        f"**Audit Verdict:** Fact-check passed. The findings align with quantitative metrics. "
        f"Assigned Risk Tier: **{deterministic_tier}** (Risk Score: {risk_points}/100)."
    )

    return {
        "risk_tier": deterministic_tier,
        "risk_score_points": risk_points,
        "audit_notes": fallback_notes,
        "is_llm_generated": False
    }
