"""
Agent 4: Reporting Agent
Compiles the findings of Detection, Analysis, and Verification into a clean,
executive-ready Fraud Investigation Brief with actionable recommendations.
"""

from agents.llm_client import query_llm

SYSTEM_PROMPT = """You are an Executive Risk Communications Lead for a FinTech platform.
Your objective is to produce a concise, professional Fraud Investigation Brief.
Structure the report cleanly with:
1. INCIDENT OVERVIEW
2. KEY EVIDENCE & INVARIANT FINDINGS
3. MULTI-AGENT AUDIT VERDICT & RISK TIER
4. RECOMMENDED ACTION PLAN
"""

def generate_report(
    detection_result: dict,
    analysis_result: dict,
    verification_result: dict,
    model_name: str = "llama3.1:8b"
) -> dict:
    """
    Executes the Reporting Agent workflow.
    """
    metrics = detection_result["metrics"]
    raw = detection_result["raw_transaction"]
    anomalies = detection_result["anomalies"]
    risk_tier = verification_result["risk_tier"]
    
    user_prompt = f"""Generate an executive Fraud Investigation Brief based on the following pipeline outputs:

[TRANSACTION RECORD]
- Type: {raw.get('type')}
- Amount: ${metrics['amount']:,.2f}
- Starting Balance: ${metrics['old_balance_orig']:,.2f}
- Recorded New Balance: ${metrics['new_balance_orig']:,.2f}
- Expected New Balance: ${metrics['expected_new_orig']:,.2f}
- Discrepancy: ${metrics['orig_balance_error']:,.2f}

[DETECTOR ANOMALIES]
{anomalies}

[ANALYST SUMMARY]
{analysis_result['analysis_text']}

[VERIFICATION AUDIT]
- Risk Tier: {risk_tier}
- Audit Notes: {verification_result['audit_notes']}

Generate a concise, polished markdown brief for the Risk Operations Director.
"""

    llm_response = query_llm(user_prompt, system_prompt=SYSTEM_PROMPT, model=model_name)

    if llm_response:
        return {
            "report_markdown": llm_response,
            "risk_tier": risk_tier,
            "is_llm_generated": True
        }

    # Deterministic fallback structured report
    actions = {
        "HIGH": [
            "Place temporary 24-hour hold on associated recipient account.",
            "Escalate to Level-2 Fraud Investigation Unit for ledger trace.",
            "Review concurrent session logs around the transaction timestamp."
        ],
        "MEDIUM": [
            "Flag transaction for end-of-day batch reconciliation audit.",
            "Send passive multi-factor authentication (MFA) confirmation to account owner.",
            "Monitor origin account for subsequent high-velocity outflows."
        ],
        "LOW": [
            "Approve transaction through standard processing queue.",
            "No manual escalation required."
        ]
    }

    action_list = "\n".join(f"- {act}" for act in actions.get(risk_tier, actions["LOW"]))

    fallback_report = f"""### 🛡️ FinTech Fraud Investigation Brief

**Transaction Profile:** `{raw.get('type')}` | **Amount:** `${metrics['amount']:,.2f}` | **Starting Balance:** `${metrics['old_balance_orig']:,.2f}`

---

#### 1. Invariant Violations & Detection Findings
- **Expected New Balance:** `${metrics['expected_new_orig']:,.2f}` vs **Recorded:** `${metrics['new_balance_orig']:,.2f}`
- **Calculated Invariant Discrepancy:** `${metrics['orig_balance_error']:,.2f}`
- **Flagged Signals:** {len(anomalies)} anomalies identified.

#### 2. Risk Assessment & Verification
- **Assigned Risk Tier:** **{risk_tier}** (Score: {verification_result.get('risk_score_points', 0)}/100)
- **Multi-Agent Consensus:** The evidence indicates an unexplained balance invariant violation. While deliberate fraud/drainage is a viable hypothesis, distributed race conditions and reconciliation lags cannot be ruled out.

#### 3. Recommended Action Plan
{action_list}
"""

    return {
        "report_markdown": fallback_report,
        "risk_tier": risk_tier,
        "is_llm_generated": False
    }
