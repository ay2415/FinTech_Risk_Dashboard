"""
Agent 2: Analysis Agent
Uses LLM reasoning to evaluate detector evidence, generating contextual 
hypotheses while strictly avoiding confirmation bias.
"""

from agents.llm_client import query_llm

SYSTEM_PROMPT = """You are a Senior FinTech Risk Analyst specializing in digital payment ledgers and anomaly triage.
Your task is to analyze evidence from automated anomaly detectors and formulate nuanced, objective investigation hypotheses.

CRITICAL GUIDELINES:
1. Examine all mathematical discrepancies and transaction attributes provided.
2. Explain clearly why the transaction violates expected financial invariants.
3. Provide MULTIPLE plausible hypotheses (e.g., distributed race conditions, asynchronous settlement lag, merchant terminal error, simulation artifact, or unauthorized account takeover).
4. NEVER state with certainty that 'a hacker' or 'fraudster' caused the issue unless direct forensic proof exists. Treat mathematical errors as invariant violations requiring investigation.
5. Be concise, professional, and structured.
"""

def analyze_transaction(detection_result: dict, model_name: str = "llama3.1:8b") -> dict:
    """
    Executes the Analysis Agent workflow.
    """
    if not detection_result.get("is_suspicious", False):
        return {
            "analysis_text": (
                "The transaction conforms to standard financial invariants. "
                "The origin balance deduction matches the transaction amount exactly, "
                "and no abnormal channel indicators were triggered."
            ),
            "hypotheses": ["Legitimate transaction with intact ledger math."],
            "is_llm_generated": False
        }

    raw = detection_result["raw_transaction"]
    metrics = detection_result["metrics"]
    anomalies = detection_result["anomalies"]

    user_prompt = f"""Review the following transaction flagged by the Detection Agent:

[TRANSACTION DETAILS]
- Type: {raw.get('type')}
- Amount: ${metrics['amount']:,.2f}
- Old Balance (Origin): ${metrics['old_balance_orig']:,.2f}
- New Balance (Origin): ${metrics['new_balance_orig']:,.2f}
- Expected New Balance: ${metrics['expected_new_orig']:,.2f}
- Mathematical Balance Discrepancy: ${metrics['orig_balance_error']:,.2f}

[DETECTOR ANOMALIES FLAGGED]
{chr(10).join(f"- {a}" for a in anomalies)}

Provide your professional analysis following this structure:
1. SUMMARY OF ANOMALIES: Explain the mathematical and behavioral breakdown.
2. POTENTIAL ROOT CAUSES: Present at least 3 distinct hypotheses (technical ledger bug/race condition, settlement delay, or fraudulent exploitation).
3. INVESTIGATION FOCUS: What specific log or account records should be inspected next?
"""

    llm_response = query_llm(user_prompt, system_prompt=SYSTEM_PROMPT, model=model_name)

    if llm_response:
        return {
            "analysis_text": llm_response,
            "is_llm_generated": True
        }

    # Deterministic fallback if LLM service is offline
    fallback_text = (
        f"### 1. Summary of Anomalies\n"
        f"The transaction of ${metrics['amount']:,.2f} ({raw.get('type')}) exhibits a balance discrepancy of "
        f"${metrics['orig_balance_error']:,.2f}. The starting balance was ${metrics['old_balance_orig']:,.2f}, "
        f"resulting in an unexpected final ledger balance of ${metrics['new_balance_orig']:,.2f}.\n\n"
        f"### 2. Potential Root Causes\n"
        f"- **Hypothesis A (Distributed Ledger Race Condition):** Multiple concurrent requests executed before database locks settled.\n"
        f"- **Hypothesis B (Batch Settlement / Sync Delay):** Asynchronous transaction clearing causing temporary out-of-order balance states.\n"
        f"- **Hypothesis C (Unauthorized Account Drainage / Exploit):** Deliberate zero-balance extraction or compromised agent terminal.\n\n"
        f"### 3. Investigation Focus\n"
        f"Audit server timestamp logs, active database locks, and destination account linkage."
    )

    return {
        "analysis_text": fallback_text,
        "is_llm_generated": False
    }
