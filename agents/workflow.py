"""
Multi-Agent Fraud Investigation Workflow Orchestrator.
Sequences: Detector (Financial + Behavioural) -> Analyst -> Verifier -> Reporter.
"""

from typing import Optional, Any
from agents.detector import detect_anomalies
from agents.analyst import analyze_transaction
from agents.verifier import verify_investigation
from agents.reporter import generate_report

def run_investigation_pipeline(
    transaction: dict, 
    user_history: Optional[Any] = None,
    model_name: str = "llama3.1:8b"
) -> dict:
    """
    Executes the 4-stage multi-agent fraud investigation.
    
    Args:
        transaction: dict containing transaction fields:
            'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', etc.
        user_history: Optional historical transactions list or profile dict.
        model_name: Name of the LLM model to query.
        
    Returns:
        Dictionary containing the full trace from all 4 agents.
    """
    # Stage 1: Deterministic Detection (Financial Math + User Behaviour)
    detection_result = detect_anomalies(transaction, user_history=user_history)
    
    # Stage 2: Contextual Analysis (LLM)
    analysis_result = analyze_transaction(detection_result, model_name=model_name)
    
    # Stage 3: Independent Verification & Auditing (LLM / Invariant Audit)
    verification_result = verify_investigation(detection_result, analysis_result, model_name=model_name)
    
    # Stage 4: Executive Incident Reporting (LLM)
    report_result = generate_report(detection_result, analysis_result, verification_result, model_name=model_name)
    
    return {
        "transaction": transaction,
        "detection": detection_result,
        "analysis": analysis_result,
        "verification": verification_result,
        "report": report_result,
        "behavioral": detection_result.get("behavioral"),
        "risk_tier": verification_result["risk_tier"]
    }
