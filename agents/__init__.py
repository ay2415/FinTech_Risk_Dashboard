"""
Multi-Agent Fraud Investigation System.

Pipeline Sequence:
1. Detection Agent (agents/detector.py): Pure deterministic Python invariant evaluator.
2. Analysis Agent (agents/analyst.py): LLM-driven hypothesis generator (anti-bias constrained).
3. Verification Agent (agents/verifier.py): Independent fact auditor & composite risk tiering.
4. Reporting Agent (agents/reporter.py): Executive incident brief & remediation planner.
"""

from agents.detector import detect_anomalies, compute_expected_balance
from agents.analyst import analyze_transaction
from agents.verifier import verify_investigation, calculate_composite_risk_score
from agents.reporter import generate_report
from agents.workflow import run_investigation_pipeline

__all__ = [
    "detect_anomalies",
    "compute_expected_balance",
    "analyze_transaction",
    "verify_investigation",
    "calculate_composite_risk_score",
    "generate_report",
    "run_investigation_pipeline",
]
