# FinTech Risk Monitor & Multi-Agent Fraud Investigation System

An interactive FinTech dashboard and multi-agent fraud investigation system built with Streamlit, Scikit-Learn, and LLM-powered agentic workflows. The system monitors transactions in real-time, calculates financial invariants, trains predictive ML models, and executes an explainable 4-stage multi-agent investigation pipeline for suspicious transactions.

---

## 🎯 Architecture: 4-Stage Multi-Agent Workflow

```text
Transaction Data
       ↓
[ Agent 1: Detection Agent ]      → Deterministic Python Financial Invariant Checks
       ↓ (Discrepancies & Facts)
[ Agent 2: Analysis Agent ]       → LLM Hypothesis Generation (No Confirmation Bias)
       ↓ (Hypotheses)
[ Agent 3: Verification Agent ]   → Independent Evidence Cross-Examiner & Risk Tiering
       ↓ (Audit & Risk Tier)
[ Agent 4: Reporting Agent ]      → Executive Incident Brief with Actionable Steps
```

| Agent | Responsibility | Implementation Nature |
| :--- | :--- | :--- |
| **Agent 1: Detection Agent** | Evaluates ledger math ($\text{Old} - \text{Amount} = \text{New}$), zero-balance drains, and channel risks. | **100% Deterministic Python** (No LLM for arithmetic). |
| **Agent 2: Analysis Agent** | Formulates multiple plausible root causes (concurrency race condition, settlement lag, exploit). | **LLM Reasoning** with anti-bias prompts. |
| **Agent 3: Verification Agent** | Audits analyst claims against raw metrics, checks for hallucinations, and assigns a risk tier (`LOW`, `MEDIUM`, `HIGH`). | **Dual Audit** (Rule-based score + LLM fact check). |
| **Agent 4: Reporting Agent** | Produces a clean, executive-ready investigation brief with clear remediation actions. | **LLM Structured Synthesis**. |

---

## 💡 Core Engineering Principle: Invariant Anomaly vs. Confirmed Fraud

A core design principle of this system is distinguishing between:
1. **Mathematical Invariant Violation:** $\text{Old Balance} - \text{Amount} \neq \text{New Balance}$
2. **Definitive Fraud / Attack:** An unsubstantiated conclusion.

> *"A mathematical ledger discrepancy is a high-priority investigation signal (which may arise from distributed race conditions, settlement lags, or data corruption), not automatic proof that an account was hacked."*

---

## 📁 Project Structure

```text
FinTech_Risk_Dashboard/
├── app.py                   # Streamlit Frontend & Multi-Agent UI
├── analysis.py              # Statistical metrics & Random Forest classifier
├── data_loader.py           # Cached dataset loading
├── model_prep.py            # XGBoost training utility
├── requirements.txt         # Project dependencies
├── agents/                  # Multi-Agent Investigation Layer
│   ├── __init__.py
│   ├── detector.py          # Agent 1: Deterministic rule engine
│   ├── analyst.py           # Agent 2: Multi-hypothesis reasoning
│   ├── verifier.py          # Agent 3: Independent evidence auditor
│   ├── reporter.py          # Agent 4: Executive report generator
│   ├── workflow.py          # Orchestration pipeline
│   └── llm_client.py        # Lightweight LLM caller (Ollama / OpenAI / Fallback)
└── data/
    └── paysim.csv           # PaySim synthetic transaction dataset
```

---

## 🚀 Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### 3. (Optional) Run with Local Ollama
Ensure Ollama is running with `llama3.1:8b` or `mistral`:
```bash
ollama run llama3.1:8b
```
*(Note: If Ollama is offline, the system automatically uses deterministic fallbacks so the app never fails.)*

---

## 📄 License
[MIT License](LICENSE)
