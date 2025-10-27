# Setup & Runbook — Loan Navigator

## 1. Prerequisites
- Python 3.11+
- FastAPI, Uvicorn, LangGraph, Streamlit, Plotly, Pandas
- Optional: MLflow for experiment logging

## 2. Backend — FastAPI
```bash
python -m uvicorn app_main:app --reload --port 8000
```
- Environment variables (optional):
  - `MLFLOW_TRACKING_URI` — your tracking server or `file:///...`
  - `MLFLOW_EXPERIMENT_NAME` — default `LoanNavigator`

## 3. Frontend — Streamlit
```bash
streamlit run streamlit_app_updated.py
```
- Configure backend in the sidebar: `http://127.0.0.1:8000`
- Tabs: Chat, What‑If Simulator, Policy Q&A, Loans, Top‑Up

## 4. MLflow UI (local file store)
**PowerShell**
```powershell
$MLRUNS_ABS = "$((Get-Location).Path)\mlruns" -replace '\','/'
mlflow ui --backend-store-uri "file:///$MLRUNS_ABS" --host 0.0.0.0 --port 5000
```
**CMD**
```bat
set "MLRUNS_ABS=%CD%\mlruns" & set "MLRUNS_ABS=%MLRUNS_ABS:\=/%" & mlflow ui --backend-store-uri "file:///%MLRUNS_ABS%" --host 0.0.0.0 --port 5000
```

## 5. Docker (optional)
Build:
```bash
docker build -t mlflow-ui:latest .
```
Run:
```bash
docker run --name mlflow_ui --rm -p 5000:5000 -v "$(pwd)/mlruns:/app/mlruns" mlflow-ui:latest
```

## 6. Health Checks
- `POST /chat` with `{"query": "How many EMIs left for Loan ID 2001?"}`
- `POST /whatif` with `{"loan_id": "2001", "prepay_amt": 10000, "prepay_month": 6, "mode": "reduce_tenure"}`

## 7. Troubleshooting
- If MLflow isn't installed, logging is skipped silently.
- Ensure the backend port (8000) isn't blocked.
- For artifact visualization, the Streamlit app will attempt local amortization charts when loan details are available.

*** Please check out following link http://loannavigator.hre7dgh3andgfmbe.centralindia.azurecontainer.io:8080/  or http://4.224.128.187:8080/
