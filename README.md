# ModelGuard AI — ML Observability Lab

**Release 1.0.0 | Local-first ML monitoring and model-risk decision support**

ModelGuard AI compares a reference dataset with a newer batch to help a practitioner spot data-quality problems, feature distribution changes, unusual rows, and potential model-health concerns before making a production decision.

> **Safety boundary:** This is a local portfolio and decision-support tool. It does not connect to, retrain, roll back, or modify a production model. Metrics and thresholds are signals for human review, not guarantees.

## Highlights

- **Data-quality checks:** row-count changes, duplicate rows, missing-value rates, and schema changes.
- **Feature drift:** Population Stability Index (PSI), two-sample Kolmogorov–Smirnov (KS) statistic, direction, and severity.
- **Model-health signals:** AUC, F1, accuracy, Brier score, and reference-trained Random Forest / Logistic Regression comparison when compatible binary labels are available.
- **Anomaly detection:** Isolation Forest fitted on the reference population and applied to the current batch.
- **Interpretability aids:** ranked drift signals, feature importance, segment summaries, and monitoring coverage.
- **Controlled simulations:** missing-data, drift-investigation, retraining-gate, rollback-gate, quarantine, and recovery scenarios. These are simulations only.
- **Shareable outputs:** downloadable HTML and JSON reports.
- **Safer local operation:** loopback-only launcher, bounded CSV uploads (10 MiB per file), validated CSV headers, and generic server-error responses.

## Windows quick start

1. Install **Python 3.11 or newer** and enable **Add Python to PATH**.
2. Extract this ZIP fully to a normal folder (do not run from inside the ZIP preview).
3. Open the extracted `ModelGuard-AI-ML-Observability-Lab` folder.
4. Double-click `run_modelguard.bat`.
5. Wait for the launcher to print **ModelGuard is READY** and open the local UI. Keep the launcher window open while using the app.

The launcher creates a local `.venv`, installs the pinned dependencies, chooses a free loopback port from 8000–8010, waits for the health endpoint, and opens the browser. Server logs are written to `modelguard_server.out.log` and `modelguard_server.err.log`.

If the launcher is blocked, open PowerShell in the extracted folder and run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\start_modelguard.ps1
```

## Use the dashboard

### Demo mode

Click **Load demo** to load the included reproducible reference/current example. It intentionally includes data shifts and quality issues so the monitoring panels have something to show.

### Analyze your own data

1. Choose a baseline / known-good **Reference CSV**.
2. Choose a later **Current CSV**.
3. Click **Analyze datasets**.
4. Review readiness, quality checks, drift, model metrics, recommendations, and simulations.
5. Use **Refresh** to re-run analysis against the active datasets after any changes.\n6. Download the HTML or JSON report if you need to save the result.

Each file must be a valid UTF-8 CSV and **10 MiB or smaller**. Use time-separated batches where possible. A shared binary target named `churn`, `target`, `label`, `y`, `outcome`, `default`, or `fraud` enables the model-metric panels when both batches contain both classes. Numeric ID-like fields are excluded from monitoring features.

## Local API

The API docs are available at `http://127.0.0.1:<port>/docs` while the app is running. The launcher prints the actual port.

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Service health and release version |
| `GET /api/scan` | Current report (loads demo if needed) |
| `POST /api/demo` | Reload deterministic demo data |
| `POST /api/analyze` | Analyze `reference` and `current` CSV uploads |
| `GET /api/preview` | Small data previews and row/column counts |
| `GET /api/readiness` | Readiness summary |
| `POST /api/what-if?action=...` | Run a controlled what-if scenario |
| `POST /api/recovery?improvement=0.65` | Run a recovery simulation |
| `GET /api/report.html` | Downloadable HTML report |
| `GET /api/report.json` | Downloadable JSON report |

## Run tests

From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q backend core tests
node --check frontend/app.js
```

Node.js is only needed for the optional JavaScript syntax check; the app itself is served by FastAPI.

## Project structure

```text
ModelGuard-AI-ML-Observability-Lab/
├── backend/main.py       # FastAPI endpoints, upload validation, report downloads
├── core/monitor.py       # Data quality, drift, model metrics, anomaly and readiness logic
├── frontend/             # Responsive dashboard UI
├── demo_data/            # Bundled demo reference/current CSVs
├── test_data/            # Healthy, drifted, and quality-issue scenarios
├── tests/                # Automated regression tests
├── run_modelguard.bat    # Windows entry point
└── start_modelguard.ps1  # Environment setup and local server launcher
```

## Known limitations

- The dashboard stores the latest analysis in a single in-memory session. It is designed for **one local user**, not simultaneous multi-user hosting.
- CSV parsing and ML analysis run locally; very large files are intentionally capped, and performance depends on the machine.
- Numeric features are monitored; non-numeric feature drift is not currently modeled.
- The binary target is inferred from a small list of common column names. This is a convenience heuristic, not schema governance.
- Reference/current AUC is an indicative comparison, not proof of causal model degradation. The reference model is trained on the reference data and evaluated on current data; population changes and label availability can affect interpretation.
- The Windows launcher and real-browser interaction should be smoke-tested on the target Windows machine after extraction. Automated checks in this release were run in a Python environment, not on every possible Windows setup.

## Portfolio description

**ModelGuard AI — ML Observability Lab** is a full-stack machine-learning monitoring project built with Python, Pandas, NumPy, SciPy, scikit-learn, FastAPI, and JavaScript. It compares reference/current datasets, measures data quality and feature drift, evaluates binary-classification health signals, detects anomalies, ranks potential issues, and exports analysis reports through a local web interface.
