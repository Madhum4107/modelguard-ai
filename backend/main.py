from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import html
import json

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from core.monitor import analyze, make_demo_data, read_csv_bytes, recovery_check, what_if_check

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
app = FastAPI(title="ModelGuard AI — ML Observability Lab", version="1.0.0", description="Local evidence-driven ML observability and model-risk decision support.")
app.mount("/assets", StaticFiles(directory=FRONTEND), name="assets")

# Single-user local portfolio session. This app is intentionally not a multi-user service.
SESSION: dict = {"report": None, "reference": None, "current": None, "source": "built-in demo dataset"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MiB per CSV
UPLOAD_CHUNK_BYTES = 64 * 1024


async def _read_upload_limited(upload: UploadFile, label: str) -> bytes:
    """Read an upload with a hard per-file limit instead of loading it unbounded."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(min(UPLOAD_CHUNK_BYTES, MAX_UPLOAD_BYTES + 1 - total))
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"{label} CSV exceeds the configured per-file limit ({MAX_UPLOAD_BYTES:,} bytes). Reduce the file size and try again.")
        chunks.append(chunk)
    return b"".join(chunks)


def _set_demo():
    ref, cur = make_demo_data()
    SESSION.update({"reference": ref, "current": cur, "report": analyze(ref, cur, "built-in demo dataset"), "source": "built-in demo dataset"})
    return SESSION["report"]


def _ensure_report():
    if SESSION["report"] is None:
        return _set_demo()
    return SESSION["report"]

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(FRONTEND / "index.html")

@app.get("/health")
def health_root():
    return health()

@app.get("/api/health")
def health():
    return {"status":"ok","service":"ModelGuard AI","version":app.version,"timestamp":datetime.now(timezone.utc).isoformat(),"data_loaded":SESSION["report"] is not None,"source":SESSION["source"]}

@app.get("/api/scan")
def get_scan():
    return _ensure_report()

@app.post("/api/demo")
def demo():
    return _set_demo()

@app.post("/api/analyze")
async def upload_and_analyze(reference: UploadFile = File(...), current: UploadFile = File(...)):
    if not (reference.filename or "").lower().endswith(".csv") or not (current.filename or "").lower().endswith(".csv"):
        raise HTTPException(400, "Please upload CSV files for both Reference and Current datasets.")
    try:
        ref_bytes = await _read_upload_limited(reference, "Reference")
        cur_bytes = await _read_upload_limited(current, "Current")
        ref = read_csv_bytes(ref_bytes)
        cur = read_csv_bytes(cur_bytes)
        shared = set(ref.columns) & set(cur.columns)
        if not shared:
            raise ValueError("The two datasets have no shared columns. They must describe comparable observations.")
        # Keep source labels readable and bounded; never use client filenames as filesystem paths.
        ref_name = (reference.filename or "reference.csv").replace("\\", "/").split("/")[-1][:120]
        cur_name = (current.filename or "current.csv").replace("\\", "/").split("/")[-1][:120]
        report = analyze(ref, cur, f"uploaded · {ref_name} vs {cur_name}")
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        # Keep internals out of API responses; the local server log can retain the traceback.
        import logging
        logging.getLogger("modelguard").exception("Dataset analysis failed")
        raise HTTPException(500, "Analysis failed unexpectedly. Check the server log and confirm both CSV files are valid.") from exc
    finally:
        await reference.close()
        await current.close()
    SESSION.update({"reference":ref,"current":cur,"report":report,"source":report["source"]})
    return report

@app.get("/api/preview")
def preview():
    if SESSION["reference"] is None or SESSION["current"] is None:
        _set_demo()
    def pack(df):
        return {"rows":len(df),"columns":[str(x) for x in df.columns],"sample":json.loads(df.head(8).to_json(orient="records",date_format="iso"))}
    return {"reference":pack(SESSION["reference"]),"current":pack(SESSION["current"]),"source":SESSION["source"]}

@app.get("/api/readiness")
def readiness(): return _ensure_report()["readiness"]

@app.post("/api/what-if")
def what_if(action: str = "fill_missing"):
    allowed={"fill_missing","investigate_drift","retrain_model","rollback_model","quarantine_segment"}
    if action not in allowed: raise HTTPException(400,"Unknown action")
    return what_if_check(_ensure_report(), action)

@app.post("/api/recovery")
def recovery(improvement: float = .65):
    if not 0 <= improvement <= 1: raise HTTPException(400,"improvement must be between 0 and 1")
    return recovery_check(_ensure_report(), improvement)

@app.get("/api/report.html")
def report_html():
    r=_ensure_report(); mh=r["model_health"]; ready=r["readiness"]
    rows=lambda items: "".join(f"<tr><td>{html.escape(str(x.get('feature',x.get('name',''))))}</td><td>{html.escape(str(x.get('severity','')))}</td><td>{html.escape(str(x.get('psi',x.get('value',''))))}</td></tr>" for x in items)
    page=f"""<!doctype html><html><head><meta charset='utf-8'><title>ModelGuard AI Report</title><style>body{{font-family:Arial,sans-serif;max-width:1000px;margin:40px auto;color:#172033}}h1{{margin-bottom:4px}}table{{border-collapse:collapse;width:100%;margin:18px 0}}th,td{{border:1px solid #d8dee8;padding:9px;text-align:left}}.hero{{padding:20px;background:#eef6fb;border-radius:12px}}.muted{{color:#65758b}}</style></head><body><div class='hero'><h1>ModelGuard AI — Analysis Report</h1><p class='muted'>{html.escape(r['source'])}</p><h2>Readiness: {ready['score']}/100 · {html.escape(ready['status'])}</h2><p>Severity: <b>{html.escape(r['severity'])}</b> · Reference rows: {r['reference_rows']} · Current rows: {r['current_rows']}</p></div><h2>Model health</h2><pre>{html.escape(json.dumps(mh,indent=2))}</pre><h2>Feature drift</h2><table><tr><th>Feature</th><th>Severity</th><th>PSI</th></tr>{rows(r['drift'])}</table><h2>Monitoring scope</h2><p>Data quality score: <b>{r['data_quality_score']}/100</b> · Monitoring coverage: <b>{r['monitoring_coverage']}%</b></p><p>Excluded identifier columns: {html.escape(', '.join(r.get('excluded_columns', [])) or 'None')}</p><h2>Data quality</h2><table><tr><th>Check</th><th>Severity</th><th>Value</th></tr>{rows(r['quality'])}</table><h2>Recommendations</h2><ul>{''.join(f"<li><b>{html.escape(x['priority'])}</b> — {html.escape(x['action'])}: {html.escape(x['why'])}</li>" for x in r['recommendations'])}</ul><p class='muted'>Generated locally by ModelGuard AI. Validate any production decision with fresh, time-separated observations.</p></body></html>"""
    return Response(page, media_type="text/html", headers={"Content-Disposition":"attachment; filename=ModelGuard-AI-report.html"})

@app.get("/api/report.json")
def report_json():
    return Response(json.dumps(_ensure_report(),indent=2),media_type="application/json",headers={"Content-Disposition":"attachment; filename=ModelGuard-AI-report.json"})

@app.get("/{path:path}", include_in_schema=False)
def spa_fallback(path: str):
    if path.startswith("api/"): raise HTTPException(404,"API route not found")
    return FileResponse(FRONTEND / "index.html")
