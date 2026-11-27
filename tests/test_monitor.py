from io import BytesIO
from pathlib import Path
import pandas as pd
from fastapi.testclient import TestClient
import backend.main as main_module
from backend.main import app
from core.monitor import analyze, make_demo_data, read_csv_bytes

client=TestClient(app)

def test_demo_is_reproducible():
    a=analyze(*make_demo_data(),"demo"); b=analyze(*make_demo_data(),"demo")
    assert a["drift"]==b["drift"] and a["readiness"]==b["readiness"]

def test_uploaded_data_changes_analysis():
    ref,cur=make_demo_data(42,500)
    cur2=cur.copy(); cur2["monthly_spend"]=(cur2["monthly_spend"]*2).clip(1,1000)
    a=analyze(ref,cur,"one"); b=analyze(ref,cur2,"two")
    assert a["source"]!=b["source"]
    assert any(x["psi"]!=y["psi"] for x,y in zip(a["drift"],b["drift"]))

def test_csv_reader():
    df=read_csv_bytes(b"age,churn\n20,0\n30,1\n")
    assert list(df.columns)==["age","churn"] and len(df)==2

def test_api_health_and_demo():
    assert client.get('/api/health').status_code==200
    r=client.post('/api/demo'); assert r.status_code==200
    assert r.json()["source"]=="built-in demo dataset"

def test_api_upload_and_report():
    ref,cur=make_demo_data(7,120)
    files={"reference":("reference.csv",ref.to_csv(index=False).encode(),"text/csv"),"current":("current.csv",cur.to_csv(index=False).encode(),"text/csv")}
    r=client.post('/api/analyze',files=files)
    assert r.status_code==200
    assert r.json()["reference_rows"]==120
    assert client.get('/api/preview').status_code==200
    assert client.get('/api/report.html').status_code==200
    assert client.get('/api/report.json').status_code==200

def test_api_rejects_non_csv():
    r=client.post('/api/analyze',files={"reference":("a.txt",b"x","text/plain"),"current":("b.csv",b"a\n1","text/csv")})
    assert r.status_code==400

def test_what_if():
    assert client.post('/api/what-if?action=investigate_drift').status_code==200
    assert client.post('/api/recovery?improvement=.65').status_code==200
    assert client.post('/api/recovery?improvement=2').status_code==400

def test_identifier_columns_are_excluded_from_monitoring():
    ref, cur = make_demo_data(42, 500)
    ref["customer_id"] = range(100000, 100000 + len(ref))
    cur["customer_id"] = range(900000, 900000 + len(cur))
    report = analyze(ref, cur, "identifier-test")
    assert "customer_id" in report["excluded_columns"]
    assert "customer_id" not in report["features"]
    assert all(row["feature"] != "customer_id" for row in report["drift"])
    assert all(c["feature"] != "customer_id" for c in report["root_causes"])

def test_model_metrics_have_reference_calibrated_threshold():
    ref, cur = make_demo_data(42, 800)
    report = analyze(ref, cur, "metric-test")
    assert report["model_health"]["available"]
    assert 0.10 <= report["model_health"]["threshold"] <= 0.90
    assert report["model_health"]["current_f1"] >= 0
    assert report["data_quality_score"] <= 100
    assert 0 <= report["monitoring_coverage"] <= 100

def test_real_uploaded_test_shape_and_drift():
    demo_dir = Path(__file__).resolve().parents[1] / "demo_data"
    ref = pd.read_csv(demo_dir / "reference.csv")
    cur = pd.read_csv(demo_dir / "current.csv")
    report = analyze(ref, cur, "demo-files")
    assert report["target"] == "churn"
    assert report["features"]
    assert "customer_id" not in report["features"]

def test_monitoring_coverage_excludes_identifiers_but_reports_real_coverage():
    ref, cur = make_demo_data(11, 250)
    ref["customer_id"] = range(1000, 1000 + len(ref))
    cur["customer_id"] = range(2000, 2000 + len(cur))
    ref["plan_code"] = range(5000, 5000 + len(ref))
    cur["plan_code"] = range(9000, 9000 + len(cur))
    report = analyze(ref, cur, "coverage-test")
    assert report["monitoring_coverage"] == 100
    assert "customer_id" in report["excluded_columns"]
    assert "plan_code" in report["excluded_columns"]


def test_drift_contains_direction_and_means():
    ref, cur = make_demo_data(12, 250)
    cur["monthly_spend"] = cur["monthly_spend"] * 1.5
    report = analyze(ref, cur, "direction-test")
    row = next(x for x in report["drift"] if x["feature"] == "monthly_spend")
    assert row["direction"] == "up"
    assert row["current_mean"] > row["reference_mean"]


def test_api_rejects_empty_csv():
    r = client.post('/api/analyze', files={"reference":("a.csv",b"","text/csv"),"current":("b.csv",b"age,churn\n20,0\n30,1\n","text/csv")})
    assert r.status_code == 400


def test_api_rejects_malformed_csv_without_server_error():
    r = client.post('/api/analyze', files={
        "reference": ("reference.csv", b'\"broken,header\n1,2', "text/csv"),
        "current": ("current.csv", b"age,churn\n20,0\n30,1\n", "text/csv"),
    })
    assert r.status_code == 400

def test_api_rejects_datasets_without_shared_columns():
    r = client.post('/api/analyze', files={
        "reference": ("reference.csv", b"age,churn\n20,0\n30,1\n", "text/csv"),
        "current": ("current.csv", b"spend,target\n20,0\n30,1\n", "text/csv"),
    })
    assert r.status_code == 400

def test_model_metrics_degrade_gracefully_when_current_target_has_one_class():
    ref, cur = make_demo_data(23, 250)
    cur["churn"] = 0
    report = analyze(ref, cur, "single-class-current")
    assert report["model_health"]["available"] is False
    assert report["model_comparison"] == []
    assert report["drift"]

def test_empty_reference_feature_values_do_not_crash_analysis():
    ref, cur = make_demo_data(24, 250)
    ref["monthly_spend"] = float('nan')
    report = analyze(ref, cur, "empty-reference-feature")
    assert report["reference_rows"] == 250
    assert isinstance(report["severity"], str)


def test_api_rejects_upload_above_configured_limit(monkeypatch):
    monkeypatch.setattr(main_module, "MAX_UPLOAD_BYTES", 16)
    response = client.post('/api/analyze', files={
        "reference": ("reference.csv", b"age,churn\n20,0\n30,1\n", "text/csv"),
        "current": ("current.csv", b"age,churn\n20,0\n30,1\n", "text/csv"),
    })
    assert response.status_code == 413
    assert "configured per-file limit" in response.json()["detail"]


def test_health_exposes_release_version():
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.json()["version"] == "1.0.0"


def test_csv_reader_rejects_duplicate_headers():
    try:
        read_csv_bytes(b"age,age\n20,30\n")
    except ValueError as exc:
        assert "duplicate column names" in str(exc).lower()
    else:
        raise AssertionError("duplicate CSV headers should be rejected")


def test_unexpected_analysis_error_does_not_leak_internals(monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("private filesystem path should not be returned")

    monkeypatch.setattr(main_module, "analyze", explode)
    response = client.post('/api/analyze', files={
        "reference": ("reference.csv", b"age,churn\n20,0\n30,1\n", "text/csv"),
        "current": ("current.csv", b"age,churn\n21,0\n31,1\n", "text/csv"),
    })
    assert response.status_code == 500
    assert "private filesystem path" not in response.text
    assert "Analysis failed unexpectedly" in response.json()["detail"]


def test_refresh_recomputes_active_dataset(monkeypatch):
    client.post("/api/demo")
    original = main_module.analyze
    calls = []

    def tracked(ref, cur, source):
        calls.append((len(ref), len(cur), source))
        return original(ref, cur, source)

    monkeypatch.setattr(main_module, "analyze", tracked)
    response = client.get("/api/scan")
    assert response.status_code == 200
    assert len(calls) == 1
    assert calls[0][2] == "built-in demo dataset"
    assert response.json()["source"] == "built-in demo dataset"
