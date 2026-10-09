from __future__ import annotations
from io import BytesIO, StringIO
import csv
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
import re

TARGET_CANDIDATES = ["churn", "target", "label", "y", "outcome", "default", "fraud"]
IDENTIFIER_NAME_RE = re.compile(r"(^|_)(id|uuid|guid|key|number|no|code)$|(^|_)(customer|user|account|transaction|invoice|record|row)[_-]?(id|key|number|no|code)$", re.I)


def make_demo_data(seed: int = 42, n: int = 6000):
    rng = np.random.default_rng(seed)
    age = rng.normal(38, 10, n).clip(18, 75)
    tenure = rng.gamma(3.2, 9, n).clip(1, 96)
    spend = rng.normal(72, 24, n).clip(15, 180)
    tickets = rng.poisson(1.7, n).clip(0, 12)
    contract = rng.choice([1, 12, 24], n, p=[.42, .38, .20])
    logits = -1.9 + .045*(45-age) + .035*tickets - .018*tenure - .018*(contract-1) + .006*(spend-70)
    y = rng.binomial(1, 1/(1+np.exp(-logits)))
    ref = pd.DataFrame({"age":age,"tenure_months":tenure,"monthly_spend":spend,"support_tickets":tickets,"contract_months":contract,"churn":y})
    cur = ref.sample(frac=1, random_state=seed+1).reset_index(drop=True).copy()
    cur["monthly_spend"] = (cur["monthly_spend"]*1.18 + rng.normal(0, 7, n)).clip(15, 220)
    cur["support_tickets"] = (cur["support_tickets"] + rng.binomial(2, .28, n)).clip(0, 15)
    cur.loc[rng.random(n)<.045, "monthly_spend"] = np.nan
    cur.loc[rng.random(n)<.018, "support_tickets"] = np.nan
    cur_logit = -1.7 + .035*(45-cur["age"].fillna(cur["age"].median())) + .085*cur["support_tickets"].fillna(0) - .012*cur["tenure_months"] - .012*(cur["contract_months"]-1) + .014*(cur["monthly_spend"].fillna(cur["monthly_spend"].median())-70)
    cur["churn"] = rng.binomial(1, 1/(1+np.exp(-cur_logit)))
    cur = pd.concat([cur, cur.iloc[:24]], ignore_index=True)
    return ref, cur


def read_csv_bytes(data: bytes) -> pd.DataFrame:
    if not data:
        raise ValueError("The uploaded CSV is empty.")
    try:
        # Inspect the original header before pandas can silently rename duplicate fields.
        text = data.decode("utf-8-sig")
        header = next(csv.reader(StringIO(text), strict=True), None)
        if not header:
            raise ValueError("The uploaded CSV contains no columns.")
        clean_header = [str(c).strip() for c in header]
        if any(not c for c in clean_header):
            raise ValueError("CSV contains an empty column name.")
        if len(set(clean_header)) != len(clean_header):
            raise ValueError("CSV contains duplicate column names.")
        df = pd.read_csv(BytesIO(data), encoding="utf-8-sig")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read CSV. Check that it is a valid UTF-8 CSV file. ({type(exc).__name__})") from exc
    if df.empty:
        raise ValueError("The uploaded CSV contains no rows.")
    if df.shape[1] < 1:
        raise ValueError("The uploaded CSV contains no columns.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _target(ref: pd.DataFrame, cur: pd.DataFrame) -> str | None:
    lower = {c.lower(): c for c in ref.columns}
    for name in TARGET_CANDIDATES:
        if name in lower and lower[name] in cur.columns:
            return lower[name]
    return None


def _numeric_features(ref: pd.DataFrame, cur: pd.DataFrame, target: str | None) -> tuple[list[str], list[str]]:
    shared = [c for c in ref.columns if c in cur.columns and c != target]
    numeric = [c for c in shared if pd.api.types.is_numeric_dtype(ref[c]) and pd.api.types.is_numeric_dtype(cur[c])]
    excluded = []
    features = []
    for col in numeric:
        name = str(col).strip().lower()
        ref_unique = ref[col].nunique(dropna=True) / max(len(ref), 1)
        cur_unique = cur[col].nunique(dropna=True) / max(len(cur), 1)
        looks_like_id = bool(IDENTIFIER_NAME_RE.search(name))
        if looks_like_id:
            excluded.append(col)
        else:
            features.append(col)
    return features, excluded


def _eligible_numeric_columns(ref: pd.DataFrame, cur: pd.DataFrame, target: str | None) -> list[str]:
    """Numeric fields that are eligible for monitoring, excluding target and IDs."""
    shared = [c for c in ref.columns if c in cur.columns and c != target]
    numeric = [c for c in shared if pd.api.types.is_numeric_dtype(ref[c]) and pd.api.types.is_numeric_dtype(cur[c])]
    return [c for c in numeric if not IDENTIFIER_NAME_RE.search(str(c).strip().lower())]


def _psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    a = pd.to_numeric(ref, errors="coerce").dropna().to_numpy(); b = pd.to_numeric(cur, errors="coerce").dropna().to_numpy()
    if len(a) < 20 or len(b) < 20: return 0.0
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3: return 0.0
    edges[0] -= 1e-9; edges[-1] += 1e-9
    pa = np.histogram(a, edges)[0] / len(a); pb = np.histogram(b, edges)[0] / len(b)
    pa = np.clip(pa, 1e-6, None); pb = np.clip(pb, 1e-6, None)
    return float(np.sum((pb-pa) * np.log(pb/pa)))


def _drift_rows(ref, cur, features):
    rows = []
    for col in features:
        r, c = ref[col], cur[col]
        psi = _psi(r, c)
        ks = float(ks_2samp(r.dropna(), c.dropna()).statistic) if r.dropna().size and c.dropna().size else 0.0
        level = "high" if psi >= .25 or ks >= .20 else "medium" if psi >= .10 or ks >= .10 else "low"
        rmean=float(pd.to_numeric(r, errors="coerce").mean()) if r.notna().any() else 0.0
        cmean=float(pd.to_numeric(c, errors="coerce").mean()) if c.notna().any() else 0.0
        direction="up" if cmean > rmean * 1.02 else "down" if cmean < rmean * 0.98 else "stable"
        rows.append({"feature": col, "psi": round(psi, 4), "ks_stat": round(ks, 4), "severity": level, "reference_mean": round(rmean, 4), "current_mean": round(cmean, 4), "direction": direction})
    return rows


def _quality(ref, cur):
    base = max(len(ref), 1)
    checks = [{"name":"Row count change", "value":round((len(cur)-len(ref))/base*100,2), "unit":"%", "severity":"high" if abs(len(cur)-len(ref))/base>.10 else "low"}]
    dup = int(cur.duplicated().sum())
    checks.append({"name":"Duplicate rows", "value":dup, "unit":"rows", "severity":"high" if dup else "low"})
    for col in cur.columns:
        null = float(cur[col].isna().mean())
        if null > 0:
            sev = "high" if null >= .05 else "medium"
            checks.append({"name":f"Null rate · {col}", "value":round(null*100,2), "unit":"%", "severity":sev})
    missing = sorted(set(ref.columns)-set(cur.columns)); extra = sorted(set(cur.columns)-set(ref.columns))
    checks.append({"name":"Schema changes", "value":len(missing)+len(extra), "unit":"fields", "severity":"high" if missing or extra else "low", "detail":{"missing":missing,"extra":extra}})
    return checks


def _binary_target(ref: pd.DataFrame, cur: pd.DataFrame, target: str | None):
    """Return a shared binary target as 0/1, supporting numeric or text labels."""
    if not target:
        return None
    raw_r, raw_c = ref[target], cur[target]
    combined = pd.concat([raw_r, raw_c], ignore_index=True).dropna()
    if combined.nunique() != 2:
        return None
    classes = list(pd.unique(combined))
    if set(classes) == {0, 1}:
        mapping = {0: 0, 1: 1}
    else:
        ordered = sorted(classes, key=lambda x: str(x).lower())
        mapping = {ordered[0]: 0, ordered[1]: 1}
    yr = raw_r.map(mapping); yc = raw_c.map(mapping)
    vr, vc = yr.notna(), yc.notna()
    yr, yc = yr[vr].astype(int), yc[vc].astype(int)
    if yr.nunique() != 2 or yc.nunique() != 2:
        return None
    return yr, yc, vr, vc

def _reference_threshold_and_auc(ref: pd.DataFrame, features: list[str], target: str):
    y = pd.to_numeric(ref[target], errors="coerce")
    valid = y.notna()
    y = y[valid].astype(int)
    if y.nunique() != 2 or len(y) < 40 or y.value_counts().min() < 5:
        return 0.5, None
    X = ref.loc[valid, features]
    n_splits = min(3, int(y.value_counts().min()))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y), dtype=float)
    for train_idx, val_idx in skf.split(X, y):
        imp = SimpleImputer(strategy="median")
        model = RandomForestClassifier(n_estimators=70, max_depth=8, min_samples_leaf=8, random_state=42, n_jobs=-1)
        model.fit(imp.fit_transform(X.iloc[train_idx]), y.iloc[train_idx])
        oof[val_idx] = model.predict_proba(imp.transform(X.iloc[val_idx]))[:, 1]
    thresholds = np.linspace(0.10, 0.90, 81)
    scores = [f1_score(y, (oof >= t).astype(int), zero_division=0) for t in thresholds]
    best = float(thresholds[int(np.argmax(scores))])
    return best, float(roc_auc_score(y, oof))


def _model_health(ref, cur, features, target):
    target_data = _binary_target(ref, cur, target)
    # An all-null reference feature cannot be median-imputed or learned from.
    features = [f for f in features if ref[f].notna().any()]
    if not target_data or len(features) < 1:
        return {"available": False, "reason": "A shared numeric binary target and at least one numeric feature are required for model metrics."}
    yr, yc, vr, vc = target_data
    threshold, ref_auc_cv = _reference_threshold_and_auc(ref, features, target)
    imp = SimpleImputer(strategy="median")
    Xr = imp.fit_transform(ref.loc[vr, features]); Xc = imp.transform(cur.loc[vc, features])
    model = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=8, random_state=42, n_jobs=-1).fit(Xr, yr)
    pp = model.predict_proba(Xc)[:, 1]
    pr = (pp >= threshold).astype(int)
    cur_auc = roc_auc_score(yc, pp)
    return {"available": True, "target": target, "reference_auc": round(float(ref_auc_cv if ref_auc_cv is not None else roc_auc_score(yr, model.predict_proba(Xr)[:,1])),4), "current_auc": round(float(cur_auc),4), "current_accuracy": round(float(accuracy_score(yc,pr)),4), "current_f1": round(float(f1_score(yc,pr,zero_division=0)),4), "threshold": round(threshold,2), "auc_drop_points": round(float((ref_auc_cv if ref_auc_cv is not None else 0)-cur_auc),4), "brier_score": round(float(brier_score_loss(yc,pp)),4), "feature_importance":[{"feature":f,"importance":round(float(v),4)} for f,v in sorted(zip(features,model.feature_importances_),key=lambda x:x[1],reverse=True)]}


def _model_comparison(ref, cur, features, target):
    target_data = _binary_target(ref, cur, target)
    # Keep model comparison aligned with the model-health path.
    features = [f for f in features if ref[f].notna().any()]
    if not target_data or len(features) < 1: return []
    yr, yc, vr, vc = target_data
    threshold, _ = _reference_threshold_and_auc(ref, features, target)
    imp=SimpleImputer(strategy="median"); Xr=imp.fit_transform(ref.loc[vr,features]); Xc=imp.transform(cur.loc[vc,features])
    models={"Random Forest":RandomForestClassifier(n_estimators=160,max_depth=8,min_samples_leaf=8,random_state=42,n_jobs=-1),"Logistic Regression":LogisticRegression(max_iter=1000,C=1.0,random_state=42)}
    out=[]
    for name,m in models.items():
        m.fit(Xr,yr); p=m.predict_proba(Xc)[:,1]; pred=(p>=threshold).astype(int)
        out.append({"model":name,"current_auc":round(float(roc_auc_score(yc,p)),4),"f1":round(float(f1_score(yc,pred,zero_division=0)),4),"brier":round(float(brier_score_loss(yc,p)),4),"threshold":round(threshold,2)})
    return sorted(out,key=lambda x:x["current_auc"],reverse=True)


def _data_quality_score(quality: list[dict]) -> float:
    penalties = 0.0
    for x in quality:
        sev = x["severity"]
        if sev == "high": penalties += 18
        elif sev == "medium": penalties += 8
    return round(max(0.0, min(100.0, 100.0 - penalties)), 1)

def _anomalies(ref, cur, features):
    """Score current rows against the reference population, not against itself."""
    features = [f for f in features if ref[f].notna().any()]
    if len(features) < 1 or len(ref) < 20 or len(cur) < 1:
        return {"available":False,"anomaly_rate":None,"top_rows":[],"baseline_rate":None}
    imp=SimpleImputer(strategy="median")
    xr=imp.fit_transform(ref[features]); xc=imp.transform(cur[features])
    iso=IsolationForest(n_estimators=160,contamination=0.05,random_state=42,n_jobs=-1).fit(xr)
    labels=iso.predict(xc); scores=-iso.score_samples(xc); idx=np.argsort(scores)[-10:][::-1]
    return {"available":True,"anomaly_rate":round(float((labels==-1).mean()*100),2),"baseline_rate":5.0,"top_rows":[{"row":int(i),"score":round(float(scores[i]),4)} for i in idx],"method":"Isolation Forest trained on reference population"}

def _segments(ref, cur, target):
    if "contract_months" not in ref.columns or "contract_months" not in cur.columns or not target: return []
    def bucket(v): return "Short contract" if v<=1 else "Mid contract" if v<=12 else "Long contract"
    rows=[]
    for name in ["Short contract","Mid contract","Long contract"]:
        rv=ref.loc[ref.contract_months.map(bucket)==name,target]; cv=cur.loc[cur.contract_months.map(bucket)==name,target]
        rv=pd.to_numeric(rv,errors="coerce").dropna(); cv=pd.to_numeric(cv,errors="coerce").dropna()
        rr=float(rv.mean()*100) if len(rv) else 0; cr=float(cv.mean()*100) if len(cv) else 0
        rows.append({"segment":name,"reference_rate":round(rr,2),"current_rate":round(cr,2),"rate_change_points":round(cr-rr,2),"customers":int(len(cv))})
    return rows


def _readiness(quality, drift, model, anomalies):
    penalties=min(30,sum(x["severity"]=="high" for x in quality)*10)+min(30,sum(x["severity"]=="high" for x in drift)*8)
    if model.get("available"): penalties += min(25,max(0,model.get("auc_drop_points",0))*120)
    if anomalies.get("available") and anomalies.get("anomaly_rate",0)>3: penalties += min(15,(anomalies["anomaly_rate"]-3)*2)
    score=max(0,round(100-penalties,1)); status="ready" if score>=80 else "review" if score>=60 else "blocked"
    return {"score":score,"status":status,"evidence":["data quality","feature drift","model performance" if model.get("available") else "model metrics unavailable","anomaly rate" if anomalies.get("available") else "anomaly scan unavailable"]}


def _severity(drift, quality, model, anomalies):
    score=sum(x["severity"]=="high" for x in drift)*22+sum(x["severity"]=="high" for x in quality)*20
    if model.get("available"): score += max(0,model.get("auc_drop_points",0))*100
    if anomalies.get("available"): score += max(0,anomalies.get("anomaly_rate",0)-3)*2
    return "critical" if score>=45 else "high" if score>=20 else "medium" if score>=8 else "low"


def analyze(ref: pd.DataFrame, cur: pd.DataFrame, source="uploaded"):
    target=_target(ref,cur); features, excluded_columns=_numeric_features(ref,cur,target)
    quality=_quality(ref,cur); drift=_drift_rows(ref,cur,features); model=_model_health(ref,cur,features,target); anomalies=_anomalies(ref,cur,features)
    quality_score=_data_quality_score(quality)
    eligible=_eligible_numeric_columns(ref,cur,target)
    coverage=round(len(features)/max(len(eligible),1)*100,1) if eligible else 0.0
    severity=_severity(drift,quality,model,anomalies)
    ranked=[]
    imports={x["feature"]:x["importance"] for x in model.get("feature_importance",[])}
    for d in drift:
        imp=imports.get(d["feature"],1/len(features) if features else 0)
        ranked.append({"feature":d["feature"],"cause_score":round(d["psi"]*imp,4),"psi":d["psi"],"model_importance":round(imp,4),"reason":"distribution shift + model sensitivity" if d["severity"]!="low" else "stable"})
    ranked.sort(key=lambda x:x["cause_score"],reverse=True); root_causes=[x for x in ranked if x["cause_score"]>0.005][:5]
    recs=[]
    nulls=[x for x in quality if x["name"].startswith("Null rate") and x["severity"] in ("high","medium")]
    if nulls: recs.append({"priority":"high","action":"Validate upstream null handling before scoring","why":"The current dataset contains missing values in one or more features","safe":True})
    for d in drift:
        if d["severity"]=="high": recs.append({"priority":"high","action":f"Investigate drift in {d['feature']}","why":f"PSI={d['psi']:.3f} / KS={d['ks_stat']:.3f}; confirm source and population changes first","safe":False})
    if model.get("available") and model.get("auc_drop_points",0)>.05: recs.append({"priority":"high","action":"Trigger model review / retraining workflow","why":f"Current AUC is {model['auc_drop_points']:.3f} points below reference","safe":False})
    if not recs: recs.append({"priority":"low","action":"Continue monitoring on the next time-separated batch","why":"No high-risk evidence was detected in this scan","safe":True})
    recs.append({"priority":"medium","action":"Compare current feature distributions with the reference window","why":"Confirm whether the shift is transient or persistent before changing the model","safe":True})
    lineage={"nodes":[{"id":"source","label":"Data Source","type":"data"},{"id":"features","label":"Feature Pipeline","type":"feature"},{"id":"model","label":"Model","type":"model"},{"id":"consumer","label":"Decision Workflow","type":"consumer"}],"edges":[["source","features"],["features","model"],["model","consumer"]]}
    readiness=_readiness(quality,drift,model,anomalies)
    return {"status":"incident" if severity in ("high","critical") else "healthy_with_warnings" if severity=="medium" else "healthy", "analysis_version":"1.0.0","severity":severity,"source":source,"reference_rows":len(ref),"current_rows":len(cur),"reference_columns":len(ref.columns),"current_columns":len(cur.columns),"target":target,"features":features,"excluded_columns":excluded_columns,"monitoring_coverage":coverage,"data_quality_score":quality_score,"quality":quality,"drift":drift,"model_health":model,"model_comparison":_model_comparison(ref,cur,features,target),"anomalies":anomalies,"root_causes":root_causes,"recommendations":recs,"impact":{"affected_assets":["Monitored Model","Risk Dashboard","Decision Workflow"],"count":3},"lineage":lineage,"segments":_segments(ref,cur,target),"readiness":readiness}


def scan():
    ref,cur=make_demo_data(); return analyze(ref,cur,"built-in demo dataset")


def recovery_check(report: dict, improvement: float=0.65):
    mh=report["model_health"]
    if not mh.get("available"): return {"available":False,"message":"Recovery simulation needs target labels in both datasets."}
    baseline=mh["current_auc"]; simulated=min(.999,baseline+max(0,min(improvement,1))*max(0,mh["auc_drop_points"]))
    return {"available":True,"before_auc":baseline,"reference_auc":mh["reference_auc"],"simulated_post_maintenance_auc":round(float(simulated),4),"recovery_fraction":round(float(improvement),2),"interpretation":"Simulation only. Validate against a fresh post-maintenance observation before deployment."}


def what_if_check(report: dict, action: str):
    before=report["severity"]
    if action=="fill_missing":
        null_pct=round(sum(x["value"] for x in report["quality"] if x["name"].startswith("Null rate")),2)
        return {"action":action,"before_severity":before,"estimated_quality_improvement_points":null_pct,"message":"Handle missing values in the affected feature pipeline, then re-score a fresh batch. This is an impact estimate, not a guaranteed outcome."}
    if action=="investigate_drift":
        top=report["root_causes"][0] if report["root_causes"] else None
        return {"action":action,"before_severity":before,"focus_feature":top["feature"] if top else None,"message":"Check source-system changes, feature generation, population mix and time-window alignment before retraining."}
    if action=="retrain_model": return {"action":action,"before_severity":before,"gate":"review_required","message":"Retrain only after confirming persistent drift and data-quality stability. Compare against the current champion on a time-separated validation window."}
    if action=="rollback_model": return {"action":action,"before_severity":before,"gate":"candidate_only","message":"Rollback requires a previously validated model and evidence that the current model caused the regression."}
    return {"action":action,"before_severity":before,"gate":"segment_review","message":"Quarantine a segment only after independently confirming anomalous data or behaviour; do not silently exclude customers."}
