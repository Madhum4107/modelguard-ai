const $ = (s) => document.querySelector(s);
const esc = (x) => String(x ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));
const fmt = (x, n = 2) => typeof x === 'number' ? x.toLocaleString(undefined, {maximumFractionDigits:n}) : esc(x ?? '—');

async function api(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { message = (await response.json()).detail || message; } catch (_) {}
    throw new Error(message);
  }
  return response.json();
}

function pill(value) {
  const text = String(value ?? '—');
  return `<b class="pill ${text.toLowerCase()}">${esc(text)}</b>`;
}

function panel(title, body) {
  return `<section class="panel"><h2>${esc(title)}</h2>${body}</section>`;
}

function row(label, detail, badge) {
  return `<div class="row"><div><b>${esc(label)}</b><small>${esc(detail)}</small></div>${pill(badge)}</div>`;
}

function render(report) {
  const model = report.model_health || {};
  const quality = report.quality || [];
  const drift = report.drift || [];
  const causes = report.root_causes || [];
  const recommendations = report.recommendations || [];
  const comparison = report.model_comparison || [];
  const segments = report.segments || [];
  const impact = report.impact || {affected_assets: []};
  const excluded = report.excluded_columns || [];
  const coverage = report.monitoring_coverage ?? 0;
  const qualityScore = report.data_quality_score ?? 0;

  const modelBody = model.available
    ? `<div class="stats">Reference AUC <b>${fmt(model.reference_auc,4)}</b> · Current AUC <b>${fmt(model.current_auc,4)}</b> · F1 <b>${fmt(model.current_f1,4)}</b> · Accuracy <b>${fmt(model.current_accuracy,4)}</b> · Brier <b>${fmt(model.brier_score,4)}</b></div>`
    : `<div class="empty">${esc(model.reason || 'Model metrics unavailable.')}</div>`;

  const driftBody = drift.length
    ? drift.map((x) => row(x.feature, `PSI ${fmt(x.psi,3)} · KS ${fmt(x.ks_stat,3)}`, x.severity)).join('')
    : '<div class="empty">No numeric shared features available.</div>';

  const causeBody = causes.length
    ? causes.map((x, i) => `<div class="cause"><i>${i+1}</i><div><b>${esc(x.feature)}</b><small>Cause score ${fmt(x.cause_score,4)} · ${esc(x.reason)}</small></div><strong>${fmt(x.psi,3)}</strong></div>`).join('')
    : '<div class="empty">No dominant root cause detected.</div>';

  const recommendationBody = recommendations.length
    ? recommendations.map((x) => `<div class="action">${pill(x.priority)}<div><b>${esc(x.action)}</b><small>${esc(x.why)}</small></div></div>`).join('')
    : '<div class="empty">No recommendations.</div>';

  const qualityBody = quality.map((x) => row(x.name, `${fmt(x.value)} ${x.unit || ''}`, x.severity)).join('') || '<div class="empty">No quality checks.</div>';
  const comparisonBody = comparison.length
    ? comparison.map((x, i) => row(`${i ? 'Alternative' : 'Current leader'} · ${x.model}`, `AUC ${fmt(x.current_auc,4)} · F1 ${fmt(x.f1,4)}`, `Brier ${fmt(x.brier,4)}`)).join('')
    : '<div class="empty">Target labels with two classes in both datasets are required.</div>';
  const segmentBody = segments.length
    ? segments.map((x) => row(x.segment, `${fmt(x.customers,0)} customers · current ${fmt(x.current_rate,2)}% · ref ${fmt(x.reference_rate,2)}%`, `${x.rate_change_points > 0 ? '+' : ''}${fmt(x.rate_change_points,2)} pts`)).join('')
    : '<div class="empty">Optional segment analysis needs contract_months + target.</div>';

  $('#app').innerHTML = `
    <main>
      <header>
        <div>
          <small>ML OBSERVABILITY · MODEL RISK CONTROL</small>
          <h1>ModelGuard <i>AI</i></h1>
          <p>Monitor data quality, detect drift, evaluate model signals, and simulate a response — without touching production systems.</p>
        </div>
        <div class="top"><span>● API CONNECTED</span><button id="demo">Load demo</button><button id="refresh">Refresh</button></div>
      </header>
      <section class="panel ingest">
        <h2>Data workspace <small>REAL INPUT · DATA-DRIVEN</small></h2>
        <div class="uploads">
          <label>① Reference CSV<input id="ref" type="file" accept=".csv"></label>
          <label>② Current CSV<input id="cur" type="file" accept=".csv"></label>
        </div>
        <div class="uploadbar"><span id="msg">Choose both CSVs or load the demo.</span><button id="analyze">Analyze datasets</button><a href="/api/report.html">Download report</a><a href="/api/report.json">JSON</a></div>
        <details><summary>What should I upload?</summary><p>Reference = baseline/known-good period. Current = latest batch. If both contain a shared binary target such as churn/target/label, model metrics are calculated too. CSV limit: 10 MiB per file. Use time-separated data when possible.</p></details>
      </section>
      <div class="source"><b>Active source:</b> ${esc(report.source)} · ${fmt(report.reference_rows,0)} reference → ${fmt(report.current_rows,0)} current · ${(report.features || []).length} monitorable numeric features · target: ${esc(report.target || 'not detected')}<br><small>Excluded from monitoring: ${esc(excluded.length ? excluded.join(', ') : 'None')}</small></div>
      <section class="metrics">
        <article><span>Readiness</span><strong>${fmt(report.readiness.score,1)}/100</strong><small>${esc(report.readiness.status)}</small></article>
        <article><span>Current AUC</span><strong>${model.available ? fmt(model.current_auc,4) : 'N/A'}</strong><small>${model.available ? `Reference ${fmt(model.reference_auc,4)}` : 'Target labels required'}</small></article>
        <article><span>AUC change</span><strong>${model.available ? fmt(model.auc_drop_points,4) : 'N/A'}</strong><small>reference − current</small></article>
        <article><span>Anomaly rate</span><strong>${report.anomalies?.available ? `${fmt(report.anomalies.anomaly_rate,2)}%` : 'N/A'}</strong><small>current batch</small></article>
        <article><span>Data quality</span><strong>${fmt(qualityScore,1)}/100</strong><small>input safety score</small></article>
        <article><span>Coverage</span><strong>${fmt(coverage,1)}%</strong><small>eligible numeric fields monitored</small></article>
      </section>
      <div class="status ${esc(report.severity)}"><b>${esc(report.status).toUpperCase()}</b><span>Severity · ${esc(report.severity)}</span><span>Top signal · ${esc(causes[0]?.feature || 'None')}</span><span>Simulation ≠ production change</span></div>
      <div class="grid">${panel('Model health & calibration', modelBody)}${panel('Feature drift', driftBody)}</div>
      <div class="grid">${panel('Root-cause signals', causeBody)}${panel('Recommended actions', recommendationBody)}</div>
      <div class="grid">${panel('Data quality gates', qualityBody)}${panel('Model comparison', comparisonBody)}</div>
      ${panel('Segment health', segmentBody)}
      ${panel('Controlled what-if lab', `<div class="buttons"><button data-a="fill_missing">Handle missing data</button><button data-a="investigate_drift">Investigate drift</button><button data-a="retrain_model">Retraining gate</button><button data-a="rollback_model">Rollback gate</button><button data-a="quarantine_segment">Quarantine gate</button><button id="recovery">Simulate recovery</button></div><div id="lab" class="output">Select a scenario.</div>`)}
      ${panel('Blast radius & lineage', `<div class="lineage">Data Source → Feature Pipeline → Model → Decision Workflow</div><small>Affected assets: ${(impact.affected_assets || []).map(esc).join(' · ')}</small>`)}
      <footer>ModelGuard AI · ML Observability Lab · v1.0.0 · Local decision support only.</footer>
    </main>`;
  bind();
}

function fileMessage() {
  const a = $('#ref').files[0];
  const b = $('#cur').files[0];
  $('#msg').textContent = `${a?.name || 'Reference not selected'} | ${b?.name || 'Current not selected'}`;
}

function bind() {
  $('#demo').onclick = async () => {
    try { render(await api('/api/demo', {method:'POST'})); } catch (e) { alert(e.message); }
  };
  $('#refresh').onclick = async () => {
    try { render(await api('/api/scan')); } catch (e) { alert(e.message); }
  };
  $('#analyze').onclick = async () => {
    const reference = $('#ref').files[0];
    const current = $('#cur').files[0];
    if (!reference || !current) { $('#msg').textContent = 'Select both Reference and Current CSV files.'; return; }
    const maxBytes = 10 * 1024 * 1024;
    if (reference.size > maxBytes || current.size > maxBytes) { $('#msg').textContent = 'Each CSV must be 10 MiB or smaller.'; return; }
    const form = new FormData();
    form.append('reference', reference);
    form.append('current', current);
    $('#msg').textContent = 'Analyzing…';
    const button = $('#analyze'); button.disabled = true; button.textContent = 'Analyzing…';
    try { render(await api('/api/analyze', {method:'POST', body:form})); $('#msg').textContent = 'Analysis complete.'; }
    catch (e) { $('#msg').textContent = `Analysis failed: ${e.message}`; }
    finally { button.disabled = false; button.textContent = 'Analyze datasets'; }
  };
  $('#ref').onchange = fileMessage;
  $('#cur').onchange = fileMessage;
  document.querySelectorAll('[data-a]').forEach((button) => {
    button.onclick = async () => {
      const output = $('#lab');
      try {
        const result = await api(`/api/what-if?action=${encodeURIComponent(button.dataset.a)}`, {method:'POST'});
        output.innerHTML = `<b>${esc(String(result.action || '').replaceAll('_',' '))}</b><br>${esc(result.message)}<br>Gate: ${esc(result.gate || 'estimate')}`;
      } catch (e) { output.textContent = e.message; }
    };
  });
  $('#recovery').onclick = async () => {
    try {
      const result = await api('/api/recovery?improvement=.65', {method:'POST'});
      $('#lab').textContent = result.available ? `AUC ${fmt(result.before_auc,4)} → ${fmt(result.simulated_post_maintenance_auc,4)}. ${result.interpretation}` : result.message;
    } catch (e) { $('#lab').textContent = e.message; }
  };
}

(async () => {
  try {
    await api('/api/health');
    render(await api('/api/scan'));
  } catch (e) {
    $('#app').innerHTML = `<main class="failure"><h1>Local service not ready</h1><p>${esc(e.message)}</p><code>Run run_modelguard.bat</code></main>`;
  }
})();
