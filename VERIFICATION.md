# ModelGuard AI — Release Verification (1.0.0)

## Automated checks run for this package

- `python -m pytest -q`: **22 passed**.
- `python -m compileall -q backend core tests`: passed.
- `node --check frontend/app.js`: passed.
- API tests cover health/version, demo loading, upload/analysis, preview, readiness, HTML/JSON reports, what-if actions, recovery validation, invalid file types, empty/malformed CSVs, no shared schema, per-file upload limits, duplicate headers, internal-error redaction, ID exclusion, coverage, drift direction, one-class target handling, and all-null reference features.

## API/data scenario smoke checks

A separate FastAPI `TestClient` smoke pass completed **25/25 checks**: UI assets, health/version, demo/scan/preview/readiness, HTML/JSON report downloads, all five what-if actions, recovery simulation and invalid recovery input, unknown API path, invalid/empty/malformed CSV rejection, and upload of the healthy, drifted, and quality-issue scenarios. Scenario readiness values observed: healthy 89.1/100, drifted 61/100, quality-issue 41/100.

## Safety and release changes

- Release version set to `1.0.0` consistently in API metadata, analysis output, frontend and package metadata.
- Added a 10 MiB per-file CSV upload cap with chunked reads and an explicit `413` response.
- CSV headers are checked before Pandas can normalize duplicate names; blank and duplicate headers are rejected.
- Unexpected analysis failures return a generic client message; traceback details are written to the local server logger instead of being returned to the browser.
- Client-supplied filenames are treated as labels only, stripped to a basename, and length-limited.
- Added a browser-side file-size check and a disabled/loading state during analysis.
- Documentation now explicitly states the single-user in-memory design and non-production boundary.

## Verification limits

This package was tested in the available Python/Linux environment. The Windows `.bat`/PowerShell launcher, dependency installation on a clean Windows machine, and visual interaction in a real Windows browser were not executed here. Those are the remaining environment-specific checks; no claim is made that every Windows configuration is verified.

## Product boundary

ModelGuard AI is local evidence-driven decision support. It does not deploy models, mutate production data, execute production rollback/retraining actions, or certify a model as safe. Treat metrics and recommendations as investigation signals.
