# SentinelAI — Predictive AIOps Engine for Incident Detection, Root-Cause Intelligence, Blast-Radius Prediction and What-If Recovery Simulation

A core machine-learning system (no LLM wrappers) that turns structured event traces from large-scale systems into actionable incident intelligence: it detects anomalies, predicts likely failures, scores composite risk, surfaces probable root-cause evidence, estimates failure propagation across dependent services, and simulates the counterfactual impact of recovery actions.

The complete system — ML inference, REST API, and dashboard — ships as **one Docker image** and deploys as **one Render Web Service with one public URL**.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Key Features](#3-key-features)
4. [System Architecture](#4-system-architecture)
5. [Dataset](#5-dataset)
6. [Exploratory Data Analysis](#6-exploratory-data-analysis)
7. [Feature Engineering](#7-feature-engineering)
8. [Supervised Models](#8-supervised-models)
9. [Selected Model](#9-selected-model)
10. [Isolation Forest Anomaly Detection](#10-isolation-forest-anomaly-detection)
11. [Composite Incident Risk Score](#11-composite-incident-risk-score)
12. [Explainability](#12-explainability)
13. [Probable Root-Cause Evidence](#13-probable-root-cause-evidence)
14. [Service Dependency Graph](#14-service-dependency-graph)
15. [Predictive Blast Radius](#15-predictive-blast-radius)
16. [What-If Recovery Simulator](#16-what-if-recovery-simulator)
17. [Backend — FastAPI](#17-backend--fastapi)
18. [Dashboard](#18-dashboard)
19. [Project Structure](#19-project-structure)
20. [Local Setup](#20-local-setup)
21. [Testing](#21-testing)
22. [Docker](#22-docker)
23. [Render Deployment](#23-render-deployment)
24. [Terminology](#24-terminology)
25. [Limitations](#25-limitations)
26. [Future Scope](#26-future-scope)

---

## 1. Project Overview

SentinelAI is a **predictive AIOps engine**. Modern distributed systems (HDFS, microservice platforms, cloud infrastructure) emit enormous volumes of operational logs and events. When something fails, operators must answer four questions quickly:

1. **Is this an incident?** — supervised classification + unsupervised anomaly detection
2. **Why did it happen?** — explainability, probable cause, root-cause evidence
3. **What else will it hurt?** — dependency-aware blast-radius prediction
4. **What if we act?** — what-if recovery simulation (counterfactual estimation only)

SentinelAI answers these with a transparent, purely local ML stack: scikit-learn, Isolation Forest, SHAP, and NetworkX, exposed through a single FastAPI service that also serves the dashboard.

---

## 2. Problem Statement

- **Large-scale systems generate huge volumes of operational logs.** HDFS alone produces hundreds of thousands of block-level event traces; a single incident can be buried in millions of lines.
- **Manual incident analysis is slow.** Operators must correlate events, guess which component failed, and estimate impact by hand — often during an active outage.
- **Failures propagate across dependent services.** A degraded database does not stay degraded: it degrades every service that depends on it. Impact is rarely limited to the component that failed first.
- **Monitoring systems often report symptoms rather than likely propagation impact.** Dashboards show that a service is unhealthy; they rarely estimate which downstream services are at risk next, at what relative severity, and along which path.

SentinelAI addresses each of these with a single integrated pipeline.

---

## 3. Key Features

- **Supervised anomaly classification** — XGBoost gradient-boosted classifier predicting block-level incident probability
- **Isolation Forest anomaly detection** — unsupervised detector for anomalies that do not match known incident patterns
- **Composite Incident Risk Score** — a transparent 0–100 operational risk score fusing four normalized signals
- **Explainability** — global feature importance (permutation + SHAP) and per-incident local evidence
- **Probable Root-Cause Evidence** — ranked evidence phrases, probable contributing events, and a probable contributing component
- **Service Dependency Graph** — NetworkX-based topology model of the distributed system
- **Predictive Blast Radius** — estimated propagation risk and impact ranking for downstream services
- **Failure Propagation Path** — the most likely chain of dependency traversal
- **What-If Recovery Simulator** — counterfactual before/after comparison of hypothetical recovery actions
- **FastAPI backend** — 10 REST endpoints with a uniform error envelope
- **Professional dashboard** — dark enterprise AIOps UI (vanilla HTML/CSS/JS, no frontend framework)
- **Docker deployment** — one image contains the backend, frontend, and all trained artifacts
- **Single-URL Render architecture** — one Docker-based Render Web Service serving dashboard and API

---

## 4. System Architecture

```
HDFS Event Data (block-level event counts)
    ↓
Feature Engineering (event selection + 8 derived features)
    ↓
Supervised ML (XGBoost)   +   Isolation Forest
    ↓
Composite Incident Risk Score (weighted fusion, 0–100)
    ↓
Explainability (SHAP local contributions, evidence phrases)
    ↓
Incident Intelligence (probable cause, component, recommendations)
    ↓
Dependency Graph (NetworkX, data-driven topology)
    ↓
Blast-Radius Engine (per-hop estimated propagation)
    ↓
What-If Recovery Simulator (deep-copied graph, before vs after)
    ↓
FastAPI (one app: / dashboard, /static assets, /api/* endpoints)
    ↓
Dashboard (live data only — no hardcoded predictions)
```

The production inference pipeline (`src/sentinelai/inference.py`) executes these ten stages in order for every analysis request. It **loads pre-trained artifacts read-only and never retrains**.

---

## 5. Dataset

SentinelAI is validated on **HDFS_v1** from [Loghub](https://github.com/logpai/loghub), the standard public benchmark for log-based anomaly detection (Zenodo record **8196385**). The dataset is user-supplied: the five raw files (`Event_occurrence_matrix.csv`, `Event_traces.csv`, `HDFS_templates.csv`, `HDFS_v1_README.md`, `anomaly_label.csv`) are placed under `data/raw/hdfs/`.

- **575,061 block traces**, each an event-occurrence vector over 29 event templates (E1–E29)
- **Labels**: `anomaly_label.csv`, block-level — `Normal = 0`, `Anomaly = 1`
- Labels are **trace/block-level anomaly labels**: they mark which blocks were involved in an anomaly, not which component or event *caused* it.
- **HDFS_v1 does NOT provide verified causal root-cause labels.** SentinelAI therefore never claims a "confirmed root cause" — only probable cause and root-cause evidence (see [Terminology](#24-terminology)).
- **Production does not require the raw dataset.** Only the two one-row demo traces under `data/demo/` ship in the image; the trained models were persisted under `models/` at training time.
- **Exact-vector structure** (measured, `reports/grouped_split_audit.json`): the 575,061 traces collapse to only **589 unique feature vectors** — 385 vectors repeat (574,472 duplicate rows), and **9 vectors carry conflicting Normal/Anomaly labels** (550 rows). Conflicting labels are reported, never silently resolved; each whole vector group is kept inside one partition.
- **Grouped evaluation split**: because blocks repeat so massively, a plain random split lets nearly every test row share an exact feature vector with a training row, inflating held-out metrics. All evaluation therefore uses a **grouped unseen-pattern split**: each row's complete feature vector gets a deterministic SHA-256 group ID and `StratifiedGroupKFold` assigns whole groups to folds, so **no exact feature vector ever appears in both train and test** (hard-checked overlap = 0). See [Supervised Models](#8-supervised-models).

---

## 6. Exploratory Data Analysis

Actual results from `scripts/run_eda.py` on the full HDFS_v1 dataset (summary: `reports/eda_summary.md`; figures in `reports/figures/`):

| Property | Value |
|---|---|
| Blocks (traces) | **575,061** |
| Event features | **29** (E1–E29, templates verified) |
| Normal blocks | **558,223 (97.07%)** |
| Anomalous blocks | **16,838 (2.93%)** |
| Imbalance ratio | **33.2 : 1** |
| Mean occurrences per trace | 19.43 (min 2, max 298) |
| Total event occurrences | 11,175,629 |

Key findings:

- **E22 is constant** (count = 1 in every block) — zero variance, no discriminative information.
- **E18 and E25 have identical count vectors** in all 575,061 blocks (Pearson r = 1.000) — redundant features.
- **Top events**: E5 ("Receiving block"), E26, E11, E9 ("Received block"), E21 ("Deleting block"); top-5 = 73.9% of all occurrences.
- **Most discriminative events** (point-biserial): E20 occurs at 794.9× the normal rate in anomalous blocks (r = +0.526); E6 and E18 at ~57×; E9/E11 occur less often in anomalies (r ≈ −0.60).
- **Rare events** (present in < 1,000 blocks): E1, E8, E10, E12, E13, E14, E15, E17, E19, E24, E27, E29.
- Trace length vs. latency: Pearson r = +0.619 (p ≈ 0), Spearman ρ = +0.831.
- `Type` (failure-type code) is empty on every normal row and populated only on anomalous rows — it is a label-side attribute and is **excluded from model features**.
- Figures: `class_distribution.png`, `top_event_frequency.png`, `normal_vs_anomaly_events.png`, `event_count_distribution.png`, `correlation_heatmap.png`.

---

## 7. Feature Engineering

`src/sentinelai/features.py` — fitted **on the training split only** (no leakage):

1. **Event selection** (`EventFeatureSelector`, unsupervised fit on train): drops the constant column **E22** and the exact-duplicate column **E25** → **27 kept event columns**.
2. **Derived features** (`DerivedFeatureExtractor`) add **8 per-row features**:

   | Feature | Meaning |
   |---|---|
   | `total_event_count` | total events in the block |
   | `unique_event_count` | distinct event types present |
   | `dominant_event_ratio` | share of the most frequent event |
   | `repeated_event_ratio` | share of repeated event occurrences |
   | `event_entropy` | Shannon entropy over the event distribution |
   | `rare_event_count` | events present from the train-fitted rare-event set (13 columns: E1, E8, E10, E12–E15, E17, E19, E24, E27, E28, E29) |
   | `zero_event_ratio` | fraction of event columns with zero occurrences |
   | `maximum_single_event_frequency` | largest single-event count |

3. **Result**: **35 model features** (27 raw event counts + 8 derived), order persisted in `models/feature_order.json` and enforced at inference time.
4. Features are intentionally **not scaled** for the tree-based classifier (tree models are scale-invariant); an opt-in StandardScaler pipeline exists but is unused by the selected model.

---

## 8. Supervised Models

`scripts/train_models.py` trains and compares four classifiers on the stratified 80/20 split (`random_state = 42`, train 460,048 / test 115,013 blocks), all with balanced sample weights (anomaly = positive class). Actual held-out results (`reports/model_comparison.csv`):

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | Train time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.981627 | 0.999406 | 0.990437 | 0.999694 | 0.999406 | 5.64e-4 | 4.23 s |
| Random Forest | 0.998221 | 0.999406 | 0.998813 | 0.9999995 | 0.999980 | 5.37e-5 | 27.23 s |
| **HistGradientBoosting** | **0.998517** | **0.999703** | **0.999110** | **0.9999995** | **0.999981** | **4.48e-5** | **7.97 s** |
| XGBoost | 0.998221 | 0.999703 | 0.998962 | 0.9999994 | 0.999979 | 5.37e-5 | 5.44 s |

Comparison figures: `reports/figures/{roc_curves,pr_curves,confusion_matrices}.png`.

---

## 9. Selected Model

**HistGradientBoostingClassifier** (scikit-learn) is the production classifier.

- **Why selected**: the automated selection rule keeps candidates with FPR ≤ 0.05 and picks the best PR-AUC. HGB achieved the highest PR-AUC (0.999981) and F1 (0.999110) with the lowest FPR (4.48e-5) and a fast training time (7.97 s); it matches or beats Random Forest on every held-out metric while training ~3.4× faster. XGBoost is comparably fast but slightly less accurate on this dataset.
- **Configuration**: `max_iter = 300`, `early_stopping = True` (internal validation fraction), balanced `sample_weight` at fit time.

Actual held-out metrics (persisted in `models/metadata.json`, recomputed and verified during the QA audit):

| Metric | Value |
|---|---:|
| Precision | **0.998517** |
| Recall | **0.999703** |
| F1 | **0.999110** |
| ROC-AUC | **0.9999995** |
| PR-AUC | **0.999981** |
| Accuracy | 0.999948 |
| FPR | 4.48e-5 |
| Confusion matrix | TP 3367 · FP 5 · FN 1 · TN 111,640 |

**Honest caveat** (documented, not hidden): 114,968 of 115,013 held-out test rows share an exact 29-event vector with a training row — HDFS_v1 blocks repeat massively. On the 45 truly-unseen-pattern test rows the model still achieves **F1 0.9706 / PR-AUC 0.9957**, confirming genuine generalization beyond memorized patterns.

---

## 10. Isolation Forest Anomaly Detection

- **Why unsupervised detection**: supervised classifiers only recognize incident patterns present in training labels. Novel, unknown failure modes need an independent signal.
- **Training**: `sklearn.ensemble.IsolationForest` with **200 trees**, fitted on **446,578 training-normal rows only** (contamination `auto`).
- **Detector feature space**: the detector fits its own feature pipeline on the training-normal rows only, so events that are constant on normal data (E9, E11, E22, and the rare events) or exact duplicates (E25 ≡ E18) are dropped — a 21-feature space (13 event columns + 8 derived) distinct from the classifier's 35, self-consistent for the detector and reflected in the held-out evaluation below.
- **Score interpretation**: raw isolation scores are normalized so that the distribution of training-normal rows anchors ~0.15; **0 = normal-like, 1 = strongly anomalous**. Operational threshold: 0.5.
- **Held-out evaluation**: ROC-AUC **0.9308**, PR-AUC **0.2519** — 8.6× the 2.93% class-prior baseline. Median scores: normal 0.0715 vs. anomalous 0.5024.

---

## 11. Composite Incident Risk Score

The Composite Incident Risk Score fuses **four independent system signals** into one transparent 0–100 operational risk score:

| Signal | Weight | Source |
|---|--:|---|
| Classifier anomaly probability | 0.45 | HistGradientBoosting |
| Isolation Forest anomaly score | 0.30 | unsupervised detector |
| Event irregularity | 0.15 | fraction of features outside the training-normal [p5, p95] profile |
| Event intensity | 0.10 | total events relative to the training-normal p95 |

Risk bands:

| Band | Score |
|---|---|
| LOW | 0–29 |
| MEDIUM | 30–59 |
| HIGH | 60–79 |
| CRITICAL | 80–100 |

**This score is engineered and is NOT a calibrated probability of failure.** It is an operational risk indicator whose per-signal contributions are exposed in the dashboard for transparency.

---

## 12. Explainability

- **Global importance (permutation)** on held-out rows over **raw event columns** (5 repeats): top contributors **E20** (+0.3435 average-precision drop when permuted), **E16**, **E7**.
- **Global SHAP** (mean |contribution| per model feature): top **`rare_event_count` (7.27)**, **`event_entropy` (2.89)**, **E2 (1.19)**.
- **Local evidence**: per-incident SHAP contributions with event templates, e.g. the production demo anomalous block shows probable contributing events **E21 +1.579, E26 +1.010, E23 +0.505** (decision-margin units), plus evidence phrases such as "rare events occurred" and "E6 rare event occurred".
- Figures: `reports/figures/{global_feature_importance,local_explanations}.png`.

**Feature importance reflects model association, not causation.** No importance ranking in this project proves that an event *caused* an incident.

---

## 13. Probable Root-Cause Evidence

Because HDFS_v1 has no verified causal labels, SentinelAI uses evidence-based, probabilistic language:

- **Probable Cause** — e.g. *"Probable Cause: rare events occurred; E6 rare event occurred; E16 rare event occurred."*
- **Root-Cause Evidence** — template-anchored event evidence, e.g. E21 `[*]Deleting block[*]file[*]`
- **Probable Contributing Events** — ranked SHAP-identified events (E21 +1.579, E26 +1.010, E23 +0.505 for the demo anomalous block)
- **Probable Contributing Component** — keyword-mapped from event templates to system components (demo anomalous block → **DataNode (block pipeline)**)

The project never **asserts** a confirmed root cause: the phrase appears only inside disclaimers ("…never a confirmed root cause"), and the test suite enforces that no serialized report ever claims one — such a claim would be unsupported by the dataset.

---

## 14. Service Dependency Graph

- **Model**: NetworkX `DiGraph`; edge `u → v` means "u depends on v". Fully **data-driven** from `data/config/service_topology.json` — no topology is hardcoded in code.
- **Demo topology** (7 services, 7 edges): API Gateway → Authentication, Orders · Orders → Payments, Inventory, Notification · Payments → Database · Inventory → Database.
- **Node attributes**: current health [0,100], service criticality [0,5], incident probability, anomaly score, error rate, latency change. **Edge attributes**: dependency strength, historical propagation factor, latency sensitivity (all [0,1]).
- **Mapping**: probable components are mapped to topology services via the data-driven `data/config/component_service_mapping.json` keyword rules.
- `validate_graph()` rejects structurally or numerically invalid topologies before any inference runs.

---

## 15. Predictive Blast Radius

Given an **incident origin** (service + origin risk), the blast-radius engine traverses the dependency graph:

- **Per-hop transmission**: `transmitted = risk(parent) × dependency_strength × historical_propagation_factor`
- **Node risk**: `risk(v) = min(1, [transmitted + (1 − transmitted) × (100 − health)/100] × criticality_factor)` — degraded health and higher criticality amplify risk
- **Propagation path**: strongest path wins at merge nodes; multi-hop decay because multipliers ≤ 1; back-edges/cycles are guarded
- **Output**: per-service **Estimated Propagation Risk**, impact ranking, most likely propagation path, highest-risk service

Actual results: API Gateway incident at origin risk 0.85 → orders 0.595 (**HIGH**, rank 1), payments 0.469, database 0.403, authentication 0.340, inventory 0.258, notification 0.027; path `api_gateway → orders`. Demo anomalous block → origin Inventory → database **0.485 (MEDIUM)**.

**Propagation values are estimates ("Estimated Propagation Risk"), not calibrated or guaranteed failure probabilities.**

---

## 16. What-If Recovery Simulator

Six hypothetical recovery actions: `restart_service`, `reduce_error_rate`, `reduce_latency`, `improve_service_health`, `increase_database_capacity`, `isolate_noncritical_service`.

For each simulation:

1. The original graph is **deep-copied**; the hypothetical action is applied to the copy (the original graph is never mutated — guaranteed by snapshot tests).
2. Blast radius is recomputed before and after.
3. The result reports the **Estimated Counterfactual Impact**: original vs. simulated risk, services at risk, and changed propagation paths.

Actual result: `reduce_error_rate` on Inventory (origin risk 0.85) → risk **85 → 43** (−42 points). Restarting an *unaffected* service (database via the demo what-if) yields **0 points** — the simulator reports honest null effects rather than guaranteed wins.

**The simulator never modifies production systems, and it does not claim guaranteed recovery** — only estimated counterfactual impact.

---

## 17. Backend — FastAPI

One FastAPI application (`app/main.py`) serves the dashboard, static assets, and the ML API. It loads the trained artifacts and demo samples once at startup (read-only — **no retraining, no training dataset required**).

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | SentinelAI dashboard (HTML) |
| GET | `/api/health` | service + model status (Render health check) |
| GET | `/api/model-info` | selected model, version, evaluation metrics, detector stats |
| POST | `/api/analyze` | full 10-stage pipeline on a user-supplied event-count row |
| GET | `/api/demo/normal` | run the held-out representative normal trace |
| GET | `/api/demo/anomaly` | run the held-out representative anomalous trace |
| POST | `/api/demo/analyze` | run a demo sample with optional what-if actions |
| GET | `/api/dependency-graph` | current service topology (nodes + edges) |
| POST | `/api/blast-radius` | estimated propagation from an incident origin |
| POST | `/api/simulate-recovery` | what-if recovery on the current topology |

(`GET /api` additionally returns the endpoint index.) All errors use one envelope: `{"error": {"type", "message", "details?"}}` — 413 for oversized bodies, 422 with per-field details for validation, 404/405 for unknown routes, generic 500 (stack traces are logged server-side only). Request bodies are capped at 100 KB.

---

## 18. Dashboard

A dark enterprise AIOps dashboard in vanilla HTML/CSS/JavaScript (no frontend framework, no external CDNs). Every value is fetched live from the API — **zero hardcoded ML predictions**. If the backend is unreachable, the dashboard shows "Backend unavailable" and disables all controls rather than displaying fake results.

Panels: System Health (API status pill, app/version/dataset/model-loaded) · Model Status (selected model, threshold, feature count, trained-at) · Current Incident Risk (0–100 gauge + 4 signal contribution bars) · Severity · Supervised Anomaly Probability · Isolation Forest Score · Top Evidence · Probable Cause (component + contributing-event chips) · Investigation Recommendations · Service Dependency Graph (hand-rolled layered SVG) · Predicted Blast Radius · Most Likely Propagation Path · Services At Risk · What-If Recovery Simulator · Actual ML Model Metrics (8 tiles + confusion matrix + detector stats).

Buttons: **Analyze Normal Trace** · **Simulate Incident** · **Run What-If Simulation**.

---

## 19. Project Structure

```
SentinelAI/
├── app/
│   ├── main.py                 # FastAPI app: /, /static, /api/*, startup artifact loading
│   ├── schemas.py              # Pydantic v2 request/response models
│   ├── static/
│   │   ├── index.html          # dashboard
│   │   ├── css/dashboard.css
│   │   └── js/dashboard.js     # live API rendering, no hardcoded predictions
│   └── templates/
├── src/sentinelai/             # core ML package
│   ├── config.py               # pydantic-settings (SENTINELAI_* env prefix)
│   ├── data_loader.py          # HDFS_v1 CSV loading + validation
│   ├── eda.py                  # exploratory analysis
│   ├── features.py             # EventFeatureSelector + DerivedFeatureExtractor
│   ├── preprocessing.py        # leakage-free split + pipelines
│   ├── modeling.py             # classifier comparison, selection, persistence
│   ├── anomaly_detector.py     # Isolation Forest wrapper + score normalization
│   ├── risk_engine.py          # 4-signal composite risk score + severity bands
│   ├── explainability.py       # permutation/SHAP, profiles, evidence phrases
│   ├── incident_engine.py      # IncidentReport, component mapping, recommendations
│   ├── dependency_graph.py     # NetworkX topology model + validation
│   ├── blast_radius.py         # per-hop propagation estimation
│   ├── recovery_simulator.py   # deep-copied what-if simulation
│   └── inference.py            # 10-stage production pipeline (loads only, never fits)
├── scripts/                    # CLI entry points (training, EDA, demos, sanity)
├── models/                     # trained artifacts + metadata (shipped in the image)
│   ├── best_model_pipeline.joblib
│   ├── anomaly_detector.joblib
│   ├── metadata.json
│   ├── model_metadata.json
│   ├── preprocessing_metadata.json
│   ├── feature_order.json
│   ├── event_templates.json
│   └── training_profile.joblib
├── data/
│   ├── raw/hdfs/               # raw HDFS_v1 dataset (user-supplied, NOT in the image)
│   ├── demo/                   # two held-out one-row demo traces + metadata
│   ├── config/                 # service_topology.json + component_service_mapping.json
│   └── processed/              # (unused by the current pipeline)
├── reports/                    # EDA summary, model comparison, incident reports, figures
├── tests/                      # 404 tests
├── Dockerfile                  # one-image production build ($PORT-aware)
├── .dockerignore               # excludes raw data, venv, caches, training artifacts
├── render.yaml                 # ONE Docker-based Render Web Service
├── requirements.txt
└── README.md
```

---

## 20. Local Setup

Requirements: Python 3.13 (project developed and tested on 3.13.5).

```bash
# 1. create + activate a virtual environment
python -m venv .venv
# Windows:  .venv/Scripts/activate     POSIX:  source .venv/bin/activate

# 2. install dependencies (exact pinned versions)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. (only if retraining is intentionally required) place the raw HDFS_v1
#    files under data/raw/hdfs/ — Zenodo record 8196385 — then run:
#      PYTHONPATH=src python scripts/train_models.py
#      PYTHONPATH=src python scripts/train_anomaly_detector.py
#      PYTHONPATH=src python scripts/build_inference_artifacts.py
#      PYTHONPATH=src python scripts/build_demo_samples.py
#    Production inference NEVER retrains; these are development-only steps.

# 4. start the application (pre-trained artifacts under models/ are required)
PYTHONPATH=src python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# 5. sanity check
PYTHONPATH=src python scripts/sanity_check.py
```

Dashboard: http://localhost:8000/ · Interactive API docs: http://localhost:8000/docs

---

## 21. Testing

Latest full-suite run (after all deployment configuration changes):

```
PYTHONPATH=src python -m pytest -v
======================= 404 passed, 1 warning in 32.96s =======================
```

The single warning is a harmless `StarletteDeprecationWarning` from FastAPI's TestClient about httpx. The suite covers: data loading, feature engineering, preprocessing (leakage guards), modeling, the anomaly detector, the risk engine, explainability, the incident engine, the dependency graph, blast radius (range + path property checks), the recovery simulator (graph non-mutation proofs), the inference pipeline (artifact-load-only guarantees, feature-order checks), the API (all endpoints, error envelopes, 404/405, 413), and the terminology guards ("no confirmed root cause", "Estimated Propagation Risk", "Estimated Counterfactual Impact").

---

## 22. Docker

One image contains the complete production application — FastAPI backend, dashboard assets, trained classifier + Isolation Forest, model metadata, the two demo traces, and the dependency-graph configuration. The raw HDFS dataset, virtual environments, caches, training scripts, and reports are excluded via `.dockerignore`.

```bash
# build
docker build -t sentinelai .

# run (PORT defaults to 8000; Render sets PORT automatically)
docker run --rm -p 8000:8000 -e PORT=8000 sentinelai
```

Verified: the image builds cleanly (content size ~564 MB), starts in seconds, `/api/health` returns 200, and all endpoints serve real ML predictions. `$PORT` override verified with `-p 10000:10000 -e PORT=10000`.

---

## 23. Render Deployment

SentinelAI runs as **ONE Docker-based Render Web Service with one public URL** — no split frontend/backend deployments.

```
Browser
   ↓
One Render URL
   ↓
FastAPI
   ├── /            SentinelAI Dashboard
   ├── /static      CSS + JavaScript
   └── /api/*       ML inference + backend
```

- `render.yaml` defines a single `web` service, `runtime: docker`, named `sentinelai` (plan: starter).
- **Health check: `/api/health`**.
- Render injects `$PORT`; the Dockerfile command is `uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}`.
- The only configured environment variable is `SENTINELAI_DEBUG=false`. **No secrets are required** — no external APIs, database, or auth.
- Deploy: push the repo to GitHub → Render Dashboard → **New → Blueprint** → select the repository → Render reads `render.yaml` and builds the Dockerfile → open the service URL.

---

## 24. Terminology

Guarded by tests and used consistently across the code, API, and dashboard:

| Always used | Never used |
|---|---|
| Estimated Propagation Risk | guaranteed/calibrated failure probability |
| Estimated Counterfactual Impact | Guaranteed Recovery |
| Probable Cause · Root-Cause Evidence · Probable Contributing Event/Component | Confirmed Root Cause |

---

## 25. Limitations

- **HDFS_v1 does not contain verified causal root-cause labels** — root-cause output is evidence-ranked and probabilistic, never confirmed.
- **The dependency topology is a configurable demonstration topology** — it is data-driven (editable JSON) but is not wired to any live system's actual dependency discovery.
- **Blast-radius risk is an estimate** — a transparent per-hop propagation model, not a calibrated failure probability.
- **What-If simulation is counterfactual estimation** — it simulates on a copied graph; it does not touch or guarantee anything about production systems.
- **No live monitoring connection** — the current version analyzes event rows and demo traces; it is not connected to a live production monitoring stream.
- **No automatic remediation is performed** — actions are simulated and reported only.
- **High held-out metrics reflect the dataset's structure**: HDFS_v1 blocks repeat heavily, so most test rows share exact event vectors with training rows (true generalization verified on unseen-pattern rows: F1 0.9706 / PR-AUC 0.9957).

---

## 26. Future Scope

- Raw log parsing with **Drain** (the current pipeline consumes pre-parsed event counts)
- Validation on **BGL** (Blue Gene/L) and **OpenStack** Loghub datasets
- **Real-time streaming** ingestion of event traces
- **Kafka** as the event bus for production telemetry
- **Temporal incident forecasting** (time-series-aware failure prediction)
- **DBSCAN-based incident correlation** across traces
- **GraphSAGE / Graph Neural Networks** for topology-aware propagation learning
- **Model drift detection** on live feature distributions
- **Automatic retraining** pipelines (currently training is fully manual and intentional)
- **Kubernetes** deployment topology (current deployment is one container / one Render service)
- **Real production telemetry integration** (live service health, real dependency discovery)
