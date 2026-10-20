/* SentinelAI dashboard — every value on this page comes live from the
 * FastAPI backend. Nothing here is a hardcoded prediction result.
 */
"use strict";

const API = "/api";

const SEVERITY_CLASS = {
  LOW: "low",
  MEDIUM: "medium",
  HIGH: "high",
  CRITICAL: "critical",
};

const CONTRIBUTION_KEYS = [
  "classifier",
  "anomaly_detector",
  "event_irregularity",
  "event_intensity",
];

const state = {
  graph: null,
  lastBlast: null, // {origin, originRisk} from the most recent analysis
  blastSkipped: false,
  busy: false,
  backendDown: false,
};

const $ = (id) => document.getElementById(id);

/* ---------------- API access ---------------- */

async function apiFetch(path, options = {}) {
  let response;
  try {
    response = await fetch(path, options);
  } catch (error) {
    markBackendDown();
    throw error;
  }
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      if (body && body.error && body.error.message) {
        message = body.error.message;
      }
    } catch (_) {
      /* non-JSON error body; keep the status message */
    }
    throw new ApiError(response.status, message);
  }
  markBackendUp();
  return response.json();
}

class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

function markBackendDown() {
  state.backendDown = true;
  $("backend-banner").hidden = false;
  $("health-pill").textContent = "API · down";
  $("health-pill").className = "pill pill-down";
  setActionsEnabled(false);
}

function markBackendUp() {
  state.backendDown = false;
  $("backend-banner").hidden = true;
  setActionsEnabled(true);
}

function setActionsEnabled(enabled) {
  const effective = enabled && !state.backendDown;
  for (const id of ["btn-normal", "btn-incident", "btn-simulate"]) {
    $(id).disabled = !effective || state.busy;
  }
  const serviceSelect = $("sim-service");
  const actionSelect = $("sim-action");
  serviceSelect.disabled = !effective || !state.graph;
  actionSelect.disabled = !effective || !state.graph;
}

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => {
    toast.hidden = true;
  }, 6000);
}

/* ---------------- formatting ---------------- */

function fmtProbability(p) {
  if (typeof p !== "number" || Number.isNaN(p)) return "—";
  if (p >= 0.0001) {
    const pct = p * 100;
    return `${pct.toFixed(pct >= 1 ? 2 : 4)}%`;
  }
  return p.toExponential(3);
}

function fmtScore(v, digits = 3) {
  return typeof v === "number" && Number.isFinite(v)
    ? v.toFixed(digits)
    : "—";
}

function fmtTrainedAt(iso) {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString();
}

function severityClass(severity) {
  return SEVERITY_CLASS[severity] || "idle";
}

function healthColor(health) {
  if (typeof health !== "number" || Number.isNaN(health)) return "#5d6c85";
  if (health >= 70) return "#34d399";
  if (health >= 40) return "#fbbf24";
  return "#f87171";
}

/* ---------------- section renderers ---------------- */

function renderHealth(data) {
  $("health-pill").textContent = "API · ok";
  $("health-pill").className = "pill pill-ok";
  $("model-pill").textContent = `model · ${data.model_version || "—"}`;
  $("model-pill").className = "pill pill-neutral";
  $("sys-status").textContent = data.status === "ok" ? "operational" : data.status;
  $("sys-app").textContent = data.app_name || "—";
  $("sys-version").textContent = data.version || "—";
  $("sys-dataset").textContent = data.dataset || "—";
  $("sys-model-loaded").textContent = data.model_loaded ? "yes" : "no";
}

function renderModelInfo(data) {
  $("model-name").textContent = data.selected_model || "—";
  $("model-version").textContent = data.model_version || "—";
  $("model-threshold").textContent =
    typeof data.classifier_threshold === "number"
      ? data.classifier_threshold.toFixed(3)
      : "—";
  $("model-features").textContent =
    typeof data.feature_count === "number" ? String(data.feature_count) : "—";
  $("model-trained").textContent = fmtTrainedAt(data.trained_at);
  renderMetrics(data);
}

function renderMetrics(data) {
  const metrics = data.evaluation_metrics || {};
  const tiles = [
    ["F1 score", metrics.f1],
    ["Precision", metrics.precision],
    ["Recall", metrics.recall],
    ["Accuracy", metrics.accuracy],
    ["ROC AUC", metrics.roc_auc],
    ["PR AUC", metrics.pr_auc],
    ["False positive rate", metrics.fpr],
    ["Train time (s)", metrics.train_seconds],
  ];
  const container = $("metric-tiles");
  container.textContent = "";
  for (const [label, value] of tiles) {
    const tile = document.createElement("div");
    tile.className = "metric-tile";
    const name = document.createElement("div");
    name.className = "m-label";
    name.textContent = label;
    const val = document.createElement("div");
    val.className = "m-value";
    val.textContent =
      typeof value === "number" && Number.isFinite(value)
        ? value > 0 && value < 0.001
          ? value.toExponential(2)
          : value.toFixed(4)
        : "—";
    tile.append(name, val);
    container.append(tile);
  }

  const confusion = $("metric-confusion");
  confusion.textContent = "";
  for (const [label, key] of [
    ["True positives", "tp"],
    ["False positives", "fp"],
    ["False negatives", "fn"],
    ["True negatives", "tn"],
  ]) {
    const cell = document.createElement("div");
    cell.className = "conf-cell";
    const name = document.createElement("span");
    name.className = "c-label";
    name.textContent = label;
    const val = document.createElement("span");
    val.className = "c-value";
    val.textContent = metrics[key] !== undefined ? String(metrics[key]) : "—";
    cell.append(name, val);
    confusion.append(cell);
  }

  const detector = data.anomaly_detector || {};
  const adMetrics = detector.evaluation_metrics || {};
  const ad = $("ad-metrics");
  ad.textContent = "";
  for (const [label, value, formatter] of [
    ["ROC AUC (held-out)", adMetrics.roc_auc, (v) => v.toFixed(4)],
    ["PR AUC (held-out)", adMetrics.pr_auc, (v) => v.toFixed(4)],
    ["Threshold", detector.threshold, (v) => v.toFixed(3)],
    ["Trees", detector.n_estimators, String],
    ["Trained on normal rows", detector.trained_on_normal_rows, (v) => v.toLocaleString()],
  ]) {
    const row = document.createElement("div");
    row.className = "kv-row";
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.className = "value mono";
    dd.textContent =
      value !== undefined && value !== null && formatter ? formatter(value) : "—";
    row.append(dt, dd);
    ad.append(row);
  }
}

function renderRisk(data) {
  const risk = data.composite_risk_score || {};
  const score = risk.risk_score;
  const severity = risk.severity || "";
  const cls = severityClass(severity);

  const gauge = $("risk-gauge");
  gauge.className = `gauge gauge-${cls}`;
  gauge.style.setProperty("--risk", Math.min(100, Math.max(0, score ?? 0)));
  $("risk-score").textContent = score !== undefined ? String(score) : "—";
  const pill = $("risk-severity");
  pill.className = `severity severity-${cls}`;
  pill.textContent = severity || "—";

  const contributions = risk.contributions || {};
  const total = score || 0;
  for (const key of CONTRIBUTION_KEYS) {
    const value = contributions[key];
    const bar = $(`contrib-${key}`);
    const label = $(`contrib-${key}-val`);
    if (typeof value === "number") {
      const share = total > 0 ? Math.min(100, (value / total) * 100) : 0;
      bar.style.width = `${share}%`;
      label.textContent = String(value);
    } else {
      bar.style.width = "0";
      label.textContent = "—";
    }
  }
  $("risk-caveat").textContent = risk.caveat || "—";
}

function renderIncident(data) {
  const incident = data.incident_intelligence || {};
  $("risk-block").textContent = data.block_id || "—";

  const probability = incident.classifier_probability;
  $("prob-value").textContent = fmtProbability(probability);
  const probBar = $("prob-bar");
  probBar.style.width =
    typeof probability === "number"
      ? `${Math.min(100, Math.max(0, probability * 100))}%`
      : "0";

  const anomalyScore = incident.anomaly_score;
  $("if-value").textContent = fmtScore(anomalyScore);
  const ifBar = $("if-bar");
  ifBar.style.width =
    typeof anomalyScore === "number"
      ? `${Math.min(100, Math.max(0, anomalyScore * 100))}%`
      : "0";

  fillList($("evidence-list"), incident.top_evidence || []);
  renderCause(incident);
  fillList($("recs-list"), incident.investigation_recommendations || []);
}

function renderCause(incident) {
  const component = incident.probable_component || "—";
  $("cause-component").textContent = component;
  $("cause-text").textContent = incident.probable_cause || "—";

  const row = $("cause-events");
  row.textContent = "";
  for (const event of incident.probable_contributing_events || []) {
    const chip = document.createElement("span");
    chip.className = "event-chip";
    chip.title = event.template || "";
    chip.textContent =
      `${event.event_id || "event"} ` +
      (typeof event.contribution === "number"
        ? `${event.contribution >= 0 ? "+" : ""}${event.contribution.toFixed(3)}`
        : "");
    row.append(chip);
  }
}

function renderBlast(data) {
  const blast = data.blast_radius_prediction;
  if (!blast || blast.status !== "computed") {
    state.lastBlast = null;
    state.blastSkipped = Boolean(blast);
    $("blast-status").textContent = blast ? "skipped" : "—";
    $("blast-origin").textContent = blast && blast.reason ? blast.reason : "—";
    $("blast-origin-risk").textContent = "—";
    $("blast-highest").textContent = "—";
    fillPath([]);
    fillAtRisk([]);
    clearGraphHighlight();
    return;
  }
  state.blastSkipped = false;
  state.lastBlast = {
    origin: blast.probable_origin?.id ?? null,
    originRisk: blast.probable_origin?.incident_risk ?? null,
  };
  $("blast-status").textContent = "estimated propagation risk";
  $("blast-origin").textContent = blast.probable_origin
    ? `${blast.probable_origin.name} (${blast.probable_origin.id})`
    : "—";
  $("blast-origin-risk").textContent =
    typeof blast.probable_origin?.incident_risk === "number"
      ? fmtProbability(blast.probable_origin.incident_risk)
      : "—";
  $("blast-highest").textContent = blast.highest_risk_service || "—";
  fillPath(blast.most_likely_propagation_path || []);
  fillAtRisk(blast.services_at_risk || []);
  highlightGraph(blast);
}

function fillList(container, items) {
  container.textContent = "";
  if (!items.length) {
    const item = document.createElement("li");
    item.className = "placeholder";
    item.textContent = "—";
    container.append(item);
    return;
  }
  for (const text of items) {
    const item = document.createElement("li");
    item.textContent = text;
    container.append(item);
  }
}

function fillPath(path) {
  const flow = $("path-flow");
  flow.textContent = "";
  if (!path.length) {
    const placeholder = document.createElement("span");
    placeholder.className = "placeholder";
    placeholder.textContent = state.blastSkipped ? "no incident origin" : "—";
    flow.append(placeholder);
    return;
  }
  path.forEach((step, index) => {
    if (index > 0) {
      const arrow = document.createElement("span");
      arrow.className = "path-arrow";
      arrow.textContent = "→";
      flow.append(arrow);
    }
    const chip = document.createElement("span");
    chip.className = "path-step";
    chip.textContent = step;
    flow.append(chip);
  });
}

function fillAtRisk(services) {
  const grid = $("atrisk-grid");
  grid.textContent = "";
  if (!services.length) {
    const placeholder = document.createElement("span");
    placeholder.className = "placeholder";
    placeholder.textContent = "—";
    grid.append(placeholder);
    return;
  }
  for (const service of services) {
    const card = document.createElement("div");
    card.className = "atrisk-card";
    const name = document.createElement("div");
    name.className = "name";
    name.textContent = `${service.service_name || service.service_id || "—"} ` +
      `(${service.service_id || ""})`;
    const riskLine = document.createElement("div");
    riskLine.className = "risk-line";
    const riskValue = document.createElement("span");
    riskValue.className = "risk-val";
    riskValue.textContent = fmtProbability(service.estimated_propagation_risk);
    const tag = document.createElement("span");
    tag.className = `risk-tag severity-${severityClass(service.risk_level)}`;
    tag.textContent = service.risk_level || "—";
    riskLine.append(riskValue, tag);
    const dist = document.createElement("div");
    dist.className = "dist";
    dist.textContent =
      `dependency distance ${service.dependency_distance ?? "—"}` +
      (service.relative_impact_order !== undefined
        ? ` · impact order #${service.relative_impact_order}`
        : "");
    card.append(name, riskLine, dist);
    grid.append(card);
  }
}

/* ---------------- dependency graph (vanilla SVG) ---------------- */

function renderGraph(data) {
  state.graph = data;
  const serviceMap = new Map();
  for (const service of data.services) {
    serviceMap.set(service.id, service);
  }
  const dependents = new Map(); // node -> services that depend on it
  const dependencies = new Map(); // service -> nodes it depends on
  for (const service of data.services) {
    dependents.set(service.id, []);
    dependencies.set(service.id, []);
  }
  for (const edge of data.dependencies) {
    dependents.get(edge.depends_on).push(edge.service_id);
    dependencies.get(edge.service_id).push(edge.depends_on);
  }

  const layers = new Map();
  const visiting = new Set();
  const layerOf = (node) => {
    if (layers.has(node)) return layers.get(node);
    if (visiting.has(node)) return 0; // cycle guard
    visiting.add(node);
    let layer = 0;
    for (const dependent of dependents.get(node) || []) {
      layer = Math.max(layer, layerOf(dependent) + 1);
    }
    visiting.delete(node);
    layers.set(node, layer);
    return layer;
  };
  for (const service of data.services) layerOf(service.id);

  const columns = new Map(); // layer -> [node ids in order]
  for (const service of data.services) {
    const layer = layers.get(service.id);
    if (!columns.has(layer)) columns.set(layer, []);
    columns.get(layer).push(service.id);
  }
  const layerCount = Math.max(1, columns.size);

  const width = 760;
  const height = 360;
  const marginX = 90;
  const marginY = 60;
  const nodeW = 128;
  const nodeH = 46;
  const positions = new Map();
  const sortedLayers = [...columns.keys()].sort((a, b) => a - b);

  sortedLayers.forEach((layer, index) => {
    const nodes = columns.get(layer);
    const x = layerCount === 1
      ? width / 2
      : marginX + (index * (width - 2 * marginX)) / (layerCount - 1);
    const usable = height - 2 * marginY;
    const step = nodes.length > 1 ? usable / (nodes.length - 1) : 0;
    nodes.forEach((nodeId, nodeIndex) => {
      const y = nodes.length === 1
        ? height / 2
        : marginY + nodeIndex * step;
      positions.set(nodeId, { x, y });
    });
  });

  const svg = $("graph-svg");
  svg.textContent = "";
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
  defs.innerHTML = `
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#38bdf8"/>
    </marker>`;
  svg.append(defs);

  const edgeGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
  const nodeGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
  svg.append(edgeGroup, nodeGroup);

  for (const edge of data.dependencies) {
    const from = positions.get(edge.service_id);
    const to = positions.get(edge.depends_on);
    if (!from || !to) continue;
    const midX = (from.x + to.x) / 2;
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute(
      "d",
      `M ${from.x - nodeW / 2} ${from.y} ` +
        `C ${midX} ${from.y}, ${midX} ${to.y}, ${to.x + nodeW / 2} ${to.y}`,
    );
    path.setAttribute("class", "graph-edge");
    path.setAttribute("marker-end", "url(#arrow)");
    path.dataset.edgeKey = `${edge.service_id}|${edge.depends_on}`;
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    title.textContent =
      `${edge.service_id} depends on ${edge.depends_on} · ` +
      `strength ${edge.dependency_strength ?? "—"} · ` +
      `propagation factor ${edge.historical_propagation_factor ?? "—"} · ` +
      `latency sensitivity ${edge.latency_sensitivity ?? "—"}`;
    path.append(title);
    edgeGroup.append(path);
  }

  const nodeElements = new Map();
  for (const service of data.services) {
    const pos = positions.get(service.id);
    if (!pos) continue;
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    group.setAttribute("class", "graph-node");
    group.dataset.nodeId = service.id;

    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", pos.x - nodeW / 2);
    rect.setAttribute("y", pos.y - nodeH / 2);
    rect.setAttribute("width", nodeW);
    rect.setAttribute("height", nodeH);
    rect.setAttribute("rx", "9");

    const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    dot.setAttribute("cx", pos.x - nodeW / 2 + 16);
    dot.setAttribute("cy", pos.y);
    dot.setAttribute("r", "5");
    dot.setAttribute("fill", healthColor(service.current_health));

    const name = document.createElementNS("http://www.w3.org/2000/svg", "text");
    name.setAttribute("x", pos.x - nodeW / 2 + 28);
    name.setAttribute("y", pos.y - 3);
    name.setAttribute("class", "graph-node-name");
    name.textContent = service.name || service.id;

    const health = document.createElementNS("http://www.w3.org/2000/svg", "text");
    health.setAttribute("x", pos.x - nodeW / 2 + 28);
    health.setAttribute("y", pos.y + 15);
    health.setAttribute("class", "graph-node-sub");
    health.textContent = `health ${
      typeof service.current_health === "number"
        ? `${service.current_health.toFixed(0)}%`
        : "—"
    } · crit ${service.service_criticality ?? "—"}`;

    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    title.textContent =
      `${service.name} (${service.id}) · ` +
      `health ${service.current_health ?? "—"} · ` +
      `criticality ${service.service_criticality ?? "—"} · ` +
      `incident probability ${service.incident_probability ?? "—"} · ` +
      `anomaly score ${service.anomaly_score ?? "—"} · ` +
      `error rate ${service.error_rate ?? "—"} · ` +
      `latency change ${service.latency_change ?? "—"}`;

    group.append(rect, dot, name, health, title);
    nodeGroup.append(group);
    nodeElements.set(service.id, group);
  }

  state.graphElements = { nodeElements, edgeGroup, positions };
  setActionsEnabled(true);
}

function clearGraphHighlight() {
  const { nodeElements, edgeGroup } = state.graphElements || {};
  if (!nodeElements) return;
  for (const element of nodeElements.values()) {
    element.classList.remove("graph-node-origin", "graph-node-risk");
  }
  for (const edge of edgeGroup.children) {
    edge.classList.remove("graph-edge-path");
  }
}

function highlightGraph(blast) {
  clearGraphHighlight();
  const { nodeElements, edgeGroup } = state.graphElements || {};
  if (!nodeElements) return;
  const origin = blast.probable_origin?.id;
  if (origin && nodeElements.has(origin)) {
    nodeElements.get(origin).classList.add("graph-node-origin");
  }
  for (const service of blast.services_at_risk || []) {
    if (nodeElements.has(service.service_id)) {
      nodeElements.get(service.service_id).classList.add("graph-node-risk");
    }
  }
  const path = blast.most_likely_propagation_path || [];
  for (let index = 0; index + 1 < path.length; index += 1) {
    const key = `${path[index]}|${path[index + 1]}`;
    for (const edge of edgeGroup.children) {
      if (edge.dataset.edgeKey === key) edge.classList.add("graph-edge-path");
    }
  }
}

/* ---------------- what-if simulator ---------------- */

function populateSimulator() {
  const select = $("sim-service");
  select.textContent = "";
  for (const service of state.graph.services) {
    const option = document.createElement("option");
    option.value = service.id;
    option.textContent = `${service.name} (${service.id})`;
    select.append(option);
  }
  if (state.lastBlast && state.lastBlast.origin) {
    select.value = state.lastBlast.origin;
  }
  setActionsEnabled(true);
}

async function runSimulation() {
  const affectedService = $("sim-service").value;
  const action = $("sim-action").value;
  if (!affectedService) {
    showToast("Select a service first.");
    return;
  }
  const payload = { action, affected_service: affectedService };
  if (state.lastBlast && state.lastBlast.origin && state.lastBlast.originRisk !== null) {
    payload.origin = state.lastBlast.origin;
    payload.origin_risk = state.lastBlast.originRisk;
  }
  $("btn-simulate").disabled = true;
  try {
    const result = await apiFetch(`${API}/simulate-recovery`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderSimulation(result);
  } catch (error) {
    if (error instanceof ApiError) showToast(error.message);
  } finally {
    setActionsEnabled(true);
  }
}

function renderSimulation(result) {
  const before = result.original_incident_risk;
  const after = result.simulated_incident_risk;
  $("sim-before").textContent = before !== undefined ? String(before) : "—";
  $("sim-after").textContent = after !== undefined ? String(after) : "—";
  $("sim-before-bar").style.width =
    before !== undefined ? `${Math.min(100, before)}%` : "0";
  $("sim-after-bar").style.width =
    after !== undefined ? `${Math.min(100, after)}%` : "0";
  $("sim-reduction").textContent =
    result.estimated_risk_reduction !== undefined
      ? `${result.estimated_risk_reduction} points`
      : "—";
  $("sim-services").textContent =
    `${result.original_services_at_risk ?? "—"} / ` +
    `${result.simulated_services_at_risk ?? "—"}`;
  const changes = result.changed_propagation_paths || [];
  $("sim-changes").textContent = changes.length
    ? changes
        .map(
          (change) =>
            `${change.service}: ` +
            `${(change.before_path || []).join(" → ") || "—"} → ` +
            `${change.after_path ? change.after_path.join(" → ") : "no longer at risk"}`,
        )
        .join(" · ")
    : "none";
  $("sim-results").hidden = false;
}

/* ---------------- analysis flows ---------------- */

async function runDemoAnalysis(sample) {
  if (state.busy) return;
  state.busy = true;
  setActionsEnabled(false);
  const note = $("actionbar-note");
  note.textContent =
    sample === "normal"
      ? "Running the held-out normal trace through the 10-stage pipeline..."
      : "Running the held-out anomalous trace through the 10-stage pipeline...";
  try {
    const result = await apiFetch(`${API}/demo/analyze`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ sample }),
    });
    renderAnalysis(result);
    note.textContent =
      sample === "normal"
        ? "Normal trace analyzed (held-out) — low risk, no incident origin."
        : "Incident simulated from a held-out anomalous trace.";
  } catch (error) {
    if (error instanceof ApiError) {
      note.textContent = `Analysis failed: ${error.message}`;
      showToast(error.message);
    } else {
      note.textContent = "Analysis unavailable — backend unreachable.";
    }
  } finally {
    state.busy = false;
    setActionsEnabled(true);
  }
}

function renderAnalysis(result) {
  renderRisk(result);
  renderIncident(result);
  renderBlast(result);
  populateSimulator();
}
/* ---------------- live service monitor ---------------- */

const MONITOR_STORAGE_KEY = "sentinelai.monitoredServices.v1";
const MAX_MONITORED_SERVICES = 10;

let monitoredServices = [];

function loadMonitoredServices() {
  try {
    const saved = JSON.parse(localStorage.getItem(MONITOR_STORAGE_KEY) || "[]");

    monitoredServices = Array.isArray(saved)
      ? saved.filter(
          (item) =>
            item &&
            typeof item.id === "string" &&
            typeof item.name === "string" &&
            typeof item.url === "string"
        )
      : [];
  } catch (_) {
    monitoredServices = [];
  }
}

function saveMonitoredServices() {
  localStorage.setItem(
    MONITOR_STORAGE_KEY,
    JSON.stringify(monitoredServices)
  );
}

function monitorStatusClass(status) {
  if (status === "healthy") return "low";
  if (status === "degraded") return "medium";
  if (status === "down") return "critical";
  return "idle";
}

function formatMonitorTime(value) {
  if (!value) return "—";

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return "—";
  }

  return date.toLocaleTimeString();
}

function renderMonitoredServices() {
  const grid = $("monitor-grid");
  const empty = $("monitor-empty");

  grid.textContent = "";
  empty.hidden = monitoredServices.length > 0;

  for (const service of monitoredServices) {
    const card = document.createElement("div");
    card.className = "atrisk-card";

    const name = document.createElement("div");
    name.className = "name";
    name.textContent = service.name;

    const url = document.createElement("div");
    url.className = "meta-line";
    url.style.wordBreak = "break-all";
    url.textContent = service.url;

    const statusRow = document.createElement("div");
    statusRow.className = "risk-line";
    statusRow.style.marginTop = "12px";

    const statusText = document.createElement("span");
    statusText.className = "risk-val";

    if (service.checking) {
      statusText.textContent = "Checking...";
    } else {
      statusText.textContent =
        service.status === "healthy"
          ? "Operational"
          : service.status === "degraded"
            ? "Degraded"
            : service.status === "down"
              ? "Unavailable"
              : "Not checked";
    }

    const tag = document.createElement("span");
    tag.className =
      `risk-tag severity-${monitorStatusClass(service.status)}`;

    tag.textContent = service.checking
      ? "CHECKING"
      : (service.status || "UNKNOWN").toUpperCase();

    statusRow.append(statusText, tag);

    const details = document.createElement("div");
    details.className = "meta-line";
    details.style.marginTop = "12px";

    const httpStatus =
      service.http_status !== null &&
      service.http_status !== undefined
        ? service.http_status
        : "—";

    const latency =
      typeof service.latency_ms === "number"
        ? `${service.latency_ms.toFixed(2)} ms`
        : "—";

    details.textContent =
      `HTTP ${httpStatus} · ${latency} · Last checked ${formatMonitorTime(
        service.checked_at
      )}`;

    const error = document.createElement("div");
    error.className = "meta-line";
    error.style.marginTop = "8px";
    error.textContent = service.error
      ? `Error: ${service.error}`
      : "";

    const actions = document.createElement("div");
    actions.style.display = "flex";
    actions.style.gap = "8px";
    actions.style.flexWrap = "wrap";
    actions.style.marginTop = "14px";

    const checkButton = document.createElement("button");
    checkButton.className = "btn btn-outline";
    checkButton.type = "button";
    checkButton.dataset.action = "check";
    checkButton.dataset.id = service.id;
    checkButton.textContent = service.checking
      ? "Checking..."
      : "Check Now";
    checkButton.disabled = Boolean(service.checking);
      const detailsButton = document.createElement("button");
    detailsButton.className = "btn btn-outline";
    detailsButton.type = "button";
    detailsButton.dataset.action = "details";
    detailsButton.dataset.id = service.id;
    detailsButton.textContent = "View Full Details";
    const removeButton = document.createElement("button");
    removeButton.className = "btn btn-outline";
    removeButton.type = "button";
    removeButton.dataset.action = "remove";
    removeButton.dataset.id = service.id;
    removeButton.textContent = "Remove";

    actions.append(
    checkButton,
    detailsButton,
    removeButton
);

    card.append(
      name,
      url,
      statusRow,
      details,
      error,
      actions
    );

    grid.append(card);
  }
}

async function checkMonitoredService(id) {
  const service = monitoredServices.find(
    (item) => item.id === id
  );

  if (!service || service.checking) return;

  service.checking = true;
  renderMonitoredServices();

  try {
    const result = await apiFetch(
      `${API}/service-monitor/check`,
      {
        method: "POST",
        headers: {
          "content-type": "application/json",
        },
        body: JSON.stringify({
          name: service.name,
          url: service.url,
        }),
      }
    );

    service.status = result.status;
    service.http_status = result.http_status;
    service.latency_ms = result.latency_ms;
    service.error = result.error;
    service.checked_at = result.checked_at;
    service.website = result.website || null;
    service.network = result.network || null;
    service.ssl = result.ssl || null;
    service.security = result.security || null;
  } catch (error) {
    service.status = "down";
    service.http_status = null;
    service.latency_ms = null;
    service.error =
      error instanceof Error
        ? error.message
        : "Health check failed.";
    service.checked_at = new Date().toISOString();
  } finally {
    service.checking = false;
    saveMonitoredServices();
    renderMonitoredServices();
  }
}

async function addMonitoredService() {
  const name = $("monitor-name").value.trim();
  const url = $("monitor-url").value.trim();

  if (!name || !url) {
    showToast("Enter both service name and URL.");
    return;
  }

  let parsedUrl;

  try {
    parsedUrl = new URL(url);
  } catch (_) {
    showToast("Enter a valid website or API URL.");
    return;
  }

  if (!["http:", "https:"].includes(parsedUrl.protocol)) {
    showToast("Only HTTP and HTTPS URLs are allowed.");
    return;
  }

  if (monitoredServices.length >= MAX_MONITORED_SERVICES) {
    showToast(
      `Maximum ${MAX_MONITORED_SERVICES} services can be monitored in this browser.`
    );
    return;
  }

  const normalizedUrl = parsedUrl.toString();

  if (
    monitoredServices.some(
      (service) => service.url === normalizedUrl
    )
  ) {
    showToast("This URL is already being monitored.");
    return;
  }

  const id =
    window.crypto && typeof window.crypto.randomUUID === "function"
      ? window.crypto.randomUUID()
      : `${Date.now()}-${Math.random()
          .toString(16)
          .slice(2)}`;

  monitoredServices.push({
    id,
    name,
    url: normalizedUrl,
    status: "unknown",
    http_status: null,
    latency_ms: null,
    error: null,
    checked_at: null,
    checking: false,
  });

  saveMonitoredServices();
  renderMonitoredServices();

  $("monitor-name").value = "";
  $("monitor-url").value = "";
  $("monitor-form").hidden = true;

  await checkMonitoredService(id);
}

function removeMonitoredService(id) {
  monitoredServices = monitoredServices.filter(
    (service) => service.id !== id
  );

  saveMonitoredServices();
  renderMonitoredServices();
}
function formatBytes(bytes) {
  if (typeof bytes !== "number") return "—";

  if (bytes < 1024) {
    return `${bytes} bytes`;
  }

  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(2)} KB`;
  }

  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function securityHeaderStatus(security, key) {
  const item = security?.headers?.[key];

  if (!item) return "Missing";

  return item.present
    ? "Present"
    : "Missing";
}

function showMonitorDetails(id) {
  const service = monitoredServices.find(
    (item) => item.id === id
  );

  if (!service) return;

  if (
    !service.website ||
    !service.network ||
    !service.ssl ||
    !service.security
  ) {
    showToast("Run Check Now first to load full website details.");
    return;
  }

  const website = service.website || {};
  const network = service.network || {};
  const ssl = service.ssl || {};
  const security = service.security || {};

  $("monitor-details-name").textContent =
    `${service.name} — Website Intelligence`;

  $("monitor-details-url").textContent =
    service.url;

  $("detail-status").textContent =
    (service.status || "unknown").toUpperCase();

  $("detail-http").textContent =
    service.http_status ?? "—";

  $("detail-latency").textContent =
    typeof service.latency_ms === "number"
      ? `${service.latency_ms.toFixed(2)} ms`
      : "—";

  $("detail-checked").textContent =
    service.checked_at
      ? new Date(service.checked_at).toLocaleString()
      : "—";

  $("detail-title").textContent =
    website.page_title || "—";

  $("detail-domain").textContent =
    website.domain || "—";

  $("detail-protocol").textContent =
    website.protocol || "—";

  $("detail-hosting").textContent =
    website.hosting_hint || "Unknown";

  $("detail-content-type").textContent =
    website.content_type || "—";

  $("detail-size").textContent =
    formatBytes(website.response_size_bytes);

  $("detail-redirects").textContent =
    website.redirect_count ?? "—";

  $("detail-final-url").textContent =
    website.final_url || "—";

  $("detail-ssl-enabled").textContent =
    ssl.enabled ? "Yes" : "No";

  $("detail-ssl-valid").textContent =
    ssl.valid ? "Valid" : "Invalid / unavailable";

  $("detail-ssl-issuer").textContent =
    ssl.issuer || "—";

  $("detail-ssl-subject").textContent =
    ssl.subject || "—";

  $("detail-ssl-expiry").textContent =
    ssl.expires_at
      ? new Date(ssl.expires_at).toLocaleString()
      : "—";

  $("detail-ssl-days").textContent =
    ssl.days_until_expiry ?? "—";

  $("detail-ips").textContent =
    Array.isArray(network.resolved_ips) &&
    network.resolved_ips.length
      ? network.resolved_ips.join(", ")
      : "—";

  $("detail-security-score").textContent =
    `${security.score ?? 0} / ${security.total ?? 0}`;

  $("detail-hsts").textContent =
    securityHeaderStatus(
      security,
      "strict_transport_security"
    );

  $("detail-csp").textContent =
    securityHeaderStatus(
      security,
      "content_security_policy"
    );

  $("detail-content-options").textContent =
    securityHeaderStatus(
      security,
      "x_content_type_options"
    );

  $("detail-frame-options").textContent =
    securityHeaderStatus(
      security,
      "x_frame_options"
    );

  $("detail-referrer").textContent =
    securityHeaderStatus(
      security,
      "referrer_policy"
    );

  $("detail-permissions").textContent =
    securityHeaderStatus(
      security,
      "permissions_policy"
    );

  $("monitor-details").hidden = false;

  $("monitor-details").scrollIntoView({
    behavior: "smooth",
    block: "start",
  });
}
async function refreshAllMonitoredServices() {
  await Promise.allSettled(
    monitoredServices.map((service) =>
      checkMonitoredService(service.id)
    )
  );
}
/* ---------------- boot ---------------- */

function startClock() {
  const clock = $("clock");
  const tick = () => {
    clock.textContent = new Date().toLocaleTimeString();
  };
  tick();
  setInterval(tick, 1000);
}

async function boot() {
  startClock();
  const results = await Promise.allSettled([
    apiFetch(`${API}/health`),
    apiFetch(`${API}/model-info`),
    apiFetch(`${API}/dependency-graph`),
  ]);
  const [health, modelInfo, graph] = results;
  if (health.status === "fulfilled") {
    renderHealth(health.value);
  } else if (health.reason instanceof ApiError) {
    showToast(`Health check failed: ${health.reason.message}`);
  }
  if (modelInfo.status === "fulfilled") {
    renderModelInfo(modelInfo.value);
  } else if (modelInfo.reason instanceof ApiError) {
    showToast(`Model info failed: ${modelInfo.reason.message}`);
  }
  if (graph.status === "fulfilled") {
    renderGraph(graph.value);
    populateSimulator();
  } else if (graph.reason instanceof ApiError) {
    showToast(`Dependency graph failed: ${graph.reason.message}`);
  }
}

$("btn-normal").addEventListener(
  "click",
  () => runDemoAnalysis("normal")
);

$("btn-incident").addEventListener(
  "click",
  () => runDemoAnalysis("anomalous")
);

$("btn-simulate").addEventListener(
  "click",
  runSimulation
);

$("btn-add-monitor").addEventListener(
  "click",
  () => {
    $("monitor-form").hidden = false;
    $("monitor-name").focus();
  }
);

$("btn-cancel-monitor").addEventListener(
  "click",
  () => {
    $("monitor-form").hidden = true;
    $("monitor-name").value = "";
    $("monitor-url").value = "";
  }
);

$("btn-save-monitor").addEventListener(
  "click",
  addMonitoredService
);

$("monitor-grid").addEventListener(
  "click",
  (event) => {
    const button = event.target.closest(
      "button[data-action]"
    );

    if (!button) return;

    const id = button.dataset.id;

   if (button.dataset.action === "check") {
  checkMonitoredService(id);
}

if (button.dataset.action === "details") {
  showMonitorDetails(id);
}

if (button.dataset.action === "remove") {
  removeMonitoredService(id);
}
  }
);
$("btn-close-monitor-details").addEventListener(
  "click",
  () => {
    $("monitor-details").hidden = true;
  }
);
$("btn-close-monitor-details").addEventListener(
  "click",
  () => {
    $("monitor-details").hidden = true;
  }
);

loadMonitoredServices();
renderMonitoredServices();

boot();
refreshAllMonitoredServices();

boot();
