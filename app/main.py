"""FastAPI backend for SentinelAI — Predictive AIOps Engine.

Serves the trained production inference pipeline (see
``sentinelai.inference``) behind a REST API, and the dashboard frontend
(``app/static/``) at ``/``. Model loading happens at
application startup through the lifespan handler; the app NEVER retrains
and does not need the training dataset — it only reads the persisted
artifacts under ``models/``, the topology under ``data/config/`` and the
two demo traces under ``data/demo/``.

Production safety:

* request bodies are capped (413 beyond the limit)
* every error is a structured JSON envelope, validation errors include
  per-field details, and unhandled exceptions return a generic message —
  stack traces are logged server-side only, never sent to clients
* logs record method/path/status/duration, never request bodies
"""
from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

# Self-contained entrypoint: `python app/main.py` must find the package
# without a separate PYTHONPATH export.
_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from sentinelai.blast_radius import estimate_blast_radius
from sentinelai.config import get_settings
from sentinelai.dependency_graph import EDGE_ATTRIBUTES, NODE_ATTRIBUTES
from sentinelai.inference import InferencePipeline
from sentinelai.recovery_simulator import simulate_recovery as run_simulation

from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BlastRadiusRequest,
    BlastRadiusResponse,
    DemoAnalyzeRequest,
    DependencyEdgeModel,
    DependencyGraphResponse,
    HealthResponse,
    ModelInfoResponse,
    ServiceNodeModel,
    SimulateRecoveryRequest,
    SimulateRecoveryResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("sentinelai.api")

MAX_BODY_BYTES = 100 * 1024  # 100 KB — analysis payloads are ~2 KB
STATIC_DIR = Path(__file__).resolve().parent / "static"

settings = get_settings()


def _error_envelope(
    error_type: str, message: str, details: list[dict] | None = None
) -> dict:
    envelope: dict = {"error": {"type": error_type, "message": message}}
    if details:
        envelope["error"]["details"] = details
    return envelope


def _load_demo_samples(settings) -> dict[str, tuple[str, pd.DataFrame]]:
    metadata_path = settings.data_demo / "demo_samples_metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(
            f"demo metadata missing at {metadata_path}; run "
            "scripts/build_demo_samples.py first"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    samples: dict[str, tuple[str, pd.DataFrame]] = {}
    for record in metadata["samples"]:
        role = str(record["role"]).removeprefix("representative_")
        path = settings.data_demo / record["file"]
        if not path.is_file():
            raise RuntimeError(f"demo sample missing at {path}")
        frame = pd.read_csv(path, index_col=0)
        samples[role] = (str(record["block_id"]), frame)
    return samples


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup: loading production artifacts (no retraining)...")
    pipeline = InferencePipeline()
    demo_samples = _load_demo_samples(settings)
    app.state.pipeline = pipeline
    app.state.demo_samples = demo_samples
    logger.info(
        "startup complete: model=%s version=%s dataset=%s "
        "features=%d demo_samples=%d",
        pipeline.metadata["selected_model"],
        pipeline.metadata["model_version"],
        pipeline.metadata["dataset"],
        pipeline.metadata["feature_count"],
        len(demo_samples),
    )
    yield
    logger.info("shutdown: application stopped")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Predictive AIOps engine: incident detection, composite risk "
        "scoring, root-cause evidence, dependency-graph blast radius and "
        "what-if recovery simulation. Serves trained artifacts only — "
        "no training dataset is required and inference never retrains."
    ),
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    if request.method in {"POST", "PUT", "PATCH"}:
        content_length = request.headers.get("content-length")
        if (
            content_length
            and content_length.isdigit()
            and int(content_length) > MAX_BODY_BYTES
        ):
            return JSONResponse(
                status_code=413,
                content=_error_envelope(
                    "payload_too_large",
                    f"request body exceeds the {MAX_BODY_BYTES // 1024} KB "
                    "limit",
                ),
            )
    return await call_next(request)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000.0
    logger.info(
        "%s %s -> %d (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
):
    details = [
        {
            "loc": [str(part) for part in err.get("loc", [])],
            "msg": str(err.get("msg")),
            "type": str(err.get("type")),
        }
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content=_error_envelope(
            "validation_error", "request validation failed", details
        ),
    )


# Registered for both classes: endpoint-raised fastapi.HTTPException and
# router-raised starlette.exceptions.HTTPException (unmatched route 404 /
# wrong method 405) must produce the same structured envelope.
@app.exception_handler(StarletteHTTPException)
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    error_types = {
        404: "not_found",
        405: "method_not_allowed",
        413: "payload_too_large",
        422: "unprocessable_entity",
        503: "service_unavailable",
    }
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_envelope(
            error_types.get(exc.status_code, "http_error"), str(exc.detail)
        ),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception(
        "unhandled error on %s %s", request.method, request.url.path
    )
    return JSONResponse(
        status_code=500,
        content=_error_envelope(
            "internal_error", "an internal error occurred"
        ),
    )


def _run_pipeline(
    pipeline: InferencePipeline,
    frame: pd.DataFrame,
    block_id: str,
    timestamp: str | None,
    what_if: list | None,
    top_k: int,
) -> dict:
    specs = [spec.model_dump() for spec in what_if] if what_if else None
    try:
        result = pipeline.run(
            frame,
            block_id=block_id,
            timestamp=timestamp,
            what_if=specs,
            top_k=top_k,
        )
    except (ValueError, TypeError) as exc:
        logger.info("inference request rejected: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return result.to_dict()


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get(f"{settings.api_prefix}", include_in_schema=False)
def api_index() -> dict:
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "api": settings.api_prefix,
        "note": (
            "serves trained artifacts only — no training dataset "
            "required, inference never retrains"
        ),
        "endpoints": [
            f"{settings.api_prefix}/health",
            f"{settings.api_prefix}/model-info",
            f"{settings.api_prefix}/analyze",
            f"{settings.api_prefix}/demo/normal",
            f"{settings.api_prefix}/demo/anomaly",
            f"{settings.api_prefix}/demo/analyze",
            f"{settings.api_prefix}/dependency-graph",
            f"{settings.api_prefix}/blast-radius",
            f"{settings.api_prefix}/simulate-recovery",
        ],
    }


@app.get(f"{settings.api_prefix}/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    metadata = request.app.state.pipeline.metadata
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
        model_loaded=True,
        dataset=metadata["dataset"],
        model_version=metadata["model_version"],
    )


@app.get(
    f"{settings.api_prefix}/model-info", response_model=ModelInfoResponse
)
def model_info(request: Request) -> ModelInfoResponse:
    metadata = request.app.state.pipeline.metadata
    return ModelInfoResponse(
        dataset=metadata["dataset"],
        trained_at=metadata["trained_at"],
        selected_model=metadata["selected_model"],
        model_version=metadata["model_version"],
        classifier_threshold=metadata["classifier_threshold"],
        feature_count=metadata["feature_count"],
        feature_names=metadata["feature_names"],
        evaluation_metrics=metadata["evaluation_metrics"],
        anomaly_detector=metadata["anomaly_detector"],
    )


@app.post(f"{settings.api_prefix}/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest, request: Request) -> dict:
    block_id = payload.block_id or f"api-{uuid4().hex[:16]}"
    frame = pd.DataFrame([payload.events], index=[block_id])
    return _run_pipeline(
        request.app.state.pipeline,
        frame,
        block_id,
        payload.timestamp,
        payload.what_if,
        payload.top_k,
    )


@app.get(
    f"{settings.api_prefix}/demo/normal", response_model=AnalyzeResponse
)
def demo_normal(request: Request) -> dict:
    block_id, frame = request.app.state.demo_samples["normal"]
    return _run_pipeline(
        request.app.state.pipeline, frame, block_id, None, None, 5
    )


@app.get(
    f"{settings.api_prefix}/demo/anomaly", response_model=AnalyzeResponse
)
def demo_anomaly(request: Request) -> dict:
    block_id, frame = request.app.state.demo_samples["anomalous"]
    return _run_pipeline(
        request.app.state.pipeline, frame, block_id, None, None, 5
    )


@app.post(
    f"{settings.api_prefix}/demo/analyze", response_model=AnalyzeResponse
)
def demo_analyze(payload: DemoAnalyzeRequest, request: Request) -> dict:
    block_id, frame = request.app.state.demo_samples[payload.sample]
    return _run_pipeline(
        request.app.state.pipeline,
        frame,
        block_id,
        payload.timestamp,
        payload.what_if,
        5,
    )


@app.get(
    f"{settings.api_prefix}/dependency-graph",
    response_model=DependencyGraphResponse,
)
def dependency_graph(request: Request) -> DependencyGraphResponse:
    graph = request.app.state.pipeline.graph
    services = [
        ServiceNodeModel(
            id=str(node),
            name=str(graph.nodes[node]["name"]),
            **{
                attr: graph.nodes[node][attr] for attr in NODE_ATTRIBUTES
            },
        )
        for node in graph.nodes
    ]
    dependencies = [
        DependencyEdgeModel(
            service_id=str(u),
            depends_on=str(v),
            **{attr: data[attr] for attr in EDGE_ATTRIBUTES},
        )
        for u, v, data in graph.edges(data=True)
    ]
    return DependencyGraphResponse(
        service_count=len(services),
        dependency_count=len(dependencies),
        services=services,
        dependencies=dependencies,
    )


@app.post(
    f"{settings.api_prefix}/blast-radius", response_model=BlastRadiusResponse
)
def blast_radius(payload: BlastRadiusRequest, request: Request) -> dict:
    graph = request.app.state.pipeline.graph
    if payload.origin not in graph:
        raise HTTPException(
            status_code=404, detail=f"unknown service {payload.origin!r}"
        )
    try:
        result = estimate_blast_radius(
            graph,
            payload.origin,
            payload.origin_risk,
            settings=settings,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return result.to_dict()


@app.post(
    f"{settings.api_prefix}/simulate-recovery",
    response_model=SimulateRecoveryResponse,
)
def simulate_recovery(
    payload: SimulateRecoveryRequest, request: Request
) -> dict:
    graph = request.app.state.pipeline.graph
    if payload.affected_service not in graph:
        raise HTTPException(
            status_code=404,
            detail=f"unknown service {payload.affected_service!r}",
        )
    if payload.origin is not None and payload.origin not in graph:
        raise HTTPException(
            status_code=404, detail=f"unknown service {payload.origin!r}"
        )
    try:
        result = run_simulation(
            graph,
            payload.action,
            payload.affected_service,
            origin=payload.origin,
            origin_risk=payload.origin_risk,
            settings=settings,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return result.to_dict()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
