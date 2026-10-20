"""Central configuration for SentinelAI.

Every setting can be overridden through environment variables prefixed with
``SENTINELAI_`` (e.g. ``SENTINELAI_PORT=9000``) or via a ``.env`` file in the
project root.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application-wide settings.

    Paths resolve against the repository root so configuration behaves
    identically no matter where the interpreter is launched from.
    """

    model_config = SettingsConfigDict(
        env_prefix="SENTINELAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- serving ----
    app_name: str = "SentinelAI — Predictive AIOps Engine"
    app_version: str = "0.1.0"
    api_prefix: str = "/api"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # ---- data & artifacts ----
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    data_demo: Path = PROJECT_ROOT / "data" / "demo"
    service_topology: Path = PROJECT_ROOT / "data" / "config" / "service_topology.json"
    component_service_mapping: Path = (
        PROJECT_ROOT / "data" / "config" / "component_service_mapping.json"
    )
    models_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"
    figures_dir: Path = PROJECT_ROOT / "reports" / "figures"

    # ---- ML pipeline ----
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    contamination: float = 0.05  # expected anomaly share for IsolationForest
    n_estimators: int = 200      # ensembles (RF / HistGB / IsolationForest)

    # ---- composite incident risk score ----
    # Weights of the four input signals (each normalized to [0, 1]).
    # Must sum to 1.0.
    weight_classifier: float = 0.45          # supervised anomaly probability
    weight_anomaly_detector: float = 0.30    # Isolation Forest anomaly score
    weight_event_irregularity: float = 0.15  # event-pattern irregularity
    weight_event_intensity: float = 0.10     # event volume intensity
    # Severity band lower bounds: 0-29 LOW, 30-59 MEDIUM, 60-79 HIGH,
    # 80-100 CRITICAL.
    risk_band_thresholds: tuple[int, int, int] = (30, 60, 80)

    @model_validator(mode="after")
    def _validate_risk_config(self) -> "Settings":
        weights = (
            self.weight_classifier,
            self.weight_anomaly_detector,
            self.weight_event_irregularity,
            self.weight_event_intensity,
        )
        total = sum(weights)
        if not math.isclose(total, 1.0, rel_tol=1e-9):
            raise ValueError(
                f"risk-score weights must sum to 1.0, got {total:.6f}"
            )
        thresholds = self.risk_band_thresholds
        if (
            len(thresholds) != 3
            or any(not isinstance(t, int) or not 0 < t < 100 for t in thresholds)
        ):
            raise ValueError(
                "risk_band_thresholds must be 3 integers strictly between "
                "0 and 100"
            )
        if not thresholds[0] < thresholds[1] < thresholds[2]:
            raise ValueError("risk_band_thresholds must be strictly increasing")
        return self

    def model_path(self, name: str) -> Path:
        """Path where a named model artifact is (de)serialized."""
        return self.models_dir / f"{name}.joblib"

    def ensure_dirs(self) -> None:
        """Create every data/artifact directory the pipeline writes to."""
        for directory in (
            self.data_raw,
            self.data_processed,
            self.data_demo,
            self.service_topology.parent,
            self.models_dir,
            self.reports_dir,
            self.figures_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton; call this instead of Settings() directly."""
    return Settings()


settings = get_settings()
