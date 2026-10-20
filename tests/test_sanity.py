"""Baseline sanity tests: environment, package, config, project tree."""
import importlib
import importlib.metadata

import pytest

DEPENDENCIES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "networkx": "networkx",
    "shap": "shap",
    "joblib": "joblib",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "jinja2": "jinja2",
    "pydantic": "pydantic",
    "pydantic_settings": "pydantic-settings",
    "matplotlib": "matplotlib",
    "pytest": "pytest",
    "httpx": "httpx",
}


@pytest.mark.parametrize("module,dist", sorted(DEPENDENCIES.items()))
def test_third_party_dependency_imports(module: str, dist: str) -> None:
    importlib.import_module(module)
    assert importlib.metadata.version(dist)


def test_sentinelai_package_imports() -> None:
    import sentinelai

    assert sentinelai.__version__

    from sentinelai.config import get_settings

    settings = get_settings()
    assert settings.random_state == 42


def test_project_tree_exists() -> None:
    from sentinelai.config import get_settings

    settings = get_settings()
    settings.ensure_dirs()

    for directory in (
        settings.data_raw,
        settings.data_processed,
        settings.data_demo,
        settings.models_dir,
        settings.reports_dir,
        settings.figures_dir,
    ):
        assert directory.is_dir(), f"missing directory: {directory}"


def test_risk_weights_sum_to_one() -> None:
    from sentinelai.config import get_settings

    settings = get_settings()
    total = (
        settings.weight_classifier
        + settings.weight_anomaly_detector
        + settings.weight_event_irregularity
        + settings.weight_event_intensity
    )
    assert total == pytest.approx(1.0)
