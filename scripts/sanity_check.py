"""SentinelAI environment sanity check.

Verifies the interpreter, third-party dependencies, the local package,
configuration and the project directory tree. Exit code 0 = healthy.
"""
from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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

REQUIRED_DIRS = (
    "data/raw",
    "data/processed",
    "data/demo",
    "models",
    "reports/figures",
    "app/templates",
    "app/static/css",
    "app/static/js",
    "scripts",
    "tests",
    "src/sentinelai",
)


def check_dependencies() -> int:
    failures = 0
    for import_name, dist_name in DEPENDENCIES.items():
        try:
            __import__(import_name)
        except Exception as exc:  # noqa: BLE001 - report every failure
            failures += 1
            print(f"[FAIL] {dist_name}: {exc}")
        else:
            version = importlib.metadata.version(dist_name)
            print(f"[ OK ] {dist_name}=={version}")
    return failures


def check_package() -> int:
    failures = 0
    try:
        import sentinelai
        from sentinelai.config import get_settings

        settings = get_settings()
        settings.ensure_dirs()
        print(f"[ OK ] sentinelai {sentinelai.__version__}")
        print(
            f"[ OK ] config loaded: api_prefix={settings.api_prefix}, "
            f"port={settings.port}, random_state={settings.random_state}, "
            f"models_dir={settings.models_dir}"
        )
    except Exception as exc:  # noqa: BLE001
        failures += 1
        print(f"[FAIL] sentinelai package: {exc}")
    return failures


def check_tree() -> int:
    failures = 0
    for rel in REQUIRED_DIRS:
        path = PROJECT_ROOT / rel
        if path.is_dir():
            print(f"[ OK ] {rel}/")
        else:
            failures += 1
            print(f"[FAIL] {rel}/ missing")
    return failures


def main() -> int:
    print(f"Python {sys.version.split()[0]} @ {sys.executable}")
    print(f"Project root: {PROJECT_ROOT}")
    print("-" * 60)
    failures = 0
    print("Third-party dependencies:")
    failures += check_dependencies()
    print("-" * 60)
    print("Local package & config:")
    failures += check_package()
    print("-" * 60)
    print("Project tree:")
    failures += check_tree()
    print("-" * 60)
    if failures:
        print(f"SANITY CHECK FAILED ({failures} problem(s))")
        return 1
    print("SANITY CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
