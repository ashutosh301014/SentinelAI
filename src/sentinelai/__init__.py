"""SentinelAI — Predictive AIOps Engine.

Core ML capabilities:
  * anomaly detection on log/event traces (Isolation Forest)
  * supervised incident prediction (Logistic Regression, Random Forest,
    HistGradientBoosting)
  * composite incident risk scoring
  * root-cause evidence ranking
  * service dependency graph (NetworkX) + blast-radius prediction
  * what-if recovery simulation

No LLM APIs, no Kafka/Kubernetes/GNNs/transformers, no auto-remediation.
"""

__version__ = "0.1.0"

from sentinelai.config import get_settings

__all__ = ["__version__", "get_settings"]
