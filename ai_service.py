"""AI inference layer.

Two interchangeable backends, selected by the AI_BACKEND environment variable:

* ``vertex`` (default, used on Cloud Run): sends the encoded route features to a
  Vertex AI ONLINE PREDICTION ENDPOINT and parses the response. If Vertex AI is
  unreachable or misconfigured the app shows an error - it never silently falls
  back to a local model.
* ``local`` (development only): loads model/artifacts/model.joblib. The UI labels
  this clearly as a local development model.

Authentication uses Application Default Credentials (ADC). On Cloud Run that is
the service account attached to the service; on a laptop it is
``gcloud auth application-default login``. No keys or secrets are stored in code.
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from src.errors import (
    AIServiceError,
    AuthenticationError,
    ConfigurationError,
    InvalidPredictionError,
)
from src.utils import LOG

SCORE_TOLERANCE = 10.0  # regressors can overshoot 0/100 slightly; beyond this we treat it as invalid
DEFAULT_LOCATION = "us-central1"
DEFAULT_TIMEOUT_S = 30.0
LOCAL_MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "artifacts" / "model.joblib"


class RiskPredictor(Protocol):
    backend_id: str
    backend_label: str

    def predict(self, instances: Sequence[Sequence[float]]) -> list[float]: ...


# ----------------------------------------------------------------------------------
# Response parsing / validation (pure functions - easy to test)
# ----------------------------------------------------------------------------------

def parse_predictions(raw: Any, expected_count: int) -> list[float]:
    """Validate a raw prediction payload and return clean 0-100 scores.

    Accepts a list whose items are numbers or single-element lists of numbers
    (the shapes returned by Vertex AI's prebuilt scikit-learn container).
    Raises InvalidPredictionError for anything else.
    """
    if raw is None or isinstance(raw, (str, bytes, dict)) or not isinstance(raw, (list, tuple)):
        raise InvalidPredictionError(detail=f"predictions payload has unexpected type {type(raw).__name__}")
    if len(raw) != expected_count:
        raise InvalidPredictionError(detail=f"expected {expected_count} predictions, received {len(raw)}")

    scores: list[float] = []
    for item in raw:
        if isinstance(item, (list, tuple)):
            if len(item) != 1:
                raise InvalidPredictionError(detail=f"prediction item has {len(item)} values, expected 1")
            item = item[0]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise InvalidPredictionError(detail=f"non-numeric prediction {item!r}")
        value = float(item)
        if not math.isfinite(value):
            raise InvalidPredictionError(detail="prediction is NaN or infinite")
        if value < -SCORE_TOLERANCE or value > 100 + SCORE_TOLERANCE:
            raise InvalidPredictionError(detail=f"prediction {value} is far outside 0-100")
        scores.append(min(100.0, max(0.0, value)))
    return scores


def _exception_names(exc: BaseException) -> set[str]:
    return {cls.__name__ for cls in type(exc).__mro__}


def translate_vertex_exception(exc: BaseException) -> Exception:
    """Map google-auth / google-api-core exceptions to friendly AppErrors.

    Matches by class name so this works (and is testable) without the google libraries installed.
    """
    names = _exception_names(exc)
    if "DefaultCredentialsError" in names:
        return AuthenticationError(
            "Google Cloud credentials were not found. On Cloud Run, attach a service account to the service.",
            detail=str(exc),
        )
    if names & {"PermissionDenied", "Unauthenticated", "Forbidden", "Unauthorized", "RefreshError"}:
        return AuthenticationError(
            "The application's service account is not allowed to call Vertex AI. "
            "Grant it the 'Vertex AI User' role (roles/aiplatform.user).",
            detail=str(exc),
        )
    if "NotFound" in names:
        return ConfigurationError(
            "The Vertex AI endpoint was not found. Check the project, region and endpoint ID settings.",
            detail=str(exc),
        )
    if names & {"ServiceUnavailable", "DeadlineExceeded", "GatewayTimeout", "InternalServerError", "RetryError"}:
        err = AIServiceError(
            "The Vertex AI endpoint is temporarily unavailable. Please retry in a moment.", detail=str(exc)
        )
        err.transient = True
        return err
    if "ResourceExhausted" in names or "TooManyRequests" in names:
        return AIServiceError("The Vertex AI endpoint is busy or over quota. Please retry shortly.", detail=str(exc))
    if "InvalidArgument" in names or "BadRequest" in names:
        return AIServiceError(
            "Vertex AI rejected the request. The deployed model may expect different input features.",
            detail=str(exc),
        )
    if names & {"ConnectionError", "Timeout", "ConnectTimeout", "ReadTimeout", "TimeoutError", "OSError"}:
        err = AIServiceError("Could not reach Vertex AI (network problem). Please retry.", detail=str(exc))
        err.transient = True
        return err
    return AIServiceError(detail=f"{type(exc).__name__}: {exc}")


# ----------------------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------------------

class VertexAIPredictor:
    """Calls a Vertex AI online prediction endpoint (the real cloud inference path)."""

    backend_id = "vertex"
    backend_label = "Google Cloud Vertex AI endpoint"

    def __init__(
        self,
        project: str,
        location: str,
        endpoint_id: str,
        *,
        endpoint: Any | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        retries: int = 1,
    ):
        if not project or not endpoint_id:
            raise ConfigurationError(
                "Vertex AI is not configured. Set VERTEX_PROJECT_ID and VERTEX_ENDPOINT_ID "
                "(and optionally VERTEX_LOCATION)."
            )
        self.project = project
        self.location = location or DEFAULT_LOCATION
        self.endpoint_id = str(endpoint_id)
        self.timeout_s = timeout_s
        self.retries = max(0, retries)
        self._endpoint = endpoint  # injectable for tests; created lazily otherwise

    def _get_endpoint(self) -> Any:
        if self._endpoint is not None:
            return self._endpoint
        try:
            from google.cloud import aiplatform
        except ImportError as exc:
            raise ConfigurationError(
                "The google-cloud-aiplatform package is not installed.", detail=str(exc)
            ) from exc
        try:
            aiplatform.init(project=self.project, location=self.location)
            self._endpoint = aiplatform.Endpoint(self.endpoint_id)
        except Exception as exc:
            raise translate_vertex_exception(exc) from exc
        return self._endpoint

    def predict(self, instances: Sequence[Sequence[float]]) -> list[float]:
        if not instances:
            raise InvalidPredictionError("There is nothing to predict.", detail="empty instances")
        payload = [list(map(float, row)) for row in instances]
        endpoint = self._get_endpoint()

        response = None
        for attempt in range(self.retries + 1):
            try:
                response = endpoint.predict(instances=payload, timeout=self.timeout_s)
                break
            except Exception as exc:
                error = translate_vertex_exception(exc)
                LOG.warning("Vertex AI predict attempt %d failed: %s", attempt + 1, error)
                if attempt < self.retries and getattr(error, "transient", False):
                    time.sleep(1.0 + attempt)  # short back-off, then retry once
                    continue
                raise error from exc

        predictions = getattr(response, "predictions", None)
        return parse_predictions(predictions, expected_count=len(payload))


class LocalModelPredictor:
    """Development-only backend that loads the joblib model from disk."""

    backend_id = "local"
    backend_label = "Local model (development mode - NOT Vertex AI)"

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path) if model_path else LOCAL_MODEL_PATH
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            if not self.model_path.exists():
                raise ConfigurationError(
                    "No local model found. Run `python model/train_model.py` first.",
                    detail=str(self.model_path),
                )
            try:
                import joblib

                self._model = joblib.load(self.model_path)
            except ImportError as exc:
                raise ConfigurationError(
                    "Local mode needs scikit-learn and joblib (pip install -r requirements-train.txt).",
                    detail=str(exc),
                ) from exc
            except Exception as exc:
                raise ConfigurationError("The local model file could not be loaded.", detail=str(exc)) from exc
        return self._model

    def predict(self, instances: Sequence[Sequence[float]]) -> list[float]:
        if not instances:
            raise InvalidPredictionError("There is nothing to predict.", detail="empty instances")
        model = self._load()
        try:
            raw = model.predict([list(map(float, row)) for row in instances])
        except Exception as exc:
            raise AIServiceError("The local model failed to make a prediction.", detail=str(exc)) from exc
        return parse_predictions([float(x) for x in raw], expected_count=len(instances))


# ----------------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------------

def get_predictor_from_env(env: Mapping[str, str] | None = None) -> RiskPredictor:
    """Build the configured predictor. Raises ConfigurationError on bad settings."""
    env = os.environ if env is None else env
    backend = (env.get("AI_BACKEND") or "vertex").strip().lower()
    if backend == "vertex":
        project = env.get("VERTEX_PROJECT_ID") or env.get("GOOGLE_CLOUD_PROJECT") or ""
        return VertexAIPredictor(
            project=project,
            location=env.get("VERTEX_LOCATION") or DEFAULT_LOCATION,
            endpoint_id=env.get("VERTEX_ENDPOINT_ID") or "",
        )
    if backend == "local":
        return LocalModelPredictor(env.get("LOCAL_MODEL_PATH"))
    raise ConfigurationError("AI_BACKEND must be 'vertex' or 'local'.", detail=f"AI_BACKEND={backend!r}")


def score_vectors(predictor: RiskPredictor, vectors: Sequence[Sequence[float]]) -> list[float]:
    """Send a batch of encoded routes to the predictor and return 0-100 scores."""
    start = time.perf_counter()
    scores = predictor.predict(vectors)
    LOG.info(
        "AI backend=%s routes=%d latency_ms=%.0f", predictor.backend_id, len(vectors),
        (time.perf_counter() - start) * 1000,
    )
    return scores
