import os
from pathlib import Path

import joblib


# Local trained model
LOCAL_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "model"
    / "artifacts"
    / "model.joblib"
)


class LocalModelPredictor:
    def __init__(self, model_path=None):
        self.model_path = Path(model_path or LOCAL_MODEL_PATH)
        self.model = joblib.load(self.model_path)
        self.backend_id = "local"
        self.backend_label = "Local development model"

    def predict(self, features):
        return self.model.predict(features)


def get_predictor_from_env():
    """
    Select the AI backend.

    For the cloud demo, local is the default so the trained
    model.joblib inside the repository is used.
    """
    backend = (os.getenv("AI_BACKEND") or "local").strip().lower()

    if backend == "local":
        return LocalModelPredictor()

    raise RuntimeError(
        f"Unsupported AI_BACKEND: {backend}. "
        "Use 'local' for the deployed demo."
    )
