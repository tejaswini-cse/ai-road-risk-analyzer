"""Feature definitions, encoding, validation and dataset helpers.

Used by BOTH the web app (to turn a route into a numeric vector before it is sent
to the model) and the training scripts (so training and inference always encode
features identically).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.errors import DataError, InvalidFeatureError

# --- Category -> number maps (ordinal: higher = riskier) -------------------------
ROAD_CONDITION = {"good": 0, "fair": 1, "poor": 2}
TRAFFIC_LEVEL = {"low": 0, "moderate": 1, "high": 2, "severe": 3}
WEATHER_CONDITION = {"clear": 0, "cloudy": 1, "rain": 2, "fog": 3, "storm": 4}
CONSTRUCTION = {"none": 0, "minor": 1, "major": 2}
ROAD_TYPES = ("highway", "arterial", "urban", "rural")  # one-hot encoded (not ordinal)

# The exact order of the model's input vector. Never reorder without retraining.
FEATURE_NAMES = [
    "road_condition",
    "traffic_level",
    "weather_condition",
    "construction",
    "distance_km",
    "road_type_highway",
    "road_type_arterial",
    "road_type_urban",
    "road_type_rural",
]

# Risk level thresholds on the 0-100 score
LOW_MAX = 30.0      # score < 30  -> LOW
MEDIUM_MAX = 55.0   # score < 55  -> MEDIUM, otherwise HIGH
MAX_DISTANCE_KM = 5000.0

RAW_COLUMNS = ["road_condition", "traffic_level", "weather_condition", "construction", "distance_km", "road_type"]


def score_to_level(score: float) -> str:
    """Map a 0-100 score to LOW / MEDIUM / HIGH."""
    if score is None or isinstance(score, bool) or not math.isfinite(float(score)):
        raise InvalidFeatureError("Risk score must be a finite number.")
    if score < LOW_MAX:
        return "LOW"
    if score < MEDIUM_MAX:
        return "MEDIUM"
    return "HIGH"


@dataclass(frozen=True)
class RouteFeatures:
    """Human-readable features describing one route."""

    road_condition: str
    traffic_level: str
    weather_condition: str
    construction: str
    distance_km: float
    road_type: str

    def validate(self) -> None:
        checks = {
            "road_condition": ROAD_CONDITION,
            "traffic_level": TRAFFIC_LEVEL,
            "weather_condition": WEATHER_CONDITION,
            "construction": CONSTRUCTION,
        }
        for name, allowed in checks.items():
            value = getattr(self, name)
            if value not in allowed:
                raise InvalidFeatureError(
                    "A route has an unrecognised condition value.",
                    detail=f"{name}={value!r}, allowed={sorted(allowed)}",
                )
        if self.road_type not in ROAD_TYPES:
            raise InvalidFeatureError(
                "A route has an unrecognised road type.", detail=f"road_type={self.road_type!r}"
            )
        d = self.distance_km
        if isinstance(d, bool) or not isinstance(d, (int, float)) or not math.isfinite(d):
            raise InvalidFeatureError("A route has an invalid distance.", detail=f"distance_km={d!r}")
        if d <= 0 or d > MAX_DISTANCE_KM:
            raise InvalidFeatureError(
                "A route distance is outside the supported range.", detail=f"distance_km={d!r}"
            )

    def to_vector(self) -> list[float]:
        """Validate, then encode into the model's numeric input order (FEATURE_NAMES)."""
        self.validate()
        one_hot = [1.0 if self.road_type == rt else 0.0 for rt in ROAD_TYPES]
        vec = [
            float(ROAD_CONDITION[self.road_condition]),
            float(TRAFFIC_LEVEL[self.traffic_level]),
            float(WEATHER_CONDITION[self.weather_condition]),
            float(CONSTRUCTION[self.construction]),
            round(float(self.distance_km), 3),
            *one_hot,
        ]
        assert len(vec) == len(FEATURE_NAMES)
        return vec

    def severity(self) -> dict[str, float]:
        """Per-factor severity in 0..1 (used for charts and for explaining differences)."""
        return {
            "Road condition": ROAD_CONDITION[self.road_condition] / 2,
            "Traffic": TRAFFIC_LEVEL[self.traffic_level] / 3,
            "Weather": WEATHER_CONDITION[self.weather_condition] / 4,
            "Construction": CONSTRUCTION[self.construction] / 2,
            "Road type": {"highway": 0.0, "arterial": 0.33, "urban": 0.66, "rural": 1.0}[self.road_type],
            "Distance": min(self.distance_km / 150.0, 1.0),
        }

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- Dataset helpers (training side; pandas / sklearn imported lazily) -----------

def load_dataset(path: str | Path):
    """Load the CSV and check it has the required columns. Raises DataError."""
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise DataError(f"Dataset not found: {path}. Run `python model/generate_dataset.py` first.")
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # malformed CSV
        raise DataError(f"Dataset could not be read: {path}", detail=str(exc)) from exc
    missing = [c for c in RAW_COLUMNS + ["risk_score"] if c not in df.columns]
    if missing:
        raise DataError(f"Dataset is missing required columns: {missing}")
    return df


def clean_dataset(df):
    """Drop rows with missing/invalid values and duplicates. Returns (clean_df, report)."""
    n0 = len(df)
    df = df.dropna(subset=RAW_COLUMNS + ["risk_score"]).copy()
    n_missing = n0 - len(df)

    valid = (
        df["road_condition"].isin(ROAD_CONDITION)
        & df["traffic_level"].isin(TRAFFIC_LEVEL)
        & df["weather_condition"].isin(WEATHER_CONDITION)
        & df["construction"].isin(CONSTRUCTION)
        & df["road_type"].isin(ROAD_TYPES)
        & (df["distance_km"] > 0)
        & (df["distance_km"] <= MAX_DISTANCE_KM)
        & df["risk_score"].between(0, 100)
    )
    n_invalid = int((~valid).sum())
    df = df[valid]
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report = {
        "rows_in": n0,
        "dropped_missing": n_missing,
        "dropped_invalid": n_invalid,
        "dropped_duplicates": before - len(df),
        "rows_out": len(df),
    }
    if len(df) < 100:
        raise DataError("Too few valid rows remain after cleaning to train a model.")
    return df, report


def encode_dataframe(df):
    """Encode a cleaned DataFrame into the model's numeric matrix (same code path as inference)."""
    import numpy as np

    rows = [
        RouteFeatures(
            road_condition=r.road_condition,
            traffic_level=r.traffic_level,
            weather_condition=r.weather_condition,
            construction=r.construction,
            distance_km=float(r.distance_km),
            road_type=r.road_type,
        ).to_vector()
        for r in df.itertuples(index=False)
    ]
    return np.asarray(rows, dtype=float)


def split_dataset(df, test_size: float = 0.2, seed: int = 42):
    """Stratified (by risk level) train/test split. Returns X_train, X_test, y_train, y_test."""
    from sklearn.model_selection import train_test_split

    X = encode_dataframe(df)
    y = df["risk_score"].to_numpy(dtype=float)
    strata = [score_to_level(s) for s in y]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strata)
