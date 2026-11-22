"""Risk assessment, explanations, route comparison and recommendation logic.

NOTE: this module does NOT compute the risk score. The score comes from the AI model
(Vertex AI endpoint). This module only (a) turns a score into a level, (b) writes
human-readable reasons from the route's input factors, and (c) compares routes.
The reasons are transparent rules over the input factors; they describe the inputs
the model was given, they are not the model's internal attribution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from src.data_processing import RouteFeatures, score_to_level
from src.errors import InvalidPredictionError, RouteUnavailableError


@dataclass(frozen=True)
class RiskAssessment:
    score: float  # 0-100
    level: str    # LOW / MEDIUM / HIGH

    @staticmethod
    def from_score(score: float) -> "RiskAssessment":
        if isinstance(score, bool) or score is None or not math.isfinite(float(score)):
            raise InvalidPredictionError(detail=f"cannot build assessment from {score!r}")
        if not 0 <= score <= 100:
            raise InvalidPredictionError(detail=f"score {score} outside 0-100")
        rounded = round(float(score), 1)
        return RiskAssessment(score=rounded, level=score_to_level(rounded))


@dataclass(frozen=True)
class Reason:
    factor: str
    text: str
    tone: str  # "good" | "ok" | "bad"


@dataclass
class AnalyzedRoute:
    """A route plus its features, AI assessment and presentation fields."""

    route_id: str
    name: str
    distance_km: float
    duration_min: float
    features: RouteFeatures
    data_sources: dict[str, str]
    assessment: RiskAssessment
    reasons: list[Reason] = field(default_factory=list)
    recommendation: str = ""
    rank: int = 0
    is_recommended: bool = False
    coordinates: list[tuple[float, float]] = field(default_factory=list)
    route_source: str = "osrm"


# ----------------------------------------------------------------------------------
# Explanations
# ----------------------------------------------------------------------------------

_CONDITION_TEXT = {
    "good": ("Good road condition", "good"),
    "fair": ("Fair road condition with some wear", "ok"),
    "poor": ("Poor road condition (rough or damaged surface likely)", "bad"),
}
_TRAFFIC_TEXT = {
    "low": ("Light traffic", "good"),
    "moderate": ("Moderate traffic", "ok"),
    "high": ("Heavy traffic", "bad"),
    "severe": ("Severe congestion", "bad"),
}
_WEATHER_TEXT = {
    "clear": ("Clear weather, low weather-related risk", "good"),
    "cloudy": ("Cloudy but dry, low weather-related risk", "good"),
    "rain": ("Rain reduces grip and visibility", "bad"),
    "fog": ("Fog reduces visibility", "bad"),
    "storm": ("Storm conditions, high weather-related risk", "bad"),
}
_CONSTRUCTION_TEXT = {
    "none": ("No construction or road work", "good"),
    "minor": ("Minor road work on the route", "ok"),
    "major": ("Major construction zone on the route", "bad"),
}
_ROAD_TYPE_TEXT = {
    "highway": ("Mostly highway: divided, higher-standard roads", "good"),
    "arterial": ("Mostly main (arterial) roads", "ok"),
    "urban": ("Mostly urban streets: many junctions and pedestrians", "ok"),
    "rural": ("Mostly rural roads: narrower, less lighting", "bad"),
}


def explain_route(features: RouteFeatures) -> list[Reason]:
    """Human-readable list of the factors that describe this route."""
    reasons = [
        Reason("Road condition", *_CONDITION_TEXT[features.road_condition]),
        Reason("Traffic", *_TRAFFIC_TEXT[features.traffic_level]),
        Reason("Weather", *_WEATHER_TEXT[features.weather_condition]),
        Reason("Construction", *_CONSTRUCTION_TEXT[features.construction]),
        Reason("Road type", *_ROAD_TYPE_TEXT[features.road_type]),
    ]
    if features.distance_km >= 150:
        reasons.append(Reason("Distance", f"Long route ({features.distance_km:.0f} km): driver fatigue risk", "bad"))
    elif features.distance_km >= 60:
        reasons.append(Reason("Distance", f"Medium-length route ({features.distance_km:.0f} km)", "ok"))
    else:
        reasons.append(Reason("Distance", f"Short route ({features.distance_km:.0f} km)", "good"))
    return reasons


# ----------------------------------------------------------------------------------
# Comparison / recommendation
# ----------------------------------------------------------------------------------

def _best_sort_key(r: AnalyzedRoute) -> tuple[float, float]:
    # lowest risk first; ties (within 0.05) broken by shorter distance
    return (round(r.assessment.score, 1), r.distance_km)


def _label_for(route: AnalyzedRoute, is_best: bool, only_route: bool) -> str:
    if only_route:
        return "Only route found"
    if is_best:
        if route.assessment.level == "HIGH":
            return "Least risky option - drive with extra caution"
        return "Recommended"
    return {"HIGH": "Avoid", "MEDIUM": "Use caution", "LOW": "Acceptable alternative"}[route.assessment.level]


def compare_routes(routes: Sequence[AnalyzedRoute]) -> list[AnalyzedRoute]:
    """Rank routes by risk (lowest first), set rank/recommendation fields, return sorted list."""
    if not routes:
        raise RouteUnavailableError()
    ordered = sorted(routes, key=_best_sort_key)
    only = len(ordered) == 1
    for i, r in enumerate(ordered, start=1):
        r.rank = i
        r.is_recommended = i == 1
        r.recommendation = _label_for(r, is_best=(i == 1), only_route=only)
    return ordered


_FACTOR_PHRASE = {
    "Road condition": lambda a, b: f"better road condition ({a.road_condition} -> {b.road_condition})",
    "Traffic": lambda a, b: f"lighter traffic ({a.traffic_level} -> {b.traffic_level})",
    "Weather": lambda a, b: f"better weather ({a.weather_condition} -> {b.weather_condition})",
    "Construction": lambda a, b: f"less construction ({a.construction} -> {b.construction})",
    "Road type": lambda a, b: f"safer road type ({a.road_type} -> {b.road_type})",
}


def explain_recommendation(ranked: Sequence[AnalyzedRoute]) -> str:
    """Plain-language explanation of WHY the top-ranked route was recommended."""
    if not ranked:
        raise RouteUnavailableError()
    best = ranked[0]
    if len(ranked) == 1:
        return (
            f"Only one route was available, so there is nothing to compare. "
            f"{best.name} has a risk score of {best.assessment.score:.0f}/100 ({best.assessment.level})."
        )

    shortest = min(ranked, key=lambda r: r.distance_km)
    all_high = all(r.assessment.level == "HIGH" for r in ranked)
    prefix = ""
    if all_high:
        prefix = "All available routes are rated HIGH risk. Consider delaying the trip or driving with extra care. "

    if shortest is best:
        others = ", ".join(f"{r.name} ({r.assessment.score:.0f})" for r in ranked[1:])
        return (
            f"{prefix}{best.name} is both the shortest route and the lowest-risk option "
            f"(score {best.assessment.score:.0f}/100, {best.assessment.level}), compared with {others}."
        )

    gap = shortest.assessment.score - best.assessment.score
    extra_km = best.distance_km - shortest.distance_km
    extra_pct = (extra_km / shortest.distance_km * 100) if shortest.distance_km else 0.0

    sev_ref, sev_best = shortest.features.severity(), best.features.severity()
    deltas = sorted(
        ((sev_ref[k] - sev_best[k], k) for k in _FACTOR_PHRASE if sev_ref[k] - sev_best[k] >= 0.25),
        reverse=True,
    )[:3]
    if deltas:
        reasons = "; ".join(_FACTOR_PHRASE[k](shortest.features, best.features) for _, k in deltas)
        why = f"Main differences: {reasons}."
    else:
        why = "The AI model rates the combination of its conditions as lower risk."

    length_note = (
        f"It is {extra_km:.1f} km ({extra_pct:.0f}%) longer than the shortest route, {shortest.name}."
        if extra_km > 0.05
        else f"It is about the same length as the shortest route, {shortest.name}."
    )
    if gap > 0:
        head = (
            f"{prefix}{best.name} is recommended because its risk score is {gap:.0f} points lower "
            f"than the shortest route {shortest.name} ({best.assessment.score:.0f} vs "
            f"{shortest.assessment.score:.0f} out of 100)."
        )
    else:
        head = f"{prefix}{best.name} is recommended (risk score {best.assessment.score:.0f}/100)."
    return f"{head} {length_note} {why}"
