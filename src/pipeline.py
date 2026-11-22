"""End-to-end orchestration used by the Streamlit app (and by tests).

user input -> geocode -> routes -> features -> AI scoring (Vertex AI) -> explanations
-> comparison -> recommendation
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime

from src import conditions, routing
from src.ai_service import RiskPredictor, score_vectors
from src.errors import AppError, InvalidInputError, NetworkError, RoutingServiceError
from src.risk_engine import (
    AnalyzedRoute,
    RiskAssessment,
    compare_routes,
    explain_recommendation,
    explain_route,
)
from src.utils import LOG, http_get_json, sanitize_location


@dataclass
class AnalysisResult:
    start: routing.Location
    end: routing.Location
    routes: list[AnalyzedRoute]          # ranked, best first
    explanation: str
    warnings: list[str] = field(default_factory=list)
    backend_label: str = ""
    backend_id: str = ""
    ai_latency_ms: float = 0.0
    used_fallback_routes: bool = False

    @property
    def recommended(self) -> AnalyzedRoute:
        return self.routes[0]


def analyze(
    start_text: str,
    end_text: str,
    predictor: RiskPredictor,
    *,
    now: datetime | None = None,
    http_get=http_get_json,
) -> AnalysisResult:
    """Run the full analysis. Raises AppError subclasses with user-friendly messages."""
    start_q = sanitize_location(start_text, "Starting location")
    end_q = sanitize_location(end_text, "Destination")
    if start_q.casefold() == end_q.casefold():
        raise InvalidInputError("The starting location and destination are the same. Please enter two different places.")

    warnings: list[str] = []

    start = routing.geocode(start_q, http_get)
    end = routing.geocode(end_q, http_get)
    if start.source != "nominatim" or end.source != "nominatim":
        warnings.append(
            "Location search was unreachable, so the built-in list of Indian cities was used for coordinates."
        )
    if abs(start.lat - end.lat) < 1e-4 and abs(start.lon - end.lon) < 1e-4:
        raise InvalidInputError("Both locations resolve to the same place. Please enter two different places.")

    used_fallback = False
    try:
        routes = routing.fetch_routes(start, end, http_get)
    except (NetworkError, RoutingServiceError) as exc:
        LOG.warning("Routing service failed (%s); using approximate fallback routes", exc)
        routes = routing.fallback_routes(start, end)
        used_fallback = True
        warnings.append(
            "The live routing service could not be reached. Showing APPROXIMATE demo routes generated "
            "from the straight-line distance. They are not real roads."
        )
    if len(routes) < 2 and not used_fallback:
        warnings.append("Only one route was found between these places, so no alternatives could be compared.")

    # Build features for each route
    weather_cache: dict = {}
    built = []
    for r in routes:
        features, sources = conditions.build_features(r, weather_cache, now=now, http_get=http_get)
        built.append((r, features, sources))
    if any(s["Weather"] == conditions.SRC_ASSUMED for _, _, s in built):
        warnings.append("Live weather could not be fetched; weather was assumed to be 'cloudy'.")

    # AI inference - one batched request containing every route
    vectors = [f.to_vector() for _, f, _ in built]
    t0 = time.perf_counter()
    scores = score_vectors(predictor, vectors)
    latency_ms = (time.perf_counter() - t0) * 1000

    analyzed: list[AnalyzedRoute] = []
    for (r, features, sources), score in zip(built, scores):
        analyzed.append(
            AnalyzedRoute(
                route_id=r.route_id,
                name=r.name,
                distance_km=round(r.distance_km, 1),
                duration_min=r.duration_min,
                features=features,
                data_sources=sources,
                assessment=RiskAssessment.from_score(score),
                reasons=explain_route(features),
                coordinates=r.coordinates,
                route_source=r.source,
            )
        )

    ranked = compare_routes(analyzed)
    return AnalysisResult(
        start=start,
        end=end,
        routes=ranked,
        explanation=explain_recommendation(ranked),
        warnings=warnings,
        backend_label=predictor.backend_label,
        backend_id=predictor.backend_id,
        ai_latency_ms=latency_ms,
        used_fallback_routes=used_fallback,
    )


__all__ = ["analyze", "AnalysisResult", "AppError"]
