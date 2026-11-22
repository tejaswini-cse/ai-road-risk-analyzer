"""Estimate the model's input features for a route.

Honest data-source summary (also shown in the UI):

| Factor            | Source                                                              |
|-------------------|---------------------------------------------------------------------|
| distance, type    | Derived from the OSRM route (road classes + average speed)          |
| weather           | LIVE: Open-Meteo current weather at the route midpoint (free, no key)|
| traffic           | ESTIMATED from time of day + road type (NOT live traffic)            |
| road condition    | SIMULATED, deterministic per route (no free nationwide dataset)      |
| construction      | SIMULATED, deterministic per route                                   |

Simulated values are seeded from the route geometry so the same route always gets the same
values. They are placeholders for real data sources (see README "Future improvements").
"""
from __future__ import annotations

import random
from datetime import datetime
from typing import Any, Callable

from src.data_processing import RouteFeatures
from src.errors import AppError
from src.routing import Route
from src.utils import LOG, http_get_json, local_now, stable_seed

HttpGet = Callable[..., Any]
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

SRC_LIVE = "live (Open-Meteo)"
SRC_ESTIMATED = "estimated (time of day + road type)"
SRC_SIMULATED = "simulated demo value"
SRC_ROUTE = "derived from route"
SRC_ASSUMED = "assumed (weather service unavailable)"


def categorize_weather(weather_code: Any) -> str:
    """Map a WMO weather code (as returned by Open-Meteo) to the model's categories."""
    try:
        code = int(weather_code)
    except (TypeError, ValueError):
        return "cloudy"
    if code in (0, 1):
        return "clear"
    if code in (2, 3):
        return "cloudy"
    if code in (45, 48):
        return "fog"
    if 95 <= code <= 99 or code in (71, 73, 75, 77, 85, 86, 66, 67):
        return "storm"  # thunderstorm, snow, freezing rain: treated as severe
    if 51 <= code <= 65 or 80 <= code <= 82:
        return "rain"
    return "cloudy"


def fetch_weather(lat: float, lon: float, http_get: HttpGet = http_get_json) -> tuple[str, str]:
    """Return (weather category, source label). Never raises: falls back to an assumed value."""
    try:
        payload = http_get(
            OPEN_METEO_URL,
            params={"latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}", "current": "weather_code"},
            timeout=8,
        )
        code = payload["current"]["weather_code"]
        return categorize_weather(code), SRC_LIVE
    except (AppError, KeyError, TypeError) as exc:
        LOG.warning("Weather lookup failed (%s); assuming 'cloudy'", exc)
        return "cloudy", SRC_ASSUMED


def estimate_traffic(road_type: str, hour: int, rng: random.Random) -> str:
    """Rule-based traffic estimate. NOT live traffic."""
    if 8 <= hour < 11 or 17 <= hour < 21:
        level = 2
    elif 11 <= hour < 17 or 6 <= hour < 8 or 21 <= hour < 22:
        level = 1
    else:
        level = 0
    level += {"urban": 1, "arterial": 0, "highway": 0, "rural": -1}[road_type]
    level += rng.choice([-1, 0, 0, 1])  # per-route variation, seeded
    return ["low", "moderate", "high", "severe"][max(0, min(3, level))]


_CONDITION_WEIGHTS = {
    "highway": [0.60, 0.30, 0.10],
    "arterial": [0.40, 0.40, 0.20],
    "urban": [0.30, 0.40, 0.30],
    "rural": [0.15, 0.40, 0.45],
}
_CONSTRUCTION_WEIGHTS = {
    "highway": [0.62, 0.25, 0.13],
    "arterial": [0.70, 0.22, 0.08],
    "urban": [0.65, 0.25, 0.10],
    "rural": [0.80, 0.15, 0.05],
}


def route_seed(route: Route) -> int:
    pts = route.coordinates
    sample = pts[:: max(1, len(pts) // 12)]
    return stable_seed(round(route.distance_km, 1), *[f"{p[0]:.3f},{p[1]:.3f}" for p in sample])


def simulate_road_condition(road_type: str, rng: random.Random) -> str:
    return rng.choices(["good", "fair", "poor"], weights=_CONDITION_WEIGHTS[road_type])[0]


def simulate_construction(road_type: str, rng: random.Random) -> str:
    return rng.choices(["none", "minor", "major"], weights=_CONSTRUCTION_WEIGHTS[road_type])[0]


def build_features(
    route: Route,
    weather_cache: dict[tuple[float, float], tuple[str, str]],
    now: datetime | None = None,
    http_get: HttpGet = http_get_json,
) -> tuple[RouteFeatures, dict[str, str]]:
    """Build model input features + a data-source label for each factor."""
    now = now or local_now()
    rng = random.Random(route_seed(route))
    road_type = route.road_type

    mid_lat, mid_lon = route.coordinates[len(route.coordinates) // 2]
    key = (round(mid_lat, 1), round(mid_lon, 1))  # reuse nearby lookups (also polite to the API)
    if key not in weather_cache:
        weather_cache[key] = fetch_weather(mid_lat, mid_lon, http_get)
    weather, weather_src = weather_cache[key]

    features = RouteFeatures(
        road_condition=simulate_road_condition(road_type, rng),
        traffic_level=estimate_traffic(road_type, now.hour, rng),
        weather_condition=weather,
        construction=simulate_construction(road_type, rng),
        distance_km=round(route.distance_km, 1),
        road_type=road_type,
    )
    sources = {
        "Road condition": SRC_SIMULATED,
        "Traffic": SRC_ESTIMATED,
        "Weather": weather_src,
        "Construction": SRC_SIMULATED,
        "Road type": SRC_ROUTE,
        "Distance": SRC_ROUTE,
    }
    return features, sources
