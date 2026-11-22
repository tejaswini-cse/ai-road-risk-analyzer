"""Geocoding and route alternatives from free/open services, with a clean fallback.

Live sources (no API key, no billing):
* Nominatim (OpenStreetMap) - place name -> coordinates. Usage policy: max 1 request/second,
  identify the app in the User-Agent. Set NOMINATIM_EMAIL / NOMINATIM_USER_AGENT.
* OSRM public demo server - driving routes + alternatives. It is a shared DEMO server with no
  uptime guarantee. For anything beyond a demo, self-host OSRM or use another provider
  (set OSRM_BASE_URL).

Fallbacks (clearly flagged in the UI):
* If geocoding is unreachable, a small built-in gazetteer of Indian cities is used.
* If routing is unreachable, smooth approximate alternatives are generated from the straight-line
  distance. These are NOT real roads; they exist so the demo still runs offline.
"""
from __future__ import annotations

import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src.errors import (
    AppError,
    InvalidInputError,
    LocationNotFoundError,
    NetworkError,
    RouteUnavailableError,
    RoutingServiceError,
)
from src.utils import LOG, haversine_km, http_get_json

HttpGet = Callable[..., Any]

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OSRM_BASE_URL = os.environ.get("OSRM_BASE_URL", "https://router.project-osrm.org")
MAX_ROUTES = 3
_NOMINATIM_MIN_INTERVAL_S = 1.1


@dataclass(frozen=True)
class Location:
    query: str
    lat: float
    lon: float
    display_name: str
    source: str = "nominatim"  # nominatim | offline-gazetteer


@dataclass
class Route:
    route_id: str
    name: str
    distance_km: float
    duration_min: float
    coordinates: list[tuple[float, float]]  # (lat, lon)
    highway_share: float = 0.0              # share of distance on motorway-class / NH roads (0..1)
    source: str = "osrm"                    # osrm | osrm-via | fallback
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def avg_speed_kmh(self) -> float:
        return self.distance_km / (self.duration_min / 60.0) if self.duration_min > 0 else 0.0

    @property
    def road_type(self) -> str:
        """Heuristic dominant road type (documented limitation: derived from road classes + speed)."""
        speed = self.avg_speed_kmh
        if self.highway_share >= 0.5 or speed >= 60:
            return "highway"
        if speed >= 42:
            return "arterial"
        if speed >= 25:
            return "rural" if self.distance_km >= 15 else "urban"
        return "urban"


# ----------------------------------------------------------------------------------
# Geocoding
# ----------------------------------------------------------------------------------

GAZETTEER: dict[str, tuple[float, float, str]] = {
    "bengaluru": (12.9716, 77.5946, "Bengaluru, Karnataka, India"),
    "mysuru": (12.2958, 76.6394, "Mysuru, Karnataka, India"),
    "mangaluru": (12.9141, 74.8560, "Mangaluru, Karnataka, India"),
    "hubballi": (15.3647, 75.1240, "Hubballi, Karnataka, India"),
    "belagavi": (15.8497, 74.4977, "Belagavi, Karnataka, India"),
    "tumakuru": (13.3392, 77.1140, "Tumakuru, Karnataka, India"),
    "mandya": (12.5218, 76.8951, "Mandya, Karnataka, India"),
    "hassan": (13.0033, 76.1004, "Hassan, Karnataka, India"),
    "shivamogga": (13.9299, 75.5681, "Shivamogga, Karnataka, India"),
    "davanagere": (14.4644, 75.9218, "Davanagere, Karnataka, India"),
    "ballari": (15.1394, 76.9214, "Ballari, Karnataka, India"),
    "kalaburagi": (17.3297, 76.8343, "Kalaburagi, Karnataka, India"),
    "vijayapura": (16.8302, 75.7100, "Vijayapura, Karnataka, India"),
    "udupi": (13.3409, 74.7421, "Udupi, Karnataka, India"),
    "madikeri": (12.4244, 75.7382, "Madikeri, Karnataka, India"),
    "chennai": (13.0827, 80.2707, "Chennai, Tamil Nadu, India"),
    "hyderabad": (17.3850, 78.4867, "Hyderabad, Telangana, India"),
    "kochi": (9.9312, 76.2673, "Kochi, Kerala, India"),
    "pune": (18.5204, 73.8567, "Pune, Maharashtra, India"),
    "mumbai": (19.0760, 72.8777, "Mumbai, Maharashtra, India"),
    "delhi": (28.6139, 77.2090, "Delhi, India"),
}
_ALIASES = {"bangalore": "bengaluru", "mysore": "mysuru", "mangalore": "mangaluru", "hubli": "hubballi",
            "belgaum": "belagavi", "bijapur": "vijayapura", "bellary": "ballari", "gulbarga": "kalaburagi",
            "shimoga": "shivamogga", "davangere": "davanagere", "mercara": "madikeri", "cochin": "kochi",
            "new delhi": "delhi", "bombay": "mumbai", "madras": "chennai"}


def lookup_gazetteer(query: str) -> Location | None:
    key = re.sub(r"[^a-z ]", "", query.lower()).strip()
    key = key.split(",")[0].strip() if "," in query else key
    key = _ALIASES.get(key, key)
    entry = GAZETTEER.get(key)
    if not entry:
        return None
    return Location(query=query, lat=entry[0], lon=entry[1], display_name=entry[2], source="offline-gazetteer")


_geocode_cache: dict[str, Location] = {}
_nominatim_lock = threading.Lock()
_last_nominatim_call = 0.0


def _user_agent() -> str:
    ua = os.environ.get("NOMINATIM_USER_AGENT")
    if ua:
        return ua
    contact = os.environ.get("NOMINATIM_EMAIL", "set-NOMINATIM_EMAIL-env-var")
    return f"AI-Road-Risk-Analyzer-college-prototype/1.0 ({contact})"


def geocode(query: str, http_get: HttpGet = http_get_json) -> Location:
    """Resolve a (pre-sanitised) place name to coordinates."""
    if not query or not query.strip():
        raise InvalidInputError("Location cannot be empty.")
    cache_key = query.strip().lower()
    if cache_key in _geocode_cache:
        return _geocode_cache[cache_key]

    global _last_nominatim_call
    try:
        with _nominatim_lock:
            wait = _NOMINATIM_MIN_INTERVAL_S - (time.monotonic() - _last_nominatim_call)
            if wait > 0 and _last_nominatim_call:
                time.sleep(wait)
            try:
                payload = http_get(
                    NOMINATIM_URL,
                    params={"q": query, "format": "jsonv2", "limit": 1},
                    headers={"User-Agent": _user_agent(), "Accept-Language": "en"},
                    timeout=10,
                )
            finally:
                _last_nominatim_call = time.monotonic()
    except (NetworkError, RoutingServiceError) as exc:
        LOG.warning("Geocoding service unavailable (%s); trying offline gazetteer", exc)
        fallback = lookup_gazetteer(query)
        if fallback:
            _geocode_cache[cache_key] = fallback
            return fallback
        raise RoutingServiceError(
            "The location search service is unreachable and this place is not in the offline list. "
            "Try again later, or try a major city such as Bengaluru or Mysuru.",
            detail=str(exc),
        ) from exc

    if not isinstance(payload, list):
        raise RoutingServiceError("The location search service returned unexpected data.", detail=repr(payload)[:200])
    if not payload:
        raise LocationNotFoundError(
            f"Could not find '{query}'. Check the spelling or add the state/country, e.g. 'Mysuru, Karnataka'."
        )
    try:
        item = payload[0]
        lat, lon = float(item["lat"]), float(item["lon"])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("coordinates out of range")
        loc = Location(query=query, lat=lat, lon=lon, display_name=str(item.get("display_name", query)))
    except (KeyError, TypeError, ValueError) as exc:
        raise RoutingServiceError("The location search service returned unreadable data.", detail=str(exc)) from exc
    _geocode_cache[cache_key] = loc
    return loc


# ----------------------------------------------------------------------------------
# OSRM parsing
# ----------------------------------------------------------------------------------

_NH_REF = re.compile(r"^(NH|AH|E)\s?-?\d", re.IGNORECASE)


def _analyse_steps(osrm_route: dict) -> tuple[float, str]:
    """Return (highway_share, dominant road label) from a route's steps."""
    total = hw = 0.0
    by_label: dict[str, float] = {}
    for leg in osrm_route.get("legs", []) or []:
        for step in leg.get("steps", []) or []:
            dist = float(step.get("distance") or 0.0)
            if dist <= 0:
                continue
            total += dist
            ref = (step.get("ref") or "").split(";")[0].strip()
            name = (step.get("name") or "").strip()
            classes = {c for inter in (step.get("intersections") or []) for c in (inter.get("classes") or [])}
            if "motorway" in classes or (ref and _NH_REF.match(ref)):
                hw += dist
            label = ref or name
            if label:
                by_label[label] = by_label.get(label, 0.0) + dist
    share = hw / total if total > 0 else 0.0
    label = max(by_label, key=by_label.get) if by_label else ""
    return share, label


def parse_osrm_routes(payload: Any, source: str = "osrm") -> list[Route]:
    """Convert an OSRM /route response into Route objects. Raises on NoRoute / bad payload."""
    if not isinstance(payload, dict):
        raise RoutingServiceError("The routing service returned unexpected data.", detail=type(payload).__name__)
    code = payload.get("code")
    if code in ("NoRoute", "NoSegment"):
        raise RouteUnavailableError()
    if code != "Ok":
        raise RoutingServiceError(detail=f"OSRM code={code!r} message={payload.get('message')!r}")

    routes: list[Route] = []
    for i, r in enumerate(payload.get("routes") or []):
        try:
            coords = [(float(lat), float(lon)) for lon, lat in r["geometry"]["coordinates"]]
            distance_km = float(r["distance"]) / 1000.0
            duration_min = float(r["duration"]) / 60.0
        except (KeyError, TypeError, ValueError):
            LOG.warning("Skipping malformed OSRM route #%d", i)
            continue
        if distance_km <= 0 or len(coords) < 2:
            continue
        share, label = _analyse_steps(r)
        routes.append(
            Route(
                route_id="?", name=label, distance_km=distance_km, duration_min=duration_min,
                coordinates=coords, highway_share=share, source=source,
            )
        )
    if not routes:
        raise RouteUnavailableError()
    return routes


def _overlap(a: Route, b: Route, samples: int = 25, threshold_km: float = 0.4) -> float:
    """Fraction of sampled points of `a` that lie within threshold_km of a vertex of `b`."""
    step = max(1, len(a.coordinates) // samples)
    pts = a.coordinates[::step]
    b_pts = b.coordinates[:: max(1, len(b.coordinates) // 400)]
    near = sum(1 for p in pts if any(haversine_km(p[0], p[1], q[0], q[1]) <= threshold_km for q in b_pts))
    return near / len(pts)


def _is_distinct(candidate: Route, existing: list[Route]) -> bool:
    for e in existing:
        if abs(candidate.distance_km - e.distance_km) / max(e.distance_km, 0.1) < 0.01:
            return False
        if _overlap(candidate, e) > 0.92 and _overlap(e, candidate) > 0.92:
            return False
    return True


def _via_point(start: Location, end: Location, side: int, fraction: float = 0.18) -> tuple[float, float]:
    """A waypoint offset sideways from the midpoint, used to coax OSRM into a different road."""
    mid_lat, mid_lon = (start.lat + end.lat) / 2, (start.lon + end.lon) / 2
    dlat, dlon = end.lat - start.lat, end.lon - start.lon
    # perpendicular in lat/lon space (adequate for this purpose)
    return (mid_lat + side * fraction * -dlon, mid_lon + side * fraction * dlat)


def _osrm_url(points: list[tuple[float, float]]) -> str:
    coords = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in points)
    return f"{OSRM_BASE_URL}/route/v1/driving/{coords}"


def fetch_routes(start: Location, end: Location, http_get: HttpGet = http_get_json) -> list[Route]:
    """Up to MAX_ROUTES distinct driving routes. Route A is the shortest.

    Raises RouteUnavailableError if OSRM says there is no route, or NetworkError /
    RoutingServiceError if the service cannot be reached (callers may then use fallback_routes).
    """
    base_params = {"overview": "full", "geometries": "geojson", "steps": "true"}
    primary = http_get(
        _osrm_url([(start.lat, start.lon), (end.lat, end.lon)]),
        params={**base_params, "alternatives": "true"},
        timeout=20,
    )
    found = parse_osrm_routes(primary)

    routes: list[Route] = []
    for r in sorted(found, key=lambda x: x.distance_km):
        if _is_distinct(r, routes):
            routes.append(r)

    # OSRM often returns a single route; try up to two via-point alternatives.
    if len(routes) < 2:
        shortest = routes[0].distance_km
        for side in (1, -1):
            if len(routes) >= MAX_ROUTES:
                break
            try:
                via = _via_point(start, end, side)
                payload = http_get(
                    _osrm_url([(start.lat, start.lon), via, (end.lat, end.lon)]),
                    params=base_params,
                    timeout=20,
                )
                for cand in parse_osrm_routes(payload, source="osrm-via"):
                    if cand.distance_km <= shortest * 1.8 and _is_distinct(cand, routes):
                        routes.append(cand)
                        break
            except AppError as exc:
                LOG.info("Via-point alternative failed (ignored): %s", exc)

    routes = sorted(routes, key=lambda x: x.distance_km)[:MAX_ROUTES]
    for letter, r in zip("ABC", routes):
        r.route_id = letter
        r.name = f"Route {letter}" + (f" via {r.name}" if r.name else "")
    return routes


# ----------------------------------------------------------------------------------
# Offline fallback routes (approximate, NOT real roads)
# ----------------------------------------------------------------------------------

def _bezier(p0, p1, p2, n: int = 40) -> list[tuple[float, float]]:
    pts = []
    for i in range(n + 1):
        t = i / n
        lat = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
        lon = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
        pts.append((lat, lon))
    return pts


def _polyline_km(pts: list[tuple[float, float]]) -> float:
    return sum(haversine_km(a[0], a[1], b[0], b[1]) for a, b in zip(pts, pts[1:]))


def fallback_routes(start: Location, end: Location) -> list[Route]:
    """Three approximate alternatives derived from the straight-line distance.

    Distances use a typical road/straight-line factor (about 1.2-1.4) and road mixes that
    differ per alternative so the comparison logic can be demonstrated offline.
    """
    straight = haversine_km(start.lat, start.lon, end.lat, end.lon)
    if straight < 0.2:
        raise RouteUnavailableError("The start and destination are too close together to compare routes.")
    p0, p2 = (start.lat, start.lon), (end.lat, end.lon)
    specs = [  # (curve offset, speed km/h, highway share, road/straight-line factor, label)
        (0.0, 62.0, 0.65, 1.12, "main highway (approx.)"),
        (0.14, 52.0, 0.25, 1.17, "state road (approx.)"),
        (-0.24, 40.0, 0.0, 1.22, "country roads (approx.)"),
    ]
    routes: list[Route] = []
    for letter, (off, speed, share, factor, label) in zip("ABC", specs):
        mid = ((p0[0] + p2[0]) / 2 + off * -(p2[1] - p0[1]), (p0[1] + p2[1]) / 2 + off * (p2[0] - p0[0]))
        pts = _bezier(p0, mid, p2)
        km = _polyline_km(pts) * factor  # allowance for road winding
        routes.append(
            Route(
                route_id=letter, name=f"Route {letter} via {label}", distance_km=round(km, 1),
                duration_min=km / speed * 60.0, coordinates=pts, highway_share=share, source="fallback",
            )
        )
    return sorted(routes, key=lambda r: r.distance_km)
