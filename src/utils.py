"""Small shared helpers: text cleaning, geometry, formatting, HTTP, logging."""
from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from src.errors import InvalidInputError, NetworkError, RoutingServiceError

LOG = logging.getLogger("road_risk")

MAX_LOCATION_LENGTH = 120
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def setup_logging() -> None:
    """Configure logging once (Cloud Run collects stdout/stderr automatically)."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=os.environ.get("LOG_LEVEL", "INFO"),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )


def sanitize_location(text: Any, field_name: str = "Location") -> str:
    """Validate and normalise a user-typed location. Raises InvalidInputError."""
    if text is None or not isinstance(text, str):
        raise InvalidInputError(f"{field_name} is required.")
    cleaned = _CONTROL_CHARS.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        raise InvalidInputError(f"{field_name} cannot be empty.")
    if len(cleaned) < 2:
        raise InvalidInputError(f"{field_name} is too short. Enter a place name such as 'Mysuru'.")
    if len(cleaned) > MAX_LOCATION_LENGTH:
        raise InvalidInputError(f"{field_name} is too long (maximum {MAX_LOCATION_LENGTH} characters).")
    if not re.search(r"[^\W\d_]", cleaned, flags=re.UNICODE):
        raise InvalidInputError(f"{field_name} must contain a place name, not only digits or symbols.")
    return cleaned


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def stable_seed(*parts: Any) -> int:
    """Deterministic integer seed from arbitrary parts (same input -> same seed on every run)."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def format_distance(km: float) -> str:
    return f"{km:.1f} km"


def format_duration(minutes: float) -> str:
    total = int(round(minutes))
    hours, mins = divmod(total, 60)
    if hours:
        return f"{hours} h {mins:02d} min"
    return f"{mins} min"


def local_now(tz_name: str | None = None) -> datetime:
    """Current local time. Falls back to fixed IST (UTC+5:30) if tz data is unavailable."""
    tz_name = tz_name or os.environ.get("APP_TIMEZONE", "Asia/Kolkata")
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo(tz_name))
    except Exception:  # pragma: no cover - depends on the OS tz database
        return datetime.now(timezone(timedelta(hours=5, minutes=30)))


def http_get_json(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
) -> Any:
    """GET a URL and return parsed JSON. Maps failures to friendly AppErrors."""
    try:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
    except requests.exceptions.Timeout as exc:
        raise NetworkError("An external service took too long to respond.", detail=str(exc)) from exc
    except requests.exceptions.RequestException as exc:
        raise NetworkError(detail=str(exc)) from exc

    if response.status_code == 429:
        raise RoutingServiceError("An external service is rate-limiting requests. Wait a minute and retry.")
    if response.status_code >= 500:
        raise RoutingServiceError(detail=f"HTTP {response.status_code} from {url.split('?')[0]}")
    if response.status_code >= 400:
        # 4xx other than 429: keep the body out of user messages; OSRM puts 'NoRoute' etc. in JSON
        try:
            return response.json()
        except ValueError:
            raise RoutingServiceError(detail=f"HTTP {response.status_code} from {url.split('?')[0]}") from None
    try:
        return response.json()
    except ValueError as exc:
        raise RoutingServiceError("An external service returned unreadable data.", detail=str(exc)) from exc


LEVEL_COLORS = {"LOW": "#2e9e6b", "MEDIUM": "#e0a100", "HIGH": "#d64545"}
