"""Application-specific exceptions.

Every exception carries a `user_message` that is safe to show in the UI
(no credentials, no stack traces, no internal identifiers).
"""
from __future__ import annotations


class AppError(Exception):
    """Base class. `user_message` is what the end user sees."""

    default_message = "Something went wrong. Please try again."

    def __init__(self, user_message: str | None = None, *, detail: str | None = None):
        self.user_message = user_message or self.default_message
        self.detail = detail  # technical detail for logs only, never shown in the UI
        super().__init__(self.user_message if detail is None else f"{self.user_message} ({detail})")


class InvalidInputError(AppError):
    default_message = "Please check the locations you entered."


class LocationNotFoundError(AppError):
    default_message = "That location could not be found. Try a more specific place name."


class NetworkError(AppError):
    default_message = "A network problem occurred while contacting an external service."


class RoutingServiceError(AppError):
    default_message = "The routing service is not responding properly."


class RouteUnavailableError(AppError):
    default_message = "No drivable route could be found between these two locations."


class DataError(AppError):
    default_message = "Required data is missing or invalid."


class ConfigurationError(AppError):
    default_message = "The AI service is not configured correctly."


class AuthenticationError(AppError):
    default_message = "The application is not authorised to use the AI service."


class AIServiceError(AppError):
    default_message = "The AI service could not produce a prediction right now."


class InvalidPredictionError(AIServiceError):
    default_message = "The AI service returned an invalid prediction."


class InvalidFeatureError(AppError, ValueError):
    default_message = "Route features are invalid."
