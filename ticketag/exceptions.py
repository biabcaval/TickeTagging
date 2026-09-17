"""Exception hierarchy for the TickeTag classification flow."""


class TickeTagError(Exception):
    """Base class for every error raised by TickeTag."""


class ConfigurationError(TickeTagError):
    """Raised when required settings are missing or invalid."""


class EmptyTicketError(TickeTagError):
    """Raised when a ticket holds no usable text."""


class InferenceError(TickeTagError):
    """Raised when the model endpoint cannot be reached or keeps failing."""


class ClassificationError(TickeTagError):
    """Raised when the model reply cannot be mapped to a valid category."""
