class RipplError(Exception):
    """Base exception for Rippl."""
    pass

class RipplValidationError(RipplError):
    """Raised when system validation fails."""
    pass
