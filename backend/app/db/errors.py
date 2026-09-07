"""Database errors mapped to HTTP 503 on dependent routes."""


class DatabaseUnavailable(Exception):
    """Raised when Postgres cannot be reached."""
