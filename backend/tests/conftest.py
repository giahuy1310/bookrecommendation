"""Unit tests use in-memory stores (no live Postgres / Open Library)."""

import os

os.environ.setdefault("STORE_BACKEND", "memory")
