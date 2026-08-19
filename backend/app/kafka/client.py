"""Kafka client helpers. No live broker required for tests."""

from __future__ import annotations

import os
from typing import Optional

TOPIC_USER_INTERACTIONS = "user_interactions"


def kafka_bootstrap_servers() -> Optional[str]:
    return os.getenv("KAFKA_BOOTSTRAP_SERVERS")


def kafka_enabled() -> bool:
    return bool(kafka_bootstrap_servers())
