"""Kafka producer for InteractionEvent. No-op when Kafka is not configured."""

from __future__ import annotations

import json
from typing import Callable, Optional

from app.kafka.client import TOPIC_USER_INTERACTIONS, kafka_bootstrap_servers, kafka_enabled
from app.schemas import InteractionEvent

_producer_override: Optional[Callable[[InteractionEvent], None]] = None


def set_producer(fn: Optional[Callable[[InteractionEvent], None]]) -> None:
    """Inject a fake producer for tests."""
    global _producer_override
    _producer_override = fn


def produce_interaction(event: InteractionEvent) -> None:
    if _producer_override is not None:
        _producer_override(event)
        return
    if not kafka_enabled():
        return
    try:
        from kafka import KafkaProducer  # type: ignore

        producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers(),
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        producer.send(TOPIC_USER_INTERACTIONS, event.model_dump())
        producer.flush()
        producer.close()
    except Exception:
        # Swallow broker errors so API stays fast; worker/retry is out of scope.
        return
