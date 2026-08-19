"""Kafka producer for InteractionEvent. No-op when Kafka is not configured."""

from __future__ import annotations

import json
import logging
from typing import Callable, Optional

from app.kafka.client import TOPIC_USER_INTERACTIONS, kafka_bootstrap_servers, kafka_enabled
from app.schemas import InteractionEvent

logger = logging.getLogger(__name__)

_SEND_TIMEOUT_S = 10

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
            key_serializer=lambda k: k.encode("utf-8") if isinstance(k, str) else k,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        # Key by userId so a user's events share a partition (ordering).
        future = producer.send(
            TOPIC_USER_INTERACTIONS,
            key=str(event.userId),
            value=event.model_dump(),
        )
        # Await broker ack so delivery errors propagate (do not discard the future).
        future.get(timeout=_SEND_TIMEOUT_S)
        producer.flush()
        producer.close()
    except Exception:
        logger.exception(
            "Failed to produce interaction event userId=%s isbn=%s",
            event.userId,
            event.isbn,
        )
        raise
