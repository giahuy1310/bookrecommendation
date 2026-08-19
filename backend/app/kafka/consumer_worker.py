"""Kafka consumer worker: apply stub recommendations into top-picks store.

Does not need to run during tests. Ready for a later real-model swap.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict

from app.kafka.client import TOPIC_USER_INTERACTIONS, kafka_bootstrap_servers, kafka_enabled
from app.schemas import InteractionEvent
from app.services import books_search, user_lists, user_state
from app.services.recommendations_stub import generate_stub_picks
from app.services.top_picks_store import set_top_picks

logger = logging.getLogger(__name__)


def apply_interaction(event: InteractionEvent) -> bool:
    """Set context ISBN, generate stub picks, write to store (same path as POST).

    Stale events (older createdAtMs) skip context/top-picks updates but still
    append to collection/cart for ADD_TO_* types. Returns False when
    context/top-picks were skipped as stale.
    """
    if event.eventType == "ADD_TO_COLLECTION":
        user_lists.add_to_collection(event.userId, event.isbn)
    elif event.eventType == "ADD_TO_CART":
        user_lists.add_to_cart(event.userId, event.isbn)

    if not user_state.set_context_isbn(event.userId, event.isbn, event.createdAtMs):
        logger.info(
            "Ignoring stale interaction userId=%s createdAtMs=%s isbn=%s",
            event.userId,
            event.createdAtMs,
            event.isbn,
        )
        return False
    books = books_search.get_catalog()
    picks = generate_stub_picks(event.userId, event.isbn, books)
    set_top_picks(
        event.userId,
        event.isbn,
        [p.model_dump() for p in picks],
        created_at_ms=event.createdAtMs,
    )
    return True


def process_message(raw: Dict[str, Any]) -> None:
    event = InteractionEvent.model_validate(raw)
    apply_interaction(event)


def run_worker(poll_timeout_ms: int = 1000) -> None:
    """Blocking consumer loop over `user_interactions`. Not used in tests."""
    if not kafka_enabled():
        raise RuntimeError("KAFKA_BOOTSTRAP_SERVERS is not set")

    from kafka import KafkaConsumer  # type: ignore

    consumer = KafkaConsumer(
        TOPIC_USER_INTERACTIONS,
        bootstrap_servers=kafka_bootstrap_servers(),
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="bookrec-stub-worker",
        consumer_timeout_ms=poll_timeout_ms,
    )
    try:
        while True:
            for message in consumer:
                process_message(message.value)
            time.sleep(0.1)
    finally:
        consumer.close()


if __name__ == "__main__":
    run_worker()
