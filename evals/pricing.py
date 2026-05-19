from __future__ import annotations

from dataclasses import dataclass

from agent.harness import ModelUsage
from config import (
    MODEL_NAME,
    PRICE_CACHE_READ_PER_MTOK,
    PRICE_CACHE_WRITE_PER_MTOK,
    PRICE_INPUT_PER_MTOK,
    PRICE_OUTPUT_PER_MTOK,
)

PRICING_VERSION = "2026-05-19.bedrock-config-defaults"


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float
    cache_creation_per_mtok: float


PRICES: dict[tuple[str, str], ModelPrice] = {
    (
        "bedrock",
        MODEL_NAME,
    ): ModelPrice(
        input_per_mtok=PRICE_INPUT_PER_MTOK,
        output_per_mtok=PRICE_OUTPUT_PER_MTOK,
        cache_read_per_mtok=PRICE_CACHE_READ_PER_MTOK,
        cache_creation_per_mtok=PRICE_CACHE_WRITE_PER_MTOK,
    )
}


def compute_cost(usage: list[ModelUsage], *, enabled: bool = True) -> float:
    if not enabled:
        return 0.0

    total = 0.0
    for record in usage:
        key = (record.provider, record.model_id)
        price = PRICES.get(key)
        if price is None:
            raise ValueError(
                "No eval pricing configured for "
                f"provider={record.provider!r}, model_id={record.model_id!r}"
            )

        total += (
            (record.input_tokens / 1_000_000) * price.input_per_mtok
            + (record.output_tokens / 1_000_000) * price.output_per_mtok
            + (record.cache_read_tokens / 1_000_000) * price.cache_read_per_mtok
            + (record.cache_creation_tokens / 1_000_000)
            * price.cache_creation_per_mtok
        )

    return total
