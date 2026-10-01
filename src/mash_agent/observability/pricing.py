"""Model pricing for cost tracking.

Prices are USD per million tokens (Anthropic first-party API list prices, as cached in the Claude
API reference on 2026-09-25). They change; override without code changes via the
``MASH_AGENT_PRICES`` environment variable, a JSON object of ``{"model-id": [input, output]}``.
A model with no known price yields ``None`` cost rather than a guess. Cached input is priced as
multiples of the input price: reads at 0.1x, 5-minute-TTL writes at 1.25x. A few models list a
different read multiple (for example 0.05x on claude-opus-5-5), so cache-heavy costs are an
estimate.
"""

import json
import os
from dataclasses import dataclass

from mash_agent.agents.models import Usage

PRICES_ENV_VAR = "MASH_AGENT_PRICES"
CACHE_READ_MULTIPLE = 0.1
CACHE_WRITE_MULTIPLE = 1.25


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float


DEFAULT_PRICES: dict[str, Price] = {
    "claude-sonnet-5-5": Price(2.0, 10.0),
    "claude-sonnet-5": Price(2.0, 10.0),
    "claude-opus-5-5": Price(4.0, 20.0),
    "claude-opus-5": Price(5.0, 25.0),
    "claude-haiku-4-5": Price(1.0, 5.0),
    "claude-fable-5-1": Price(10.0, 50.0),
}


def load_prices(env: dict[str, str] | None = None) -> dict[str, Price]:
    """Default prices merged with any overrides from the environment."""
    prices = dict(DEFAULT_PRICES)
    raw = (env if env is not None else os.environ).get(PRICES_ENV_VAR)
    if raw:
        for model, pair in json.loads(raw).items():
            prices[model] = Price(float(pair[0]), float(pair[1]))
    return prices


def cost_usd(usage: Usage, model: str, prices: dict[str, Price] | None = None) -> float | None:
    price = (prices if prices is not None else load_prices()).get(model)
    if price is None:
        return None
    uncached = usage.input_tokens - usage.cache_read_tokens - usage.cache_creation_tokens
    input_cost = price.input_per_mtok * (
        uncached
        + usage.cache_read_tokens * CACHE_READ_MULTIPLE
        + usage.cache_creation_tokens * CACHE_WRITE_MULTIPLE
    )
    return (input_cost + usage.output_tokens * price.output_per_mtok) / 1_000_000
