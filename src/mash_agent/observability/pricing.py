"""Model pricing for cost tracking.

Prices are USD per million tokens (Anthropic first-party API list prices, as cached in the Claude
API reference on 2026-09-25). They change; override without code changes via the
``MASH_AGENT_PRICES`` environment variable, a JSON object of ``{"model-id": [input, output]}``.
A model with no known price yields ``None`` cost rather than a guess. Cache-read/write pricing is
not modelled because the agents do not use prompt caching.
"""

import json
import os
from dataclasses import dataclass

from mash_agent.agents.models import Usage

PRICES_ENV_VAR = "MASH_AGENT_PRICES"


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
    return (
        usage.input_tokens * price.input_per_mtok + usage.output_tokens * price.output_per_mtok
    ) / 1_000_000
