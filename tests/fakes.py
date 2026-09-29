"""Test doubles for the LLM seam."""

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from mash_agent.agents.llm import Generated
from mash_agent.agents.models import Usage


class ScriptedLLM:
    """Returns pre-scripted structured outputs in order and records every call."""

    def __init__(self, responses: Sequence[BaseModel], tokens: tuple[int, int] = (100, 20)) -> None:
        self._responses = list(responses)
        self._tokens = tokens
        self.calls: list[dict[str, Any]] = []

    async def generate[T: BaseModel](
        self, schema: type[T], *, system: str, user: str
    ) -> Generated[T]:
        self.calls.append({"schema": schema, "system": system, "user": user})
        response = self._responses.pop(0)
        assert isinstance(response, schema), f"scripted {type(response)} but asked for {schema}"
        return Generated(
            response, Usage(input_tokens=self._tokens[0], output_tokens=self._tokens[1])
        )
