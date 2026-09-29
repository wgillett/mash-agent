"""Test doubles for the LLM seam."""

import inspect
from collections.abc import Awaitable, Callable, Sequence
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


class FunctionLLM:
    """Routes each call to ``handler(schema, system, user)``; safe under parallel specialists."""

    def __init__(
        self,
        handler: Callable[[type[BaseModel], str, str], BaseModel | Awaitable[BaseModel]],
        tokens: tuple[int, int] = (10, 5),
    ) -> None:
        self._handler = handler
        self._tokens = tokens
        self.calls: list[tuple[str, str]] = []  # (schema name, system prompt)

    async def generate[T: BaseModel](
        self, schema: type[T], *, system: str, user: str
    ) -> Generated[T]:
        self.calls.append((schema.__name__, system))
        out = self._handler(schema, system, user)
        if inspect.isawaitable(out):
            out = await out
        assert isinstance(out, schema)
        return Generated(out, Usage(input_tokens=self._tokens[0], output_tokens=self._tokens[1]))
