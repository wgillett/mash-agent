"""Structured-output LLM seam: a small protocol so agents can be tested without a model."""

import os
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from mash_agent.agents.models import Usage

DEFAULT_MODEL = "claude-sonnet-5-5"
MODEL_ENV_VAR = "MASH_AGENT_MODEL"


@dataclass
class Generated[T: BaseModel]:
    value: T
    usage: Usage


class StructuredLLM(Protocol):
    async def generate[T: BaseModel](
        self, schema: type[T], *, system: str, user: str
    ) -> Generated[T]: ...


def unpack[T: BaseModel](schema: type[T], out: dict[str, Any]) -> Generated[T]:
    """Convert LangChain's ``include_raw=True`` output into ``Generated``."""
    if (err := out.get("parsing_error")) is not None:
        raise ValueError(f"model output did not match {schema.__name__}: {err}")
    parsed = out.get("parsed")
    if not isinstance(parsed, schema):
        raise ValueError(f"model returned no structured {schema.__name__} output")
    raw = out.get("raw")
    meta = getattr(raw, "usage_metadata", None) or {}
    usage = Usage(
        input_tokens=int(meta.get("input_tokens", 0)),
        output_tokens=int(meta.get("output_tokens", 0)),
    )
    return Generated(parsed, usage)


class AnthropicLLM:
    """Claude via langchain-anthropic. Model comes from ``MASH_AGENT_MODEL`` unless given.

    Uses native structured outputs (``output_config.format``) rather than forced tool calling:
    newer models (e.g. claude-sonnet-5-5) reject ``tool_choice`` of type ``tool``/``any`` with a
    400, and langchain-anthropic's default ``function_calling`` method sends exactly that.
    """

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 4096,
        chat: BaseChatModel | None = None,
    ) -> None:
        self.model = model or os.environ.get(MODEL_ENV_VAR, DEFAULT_MODEL)
        self._chat = chat or ChatAnthropic(model=self.model, max_tokens=max_tokens)  # type: ignore[call-arg]

    async def generate[T: BaseModel](
        self, schema: type[T], *, system: str, user: str
    ) -> Generated[T]:
        runnable = self._chat.with_structured_output(schema, include_raw=True, method="json_schema")
        messages: list[BaseMessage] = [SystemMessage(system), HumanMessage(user)]
        out = await runnable.ainvoke(messages)
        return unpack(schema, dict(out))
