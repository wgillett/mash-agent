from typing import Any, cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from mash_agent.agents.llm import AnthropicLLM, OutputTruncatedError, unpack
from mash_agent.agents.models import ExtractedFindings, RawFinding


def test_unpack_reads_parsed_output_and_usage() -> None:
    parsed = ExtractedFindings(findings=[RawFinding(claim="c", source_id="s", evidence="e")])
    raw = AIMessage(
        content="", usage_metadata={"input_tokens": 12, "output_tokens": 3, "total_tokens": 15}
    )
    out = unpack(ExtractedFindings, {"raw": raw, "parsed": parsed, "parsing_error": None})
    assert out.value is parsed
    assert (out.usage.input_tokens, out.usage.output_tokens) == (12, 3)


def test_unpack_raises_on_parse_failure() -> None:
    with pytest.raises(ValueError, match="did not match"):
        unpack(ExtractedFindings, {"raw": None, "parsed": None, "parsing_error": ValueError("bad")})
    with pytest.raises(ValueError, match="no structured"):
        unpack(ExtractedFindings, {"raw": None, "parsed": None, "parsing_error": None})


def test_model_id_is_configurable_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("MASH_AGENT_MODEL", "some-model-id")
    assert AnthropicLLM().model == "some-model-id"
    assert AnthropicLLM(model="explicit").model == "explicit"


class _RecordingChat:
    """Stands in for ChatAnthropic: records with_structured_output kwargs, replays a response."""

    def __init__(self, out: dict[str, Any]) -> None:
        self.kwargs: dict[str, Any] = {}
        self._out = out

    def with_structured_output(self, schema: type, **kwargs: Any) -> "_RecordingChat":
        self.kwargs = kwargs
        return self

    async def ainvoke(self, messages: list[Any]) -> dict[str, Any]:
        self.messages = messages
        return self._out


async def test_generate_uses_native_structured_output_not_forced_tool_calling() -> None:
    parsed = ExtractedFindings(findings=[])
    raw = AIMessage(
        content="", usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
    )
    chat = _RecordingChat({"raw": raw, "parsed": parsed, "parsing_error": None})
    llm = AnthropicLLM(model="m", chat=cast(BaseChatModel, chat))
    out = await llm.generate(ExtractedFindings, system="sys", user="usr")
    # function_calling (the langchain default) forces tool_choice, which claude-sonnet-5-5 rejects
    assert chat.kwargs == {"include_raw": True, "method": "json_schema"}
    assert out.value is parsed and out.usage.input_tokens == 5
    assert [m.content for m in chat.messages] == ["sys", "usr"]


def test_unpack_flags_truncated_output_before_parse_error() -> None:
    raw = AIMessage(content="", response_metadata={"stop_reason": "max_tokens"})
    out = {"raw": raw, "parsed": None, "parsing_error": ValueError("Field required")}
    with pytest.raises(OutputTruncatedError, match="cut off at max_tokens"):
        unpack(ExtractedFindings, out)


def test_unpack_reads_cache_tokens() -> None:
    parsed = ExtractedFindings(findings=[])
    raw = AIMessage(
        content="",
        usage_metadata={
            "input_tokens": 1000,
            "output_tokens": 3,
            "total_tokens": 1003,
            "input_token_details": {"cache_read": 800, "cache_creation": 100},
        },
    )
    out = unpack(ExtractedFindings, {"raw": raw, "parsed": parsed, "parsing_error": None})
    assert (out.usage.cache_read_tokens, out.usage.cache_creation_tokens) == (800, 100)
