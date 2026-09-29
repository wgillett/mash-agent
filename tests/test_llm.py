import pytest
from langchain_core.messages import AIMessage

from mash_agent.agents.llm import AnthropicLLM, unpack
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
