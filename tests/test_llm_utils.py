from collections import Counter

import httpx as _httpx
import litellm as _litellm
import pytest
from litellm.main import ModelResponse, Usage

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool, as_tool
from tau2.utils import llm_utils as _llm_utils
from tau2.utils.llm_utils import (
    DEFAULT_HTTPX_MAX_CONNECTIONS,
    HTTPX_MAX_CONNECTIONS_ENV,
    HTTPX_MAX_KEEPALIVE_CONNECTIONS_ENV,
    PROMPT_CACHING_DISABLED_ENV,
    _build_httpx_limits,
    apply_prompt_caching,
    check_openrouter_kwargs,
    generate,
    get_response_cost,
    get_response_usage,
    normalize_model_name,
    openrouter_provider_assignments,
    openrouter_provider_for_key,
    openrouter_provider_pin_disabled,
    parse_openrouter_providers,
    with_openrouter_provider_pin,
    with_openrouter_session_id,
)


@pytest.fixture
def model() -> str:
    return "openrouter/openai/gpt-4o-mini"


@pytest.fixture
def messages() -> list[Message]:
    messages = [
        SystemMessage(role="system", content="You are a helpful assistant."),
        UserMessage(role="user", content="What is the capital of the moon?"),
    ]
    return messages


@pytest.fixture
def tool() -> Tool:
    def calculate_square(x: int) -> int:
        """Calculate the square of a number.
            Args:
            x (int): The number to calculate the square of.
        Returns:
            int: The square of the number.
        """
        return x * x

    return as_tool(calculate_square)


@pytest.fixture
def tool_call_messages() -> list[Message]:
    messages = [
        SystemMessage(role="system", content="You are a helpful assistant."),
        UserMessage(
            role="user",
            content="What is the square of 5? Just give me the number, no explanation.",
        ),
    ]
    return messages


def test_generate_no_tool_call(model: str, messages: list[Message]):
    response = generate(model, messages)
    assert isinstance(response, AssistantMessage)
    assert response.content is not None


def test_generate_tool_call(model: str, tool_call_messages: list[Message], tool: Tool):
    response = generate(model, tool_call_messages, tools=[tool])
    assert isinstance(response, AssistantMessage)
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "calculate_square"
    assert response.tool_calls[0].arguments == {"x": 5}
    follow_up_messages = [
        response,
        ToolMessage(role="tool", id=response.tool_calls[0].id, content="25"),
    ]
    response = generate(
        model,
        tool_call_messages + follow_up_messages,
        tools=[tool],
    )
    assert isinstance(response, AssistantMessage)
    assert response.tool_calls is None
    assert response.content == "25"


# --- prompt caching ---------------------------------------------------------


def _cached(blocks) -> bool:
    return isinstance(blocks, list) and any(
        isinstance(b, dict) and "cache_control" in b for b in blocks
    )


def _convo() -> list[dict]:
    return [
        {"role": "system", "content": "policy text"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "t0"}]},
        {"role": "tool", "content": "tool output", "tool_call_id": "t0"},
    ]


def test_prompt_caching_marks_system_last_user_and_trailing_tool_result():
    """System (which covers the tool schemas too), the last user turn and a
    trailing tool result get a breakpoint; the assistant tool call is untouched."""
    out = apply_prompt_caching("anthropic/claude-opus-4-6", _convo())
    assert _cached(out[0]["content"])
    assert _cached(out[1]["content"])
    assert out[2]["content"] is None
    assert out[3]["content"] == [
        {"type": "text", "text": "tool output", "cache_control": {"type": "ephemeral"}}
    ]
    assert out[3]["tool_call_id"] == "t0"


def test_prompt_caching_marks_only_the_last_tool_result():
    convo = _convo() + [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "t1"}]},
        {"role": "tool", "content": "second output", "tool_call_id": "t1"},
    ]
    out = apply_prompt_caching("anthropic/claude-opus-4-6", convo)
    assert out[3]["content"] == "tool output"
    assert _cached(out[5]["content"])


def test_prompt_caching_leaves_tool_results_alone_when_user_is_last():
    convo = _convo() + [
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "thanks"},
    ]
    out = apply_prompt_caching("anthropic/claude-opus-4-6", convo)
    assert out[3]["content"] == "tool output"
    assert not _cached(out[1]["content"])
    assert _cached(out[5]["content"])


def test_prompt_caching_marks_only_the_last_user_turn():
    convo = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ]
    out = apply_prompt_caching("anthropic/claude-opus-4-6", convo)
    assert not _cached(out[1]["content"])
    assert _cached(out[3]["content"])


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-5.4",
        "gpt-4.1-2025-04-14",
        "gemini/gemini-3.5-flash",
        "gpt-3.5-turbo",  # used by tests/test_run.py -- must stay a no-op
        "openrouter/openai/gpt-5.4:exacto",
    ],
)
def test_prompt_caching_is_anthropic_only(model):
    """Anthropic-style cache_control must never reach another provider."""
    assert apply_prompt_caching(model, _convo()) == _convo()


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-opus-4-8",
        "claude-opus-5",
        "bedrock/anthropic.claude-opus-4-6",
        "openrouter/anthropic/claude-opus-4.6:exacto",
    ],
)
def test_prompt_caching_recognizes_anthropic_spellings(model):
    assert _cached(apply_prompt_caching(model, _convo())[0]["content"])


def test_prompt_caching_kill_switch(monkeypatch):
    monkeypatch.setenv(PROMPT_CACHING_DISABLED_ENV, "1")
    assert apply_prompt_caching("anthropic/claude-opus-4-6", _convo()) == _convo()


def test_prompt_caching_skips_empty_content():
    convo = [{"role": "system", "content": "   "}, {"role": "user", "content": "hi"}]
    out = apply_prompt_caching("anthropic/claude-opus-4-6", convo)
    assert out[0]["content"] == "   "
    assert _cached(out[1]["content"])


def test_prompt_caching_is_idempotent():
    once = apply_prompt_caching("anthropic/claude-opus-4-6", _convo())
    assert apply_prompt_caching("anthropic/claude-opus-4-6", once) == once


# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model, expected",
    [
        (
            "openrouter/anthropic/claude-opus-4.6",
            "openrouter/anthropic/claude-opus-4.6:exacto",
        ),
        ("openrouter/openai/gpt-4.1:exacto", "openrouter/openai/gpt-4.1:exacto"),
        ("openrouter/z-ai/glm-4.6:nitro", "openrouter/z-ai/glm-4.6:nitro"),
        ("openrouter/openai/gpt-4o-mini:free", "openrouter/openai/gpt-4o-mini:free"),
        ("anthropic/claude-opus-4-6", "anthropic/claude-opus-4-6"),
        ("openai/gpt-4o", "openai/gpt-4o"),
        ("gpt-4.1-2025-04-14", "gpt-4.1-2025-04-14"),
    ],
)
def test_normalize_model_name(model, expected):
    assert normalize_model_name(model) == expected
    assert normalize_model_name(normalize_model_name(model)) == expected


@pytest.mark.parametrize(
    "model, kwargs",
    [
        ("anthropic/claude-opus-4-8", {"reasoning_effort": "low"}),
        ("anthropic/claude-opus-4-6", {"thinking": {"type": "disabled"}}),
        ("anthropic/claude-opus-4-8", {"output_config": {"effort": "low"}}),
        ("openrouter/anthropic/claude-opus-4.6:exacto", {"temperature": 0.0}),
        (
            "openrouter/anthropic/claude-opus-4.6:exacto",
            {"reasoning": {"effort": "low"}},
        ),
        ("openrouter/qwen/qwen3.6-27b:exacto", {"reasoning": {"enabled": False}}),
        (
            "openrouter/anthropic/claude-opus-4.6:exacto",
            {"extra_body": {"reasoning": {"max_tokens": 2048}}},
        ),
    ],
)
def test_openrouter_kwargs_check_accepts(model, kwargs):
    check_openrouter_kwargs(model, kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reasoning_effort": "low"},
        {"thinking": {"type": "enabled", "budget_tokens": 2048}},
        {"thinking": {"type": "disabled"}},
        {"output_config": {"effort": "low"}},
        {
            "temperature": 0.0,
            "reasoning_effort": "low",
            "thinking": {"type": "adaptive"},
        },
    ],
)
def test_openrouter_kwargs_check_rejects_vendor_native_reasoning(kwargs):
    """OpenRouter silently ignores these on models it does not map them for, so
    they must fail loudly and point at the unified `reasoning` object."""
    with pytest.raises(ValueError, match="reasoning"):
        check_openrouter_kwargs("openrouter/anthropic/claude-opus-4.6:exacto", kwargs)


def test_openrouter_session_id_is_added_per_simulation():
    args = {"reasoning": {"enabled": False}}
    out = with_openrouter_session_id(
        "openrouter/anthropic/claude-opus-5", args, "sim-1"
    )
    assert out == {"reasoning": {"enabled": False}, "session_id": "sim-1"}
    assert "session_id" not in args  # input is not mutated


@pytest.mark.parametrize(
    "model", ["claude-opus-5", "gpt-5.6", "vertex_ai/gemini-3-pro"]
)
def test_openrouter_session_id_skips_other_providers(model):
    assert with_openrouter_session_id(model, {"temperature": 0.5}, "sim-1") == {
        "temperature": 0.5
    }
    assert with_openrouter_session_id(model, None, "sim-1") == {}


def test_openrouter_session_id_respects_explicit_value():
    out = with_openrouter_session_id(
        "openrouter/openai/gpt-5.6-sol", {"session_id": "mine"}, "sim-1"
    )
    assert out["session_id"] == "mine"


def test_openrouter_session_id_kill_switch(monkeypatch):
    monkeypatch.setenv("TAU2_DISABLE_OPENROUTER_SESSION_ID", "1")
    assert with_openrouter_session_id(
        "openrouter/anthropic/claude-opus-5", {"temperature": 0.5}, "sim-1"
    ) == {"temperature": 0.5}


def test_run_config_rejects_vendor_native_reasoning_for_openrouter():
    from pydantic import ValidationError

    from tau2.data_model.simulation import TextRunConfig

    with pytest.raises(ValidationError, match="reasoning"):
        TextRunConfig(
            domain="airline",
            llm_agent="openrouter/anthropic/claude-opus-4.8",
            llm_args_agent={"temperature": 0.0, "reasoning_effort": "low"},
        )
    with pytest.raises(ValidationError, match="reasoning"):
        TextRunConfig(
            domain="airline",
            llm_user="openrouter/qwen/qwen3.6-27b",
            llm_args_user={"thinking": {"type": "disabled"}},
        )
    # direct-provider names keep upstream behaviour
    cfg = TextRunConfig(
        domain="airline",
        llm_agent="anthropic/claude-opus-4-8",
        llm_args_agent={"output_config": {"effort": "low"}},
    )
    assert cfg.llm_agent == "anthropic/claude-opus-4-8"
    # the unified object is fine, and the sent name is recorded
    cfg = TextRunConfig(
        domain="airline",
        llm_agent="openrouter/anthropic/claude-opus-4.8",
        llm_args_agent={"reasoning": {"effort": "low"}},
    )
    assert cfg.llm_agent == "openrouter/anthropic/claude-opus-4.8:exacto"


def _fake_response(
    usage: dict, model: str = "anthropic/claude-opus-4.6"
) -> ModelResponse:
    return ModelResponse(
        model=model,
        choices=[
            {"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
        ],
        usage=Usage(**usage),
    )


def test_response_cost_prefers_provider_billed_cost():
    """OpenRouter reports the charged amount as usage.cost; litellm's price map
    would either mis-estimate it or raise for a `:exacto` slug."""
    resp = _fake_response(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "total_tokens": 1100,
            "cost": 0.00123,
        },
        model="anthropic/claude-opus-4.6:exacto",
    )
    assert get_response_cost(resp) == pytest.approx(0.00123)
    assert get_response_cost(resp, skip_cost=True) == 0.0


def test_response_cost_falls_back_to_price_map_without_provider_cost():
    resp = _fake_response(
        {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100}
    )
    resp.model = "gpt-4o-mini"
    assert get_response_cost(resp) > 0.0


def test_response_usage_reads_openrouter_cache_fields():
    resp = _fake_response(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "total_tokens": 1100,
            "prompt_tokens_details": {"cached_tokens": 50, "cache_write_tokens": 20},
        }
    )
    assert get_response_usage(resp) == {
        "completion_tokens": 100,
        "prompt_tokens": 1000,
        "cache_creation_input_tokens": 20,
        "cache_read_input_tokens": 50,
    }


def test_response_usage_prefers_anthropic_native_cache_fields():
    resp = _fake_response(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "total_tokens": 1100,
            "cache_creation_input_tokens": 7,
            "cache_read_input_tokens": 9,
            "prompt_tokens_details": {"cached_tokens": 50, "cache_write_tokens": 20},
        }
    )
    usage = get_response_usage(resp)
    assert usage["cache_creation_input_tokens"] == 7
    assert usage["cache_read_input_tokens"] == 9


def test_response_usage_records_reasoning_tokens_when_reported():
    resp = _fake_response(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "total_tokens": 1100,
            "completion_tokens_details": {"reasoning_tokens": 42},
        }
    )
    assert get_response_usage(resp)["reasoning_tokens"] == 42


def test_response_usage_omits_reasoning_tokens_when_absent():
    resp = _fake_response(
        {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100}
    )
    assert "reasoning_tokens" not in get_response_usage(resp)


# ---------------------------------------------------------------------------
# httpx connection pool limits
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_pool_env(monkeypatch):
    """Both pool env vars unset, whatever the caller's shell had."""
    monkeypatch.delenv(HTTPX_MAX_CONNECTIONS_ENV, raising=False)
    monkeypatch.delenv(HTTPX_MAX_KEEPALIVE_CONNECTIONS_ENV, raising=False)


def test_httpx_limits_default_is_large_and_fully_keepalive(clean_pool_env):
    """Unset env: a pool big enough that --max-concurrency, not the socket
    pool, is the binding constraint; every connection may be kept alive."""
    limits = _build_httpx_limits()
    assert limits.max_connections == DEFAULT_HTTPX_MAX_CONNECTIONS
    assert limits.max_keepalive_connections == DEFAULT_HTTPX_MAX_CONNECTIONS
    # ~2 endpoints per simulation, so the default must clear the concurrencies
    # we actually run (64, and headroom well past 192).
    assert DEFAULT_HTTPX_MAX_CONNECTIONS >= 2 * 192


def _pool_of(client):
    """The connection pool httpx actually built, or None if internals moved."""
    return getattr(getattr(client, "_transport", None), "_pool", None)


def test_module_level_limits_are_installed_on_the_litellm_sessions():
    """The sync and async sessions litellm uses carry our limits."""
    from tau2.utils import llm_utils

    assert llm_utils.httpx_limits.max_connections == DEFAULT_HTTPX_MAX_CONNECTIONS
    assert (
        llm_utils.httpx_limits.max_keepalive_connections
        == DEFAULT_HTTPX_MAX_CONNECTIONS
    )
    for client in (llm_utils.litellm.client_session, llm_utils.litellm.aclient_session):
        pool = _pool_of(client)
        if pool is None:  # pragma: no cover - httpx internals changed
            pytest.skip("httpx transport internals changed")
        assert pool._max_connections == DEFAULT_HTTPX_MAX_CONNECTIONS
        assert pool._max_keepalive_connections == DEFAULT_HTTPX_MAX_CONNECTIONS


def test_httpx_limits_from_env(clean_pool_env, monkeypatch):
    monkeypatch.setenv(HTTPX_MAX_CONNECTIONS_ENV, "256")
    limits = _build_httpx_limits()
    assert limits.max_connections == 256
    # keep-alive follows max_connections when it is not set explicitly
    assert limits.max_keepalive_connections == 256


def test_httpx_limits_keepalive_env_is_independent(clean_pool_env, monkeypatch):
    monkeypatch.setenv(HTTPX_MAX_CONNECTIONS_ENV, "256")
    monkeypatch.setenv(HTTPX_MAX_KEEPALIVE_CONNECTIONS_ENV, "32")
    limits = _build_httpx_limits()
    assert limits.max_connections == 256
    assert limits.max_keepalive_connections == 32


@pytest.mark.parametrize("value", ["0", "none", "UNLIMITED", " none "])
def test_httpx_limits_env_can_request_unlimited(clean_pool_env, monkeypatch, value):
    monkeypatch.setenv(HTTPX_MAX_CONNECTIONS_ENV, value)
    limits = _build_httpx_limits()
    assert limits.max_connections is None
    assert limits.max_keepalive_connections is None


@pytest.mark.parametrize("value", ["abc", "-5", "12.5", "1e3", "  "])
def test_httpx_limits_invalid_env_falls_back_to_default(
    clean_pool_env, monkeypatch, value
):
    """A typo must degrade to the default, never to a small or zero pool."""
    monkeypatch.setenv(HTTPX_MAX_CONNECTIONS_ENV, value)
    limits = _build_httpx_limits()
    assert limits.max_connections == DEFAULT_HTTPX_MAX_CONNECTIONS
    assert limits.max_keepalive_connections == DEFAULT_HTTPX_MAX_CONNECTIONS


def test_httpx_limits_invalid_keepalive_env_falls_back_to_max_connections(
    clean_pool_env, monkeypatch
):
    monkeypatch.setenv(HTTPX_MAX_CONNECTIONS_ENV, "128")
    monkeypatch.setenv(HTTPX_MAX_KEEPALIVE_CONNECTIONS_ENV, "not-a-number")
    limits = _build_httpx_limits()
    assert limits.max_connections == 128
    assert limits.max_keepalive_connections == 128


# ---------------------------------------------------------------------------
# Per-task OpenRouter provider pin
# ---------------------------------------------------------------------------

OPUS = "openrouter/anthropic/claude-opus-5:exacto"
POOL = ["amazon-bedrock", "azure", "google-vertex"]


def test_provider_pin_sets_single_provider_order():
    args = {"reasoning": {"enabled": False}}
    out = with_openrouter_provider_pin(OPUS, args, "airline_12", POOL)
    assert out["reasoning"] == {"enabled": False}
    assert out["provider"]["order"] == [openrouter_provider_for_key("airline_12", POOL)]
    assert len(out["provider"]["order"]) == 1
    assert out["provider"]["allow_fallbacks"] is False
    assert "provider" not in args  # input is not mutated


def test_provider_pin_allow_fallbacks_is_opt_in():
    out = with_openrouter_provider_pin(
        OPUS, None, "airline_12", POOL, allow_fallbacks=True
    )
    assert out["provider"]["allow_fallbacks"] is True


def test_provider_choice_is_stable_across_processes():
    """The assignment must survive a fresh interpreter, a different host and a
    later run, so it is hard-coded here rather than recomputed. Python's
    salted hash() would fail this; sha256 does not."""
    expected = {
        "1": "google-vertex",
        "5": "google-vertex",
        "6": "google-vertex",
        "airline_01": "amazon-bedrock",
        "airline_02": "amazon-bedrock",
        "[retail] 0042": "amazon-bedrock",
    }
    assert {k: openrouter_provider_for_key(k, POOL) for k in expected} == expected


def test_provider_choice_really_is_rendezvous_sha256():
    """Guards the hard-coded table above against being 'fixed' to match a
    changed implementation: recompute it independently."""
    import hashlib

    def score(key, provider):
        digest = hashlib.sha256(f"{key}\x00{provider}".encode()).digest()
        return int.from_bytes(digest[:8], "big")

    for key in ("1", "airline_01", "[retail] 0042"):
        assert openrouter_provider_for_key(key, POOL) == max(
            POOL, key=lambda p: (score(key, p), p)
        )


def test_provider_choice_ignores_pool_order_and_duplicates():
    """Two cells whose shell configs list the same providers in a different
    order must still agree, or routing re-enters the locale axis."""
    for pool in ([*POOL], list(reversed(POOL)), POOL + POOL, [" azure ", *POOL]):
        assert openrouter_provider_for_key("airline_02", pool) == "amazon-bedrock"


def test_dropping_a_provider_only_moves_that_provider_s_tasks():
    """After an outage the operator re-runs with the dead endpoint removed.
    Rendezvous hashing keeps every other task where it was, so the repair pass
    is routed like the cell it repairs; plain `hash % n` would re-shuffle all
    of them."""
    task_ids = [f"[airline] task_{i:03d}" for i in range(114)]
    before = {t: openrouter_provider_for_key(t, POOL) for t in task_ids}
    reduced = [p for p in POOL if p != "google-vertex"]
    after = {t: openrouter_provider_for_key(t, reduced) for t in task_ids}
    moved = {t for t in task_ids if before[t] != after[t]}
    assert moved == {t for t in task_ids if before[t] == "google-vertex"}
    assert moved  # the dropped provider did hold tasks
    assert set(after.values()) == set(reduced)


def test_provider_pin_is_identical_across_locale_domain_and_agent():
    """Same task id -> same provider, whatever else the cell varies."""
    cells = [
        (OPUS, {"temperature": None}),
        (OPUS, {"reasoning": {"enabled": False}}),
        ("openrouter/openai/gpt-5.6-sol:exacto", {}),
    ]
    chosen = {
        with_openrouter_provider_pin(model, args, "airline_12", POOL)["provider"][
            "order"
        ][0]
        for model, args in cells
    }
    assert len(chosen) == 1


def test_provider_pin_spreads_load_over_the_pool():
    """A realistic task set must not pile onto one endpoint: every provider
    gets a share within a factor of ~1.5 of even."""
    task_ids = [f"[airline] task_{i:03d}" for i in range(114)]
    counts = Counter(openrouter_provider_for_key(t, POOL) for t in task_ids)
    assert set(counts) == set(POOL)
    even = len(task_ids) / len(POOL)
    assert all(0.5 * even <= n <= 1.5 * even for n in counts.values())


@pytest.mark.parametrize(
    "model", ["claude-opus-5", "gpt-5.6", "vertex_ai/gemini-3-pro"]
)
def test_provider_pin_skips_other_providers(model):
    assert with_openrouter_provider_pin(model, {"temperature": 0.5}, "t1", POOL) == {
        "temperature": 0.5
    }
    assert with_openrouter_provider_pin(model, None, "t1", POOL) == {}


def test_provider_pin_respects_caller_supplied_provider_block():
    """How Nemotron is pinned to its single endpoint today (agents.sh)."""
    mine = {"order": ["baseten"], "allow_fallbacks": False}
    out = with_openrouter_provider_pin(
        "openrouter/nvidia/nemotron-3-ultra-550b-a55b", {"provider": mine}, "t1", POOL
    )
    assert out["provider"] == mine


@pytest.mark.parametrize("providers", [None, [], ["  "]])
def test_provider_pin_without_a_pool_is_a_no_op(providers):
    assert with_openrouter_provider_pin(
        OPUS, {"temperature": 0.5}, "t1", providers
    ) == {"temperature": 0.5}


def test_provider_pin_kill_switch(monkeypatch):
    monkeypatch.setenv("TAU2_DISABLE_OPENROUTER_PROVIDER_PIN", "1")
    assert with_openrouter_provider_pin(OPUS, {"temperature": 0.5}, "t1", POOL) == {
        "temperature": 0.5
    }
    assert openrouter_provider_pin_disabled() is True


def test_provider_pin_and_session_id_compose():
    """Both are applied to the same dict and neither clobbers the other: the
    pin decides the provider, session_id only groups the calls in OpenRouter's
    Sessions view."""
    args = with_openrouter_session_id(OPUS, {"temperature": None}, "sim-7")
    args = with_openrouter_provider_pin(OPUS, args, "airline_12", POOL)
    assert args["session_id"] == "sim-7"
    assert args["provider"]["order"] == ["amazon-bedrock"]
    assert args["temperature"] is None

    # Order of application must not matter.
    other = with_openrouter_provider_pin(
        OPUS, {"temperature": None}, "airline_12", POOL
    )
    other = with_openrouter_session_id(OPUS, other, "sim-7")
    assert other == args


def test_provider_pin_kill_switch_leaves_session_id_alone(monkeypatch):
    monkeypatch.setenv("TAU2_DISABLE_OPENROUTER_PROVIDER_PIN", "1")
    args = with_openrouter_session_id(OPUS, {}, "sim-7")
    assert with_openrouter_provider_pin(OPUS, args, "t1", POOL) == {
        "session_id": "sim-7"
    }


def test_provider_assignments_record_matches_what_is_sent():
    task_ids = ["airline_01", "airline_02", "airline_03"]
    assignments = openrouter_provider_assignments(task_ids, POOL)
    assert set(assignments) == set(task_ids)
    for task_id, provider in assignments.items():
        sent = with_openrouter_provider_pin(OPUS, None, task_id, POOL)
        assert sent["provider"]["order"] == [provider]


@pytest.mark.parametrize(
    "spec,expected",
    [
        (None, None),
        ("", None),
        ("  ,  ", None),
        ("azure", ["azure"]),
        (" azure , amazon-bedrock ", ["azure", "amazon-bedrock"]),
    ],
)
def test_parse_openrouter_providers(spec, expected):
    assert parse_openrouter_providers(spec) == expected


# ---------------------------------------------------------------------------
# Rate-limit wait: a 429 parks the call until the window resets, keeps the turn
# ---------------------------------------------------------------------------


def _rate_limit_error(reset_header=None):
    """litellm.RateLimitError as raised for an OpenRouter 429, optionally with the
    X-RateLimit-Reset response header (litellm copies response headers onto
    ``exc.response``)."""
    response = None
    if reset_header is not None:
        response = _httpx.Response(
            429,
            headers={"X-RateLimit-Reset": str(reset_header)},
            request=_httpx.Request(
                "POST", "https://openrouter.ai/api/v1/chat/completions"
            ),
        )
    return _litellm.RateLimitError(
        message="OpenrouterException - rate limited",
        llm_provider="openrouter",
        model="openrouter/google/gemini-3.8-flash",
        response=response,
    )


def test_rate_limit_wait_uses_provider_reset_header_in_ms():
    """Reset 11 s in the future (OpenRouter sends epoch ms) -> wait 11-13 s incl. jitter."""
    exc = _rate_limit_error(reset_header=1790085720000)
    wait = _llm_utils.rate_limit_wait_seconds(exc, attempt=0, now=1790085709.0)
    assert 11.0 <= wait <= 13.0


def test_rate_limit_wait_accepts_reset_header_in_seconds():
    exc = _rate_limit_error(reset_header=1790085720)
    wait = _llm_utils.rate_limit_wait_seconds(exc, attempt=0, now=1790085709.0)
    assert 11.0 <= wait <= 13.0


def test_rate_limit_wait_reset_in_the_past_still_waits_a_moment():
    exc = _rate_limit_error(reset_header=1790085720000)
    wait = _llm_utils.rate_limit_wait_seconds(exc, attempt=0, now=1790085730.0)
    assert 1.0 <= wait <= 3.0


def test_rate_limit_wait_ignores_unparseable_reset_header():
    exc = _rate_limit_error(reset_header="soon")
    wait = _llm_utils.rate_limit_wait_seconds(exc, attempt=0, now=0.0)
    assert 5.0 <= wait <= 7.0  # falls back to the first backoff step


def test_rate_limit_wait_backs_off_exponentially_without_a_header():
    exc = _rate_limit_error()
    for attempt, base in [
        (0, 5.0),
        (1, 10.0),
        (2, 20.0),
        (3, 40.0),
        (4, 60.0),
        (9, 60.0),
    ]:
        wait = _llm_utils.rate_limit_wait_seconds(exc, attempt=attempt, now=0.0)
        assert base <= wait <= base + 2.0, (attempt, wait)


def test_completion_waits_out_429_and_keeps_the_call(monkeypatch):
    calls = []
    sleeps = []
    sentinel = object()

    def fake_completion(**kwargs):
        calls.append(kwargs)
        if len(calls) <= 2:
            raise _rate_limit_error()
        return sentinel

    monkeypatch.setattr(_llm_utils, "completion", fake_completion)
    monkeypatch.setattr(_llm_utils.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(_llm_utils.random, "uniform", lambda a, b: 0.0)

    out = _llm_utils._completion_with_rate_limit_wait(model="m", messages=[])

    assert out is sentinel
    assert len(calls) == 3
    assert sleeps == [5.0, 10.0]  # exponential backoff, no header, no jitter
    assert all(c == {"model": "m", "messages": []} for c in calls)  # kwargs untouched


def test_completion_gives_up_after_max_cumulative_wait(monkeypatch):
    calls = []
    sleeps = []

    def always_429(**kwargs):
        calls.append(kwargs)
        raise _rate_limit_error()

    monkeypatch.setattr(_llm_utils, "completion", always_429)
    monkeypatch.setattr(_llm_utils.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(_llm_utils.random, "uniform", lambda a, b: 0.0)
    monkeypatch.setenv(_llm_utils.RATE_LIMIT_MAX_WAIT_ENV, "12")

    with pytest.raises(_litellm.RateLimitError):
        _llm_utils._completion_with_rate_limit_wait(model="m", messages=[])

    # 5 s fits under 12, 5+10 would not: one wait, then the error propagates.
    assert sleeps == [5.0]
    assert len(calls) == 2


def test_completion_max_wait_zero_disables_waiting(monkeypatch):
    monkeypatch.setattr(
        _llm_utils,
        "completion",
        lambda **kw: (_ for _ in ()).throw(_rate_limit_error()),
    )
    monkeypatch.setattr(
        _llm_utils.time, "sleep", lambda s: pytest.fail("must not sleep")
    )
    monkeypatch.setenv(_llm_utils.RATE_LIMIT_MAX_WAIT_ENV, "0")
    with pytest.raises(_litellm.RateLimitError):
        _llm_utils._completion_with_rate_limit_wait(model="m", messages=[])


def test_completion_other_errors_are_not_retried_here(monkeypatch):
    calls = []

    def bad_request(**kwargs):
        calls.append(1)
        raise _litellm.BadRequestError(
            message="nope", llm_provider="openrouter", model="m"
        )

    monkeypatch.setattr(_llm_utils, "completion", bad_request)
    monkeypatch.setattr(
        _llm_utils.time, "sleep", lambda s: pytest.fail("must not sleep")
    )
    with pytest.raises(_litellm.BadRequestError):
        _llm_utils._completion_with_rate_limit_wait(model="m", messages=[])
    assert calls == [1]


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, 600.0),
        ("", 600.0),
        ("90", 90.0),
        ("0", 0.0),
        ("-3", 600.0),
        ("abc", 600.0),
    ],
)
def test_rate_limit_max_wait_env(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(_llm_utils.RATE_LIMIT_MAX_WAIT_ENV, raising=False)
    else:
        monkeypatch.setenv(_llm_utils.RATE_LIMIT_MAX_WAIT_ENV, raw)
    assert _llm_utils.rate_limit_max_wait_seconds() == expected
