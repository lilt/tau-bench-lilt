"""The per-task OpenRouter pin, as the runner actually applies it.

test_llm_utils covers the hash; what is under test here is the wiring: the
shard key is the *task* id (not the simulation id), it reaches both the agent
and the user simulator, it survives alongside session_id, and the mapping
written into ``info`` is the one the orchestrators will send.
"""

import pytest

from tau2.data_model.simulation import TextRunConfig
from tau2.run import get_tasks
from tau2.runner.build import build_text_orchestrator
from tau2.runner.helpers import build_provider_pin_info

POOL = ["amazon-bedrock", "azure", "google-vertex"]
AGENT_MODEL = "openrouter/anthropic/claude-opus-5:exacto"
USER_MODEL = "openrouter/google/gemini-3.5-flash:exacto"


def _config(**overrides) -> TextRunConfig:
    defaults = dict(
        domain="mock",
        agent="llm_agent",
        user="user_simulator",
        llm_agent=AGENT_MODEL,
        llm_args_agent={},
        llm_user=USER_MODEL,
        llm_args_user={},
        num_trials=4,
        max_steps=20,
        max_errors=10,
        openrouter_providers_agent=POOL,
        openrouter_providers_user=POOL,
    )
    defaults.update(overrides)
    return TextRunConfig(**defaults)


def _tasks():
    return get_tasks("mock")


def _built(config, task, simulation_id="sim-1"):
    orchestrator = build_text_orchestrator(
        config, task, simulation_id=simulation_id, seed=42
    )
    return orchestrator.agent.llm_args, orchestrator.user.llm_args


def test_agent_and_user_are_pinned_per_task():
    config = _config()
    for task in _tasks():
        agent_args, user_args = _built(config, task)
        for args in (agent_args, user_args):
            assert len(args["provider"]["order"]) == 1
            assert args["provider"]["order"][0] in POOL
            assert args["provider"]["allow_fallbacks"] is False


def test_pin_is_keyed_on_the_task_not_the_simulation():
    """Every trial of a task must share a provider, and two tasks must not be
    forced onto the same one — so the key is the task id, and a different
    simulation id cannot change the answer."""
    config = _config()
    tasks = _tasks()
    per_task = {}
    for task in tasks:
        providers = {
            _built(config, task, simulation_id=f"sim-{trial}")[0]["provider"]["order"][
                0
            ]
            for trial in range(config.num_trials)
        }
        assert len(providers) == 1, f"{task.id} flipped provider between trials"
        per_task[task.id] = providers.pop()
    assert len(set(per_task.values())) > 1, (
        "every task landed on one provider; the pool is not being spread"
    )


def test_pin_does_not_change_with_locale_or_agent_config():
    """The routing axis must stay orthogonal to the locale axis."""
    tasks = _tasks()
    baseline = {
        task.id: _built(_config(), task)[0]["provider"]["order"][0] for task in tasks
    }
    for variant in (
        _config(language="de-DE"),
        _config(language="ko-KR"),
        _config(llm_args_agent={"temperature": None}),
        _config(llm_args_agent={"reasoning": {"enabled": False}}),
        _config(llm_user="openrouter/openai/gpt-5.6-sol:exacto"),
    ):
        for task in tasks:
            assert _built(variant, task)[0]["provider"]["order"][0] == baseline[task.id]


def test_session_id_and_pin_coexist():
    config = _config()
    task = _tasks()[0]
    agent_args, user_args = _built(config, task, simulation_id="sim-xyz")
    assert agent_args["session_id"] == "sim-xyz"
    assert user_args["session_id"] == "sim-xyz"
    assert agent_args["provider"]["order"]


def test_no_pool_configured_leaves_llm_args_untouched():
    config = _config(openrouter_providers_agent=None, openrouter_providers_user=None)
    agent_args, user_args = _built(config, _tasks()[0])
    assert "provider" not in agent_args
    assert "provider" not in user_args
    assert build_provider_pin_info(config, _tasks()) is None


def test_non_openrouter_models_are_untouched():
    config = _config(llm_agent="gpt-4o-mini", llm_user="gpt-4o-mini")
    agent_args, user_args = _built(config, _tasks()[0])
    assert "provider" not in agent_args
    assert "provider" not in user_args
    assert build_provider_pin_info(config, _tasks()) is None


def test_caller_supplied_provider_block_wins():
    """Nemotron-style manual pin (agents.sh) must not be overwritten."""
    mine = {"order": ["baseten"], "allow_fallbacks": False}
    config = _config(llm_args_agent={"provider": mine})
    agent_args, _ = _built(config, _tasks()[0])
    assert agent_args["provider"] == mine


def test_kill_switch_disables_pin_and_its_record(monkeypatch):
    monkeypatch.setenv("TAU2_DISABLE_OPENROUTER_PROVIDER_PIN", "1")
    config = _config()
    agent_args, user_args = _built(config, _tasks()[0])
    assert "provider" not in agent_args
    assert "provider" not in user_args
    assert build_provider_pin_info(config, _tasks()) is None


def test_info_record_matches_what_the_orchestrators_send():
    config = _config()
    tasks = _tasks()
    pin = build_provider_pin_info(config, tasks)
    assert pin is not None
    assert pin.shard_key == "task_id"
    assert pin.allow_fallbacks is False
    assert pin.agent_providers == sorted(POOL)
    assert pin.user_providers == sorted(POOL)
    for task in tasks:
        agent_args, user_args = _built(config, task)
        assert pin.agent_assignments[task.id] == agent_args["provider"]["order"][0]
        assert pin.user_assignments[task.id] == user_args["provider"]["order"][0]


def test_allow_fallbacks_flows_through():
    config = _config(openrouter_allow_fallbacks=True)
    agent_args, _ = _built(config, _tasks()[0])
    assert agent_args["provider"]["allow_fallbacks"] is True
    assert build_provider_pin_info(config, _tasks()).allow_fallbacks is True


@pytest.mark.parametrize("pool", [["azure"], POOL, POOL + ["anthropic"]])
def test_pool_size_one_is_a_plain_pin(pool):
    config = _config(openrouter_providers_agent=pool)
    providers = {_built(config, task)[0]["provider"]["order"][0] for task in _tasks()}
    assert providers <= set(pool)
    if len(pool) == 1:
        assert providers == set(pool)
