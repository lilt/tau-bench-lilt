"""The data locale must reach every process that runs simulations.

The CLI sets it only in its own process; --workers subprocesses and
programmatic run_domain/run_tasks callers take it from the run config.
"""

import pytest

import tau2.domain_data
from tau2.data_model.simulation import TextRunConfig
from tau2.domain_data import get_data_lang
from tau2.evaluator.evaluator import EvaluationType
from tau2.run import get_tasks
from tau2.runner import batch as batch_mod
from tau2.runner import worker as worker_mod
from tau2.runner.batch import activate_data_lang
from tau2.runner.work import WorkUnit


@pytest.fixture(autouse=True)
def reset_data_lang(monkeypatch):
    monkeypatch.setattr(tau2.domain_data, "_data_lang", None)


def _make_config(**overrides) -> TextRunConfig:
    defaults = dict(
        domain="mock",
        agent="llm_agent",
        user="user_simulator",
        llm_agent="gpt-3.5-turbo",
        llm_args_agent={},
        llm_user="gpt-3.5-turbo",
        llm_args_user={},
        num_trials=1,
        max_steps=20,
        max_errors=10,
    )
    defaults.update(overrides)
    return TextRunConfig(**defaults)


def test_activate_data_lang_sets_locale():
    activate_data_lang(_make_config(data_lang="de-DE"))
    assert get_data_lang() == "de-DE"


def test_activate_data_lang_without_locale_leaves_it():
    tau2.domain_data.set_data_lang("ko-KR")
    activate_data_lang(_make_config())
    assert get_data_lang() == "ko-KR"


def test_worker_runs_unit_with_config_locale(monkeypatch):
    seen = []
    monkeypatch.setattr(
        batch_mod, "run_unit", lambda ctx, *args: seen.append(get_data_lang())
    )
    config = _make_config(data_lang="de-DE")
    task = get_tasks("mock")[0]
    payload = {
        "unit": WorkUnit(
            unit_id="u", run_id="r", task_id=task.id, trial=0, seed=1
        ).model_dump(),
        "run": {
            "config_kind": "text",
            "config": config.model_dump(mode="json"),
            "task": task.model_dump(mode="json"),
            "save_dir": None,
            "evaluation_type": EvaluationType.ALL.value,
        },
    }
    worker_mod.execute_lease(payload)
    assert seen == ["de-DE"]
