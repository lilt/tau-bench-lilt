"""
Tests for the per-turn retry of failed user-simulator turns.

The user simulator's provider call intermittently fails (litellm reports a malformed
provider response as APIConnectionError), and it can also return a message with neither
content nor tool calls. Both are retried for that turn only, keeping the conversation
intact; once the retries are exhausted the simulation ends with USER_ERROR, which is
scored 0.0 rather than excluded. Non-transient errors must still reach the
whole-simulation retry in `tau2.runner.progress.run_with_retry`.
"""

from typing import Callable, Optional

import pytest
from litellm.exceptions import APIConnectionError, AuthenticationError, RateLimitError

from tau2.agent.base_agent import HalfDuplexAgent
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    UserMessage,
)
from tau2.data_model.simulation import TerminationReason
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.orchestrator.orchestrator import (
    Orchestrator,
    Role,
)
from tau2.runner.progress import run_with_retry
from tau2.runner.simulation import run_simulation
from tau2.user.user_simulator import UserSimulator
from tau2.user.user_simulator_base import HalfDuplexUser


class ScriptedState:
    """Minimal participant state exposing a `messages` list, like UserState."""

    def __init__(self, messages: Optional[list[Message]] = None):
        self.messages: list[Message] = list(messages or [])


def invalid_user_message() -> UserMessage:
    """A user message with neither content nor tool calls."""
    return UserMessage(role="user", content=None, tool_calls=None, cost=0.0)


def transient_error() -> APIConnectionError:
    """The failure seen in production: litellm wrapping a malformed response."""
    return APIConnectionError(
        message="OpenrouterException - Invalid response object",
        llm_provider="openrouter",
        model="google/gemini-3.5-flash",
    )


class ScriptedUser(HalfDuplexUser[ScriptedState]):
    """User simulator that replays a scripted list of outcomes.

    An outcome may be a UserMessage (returned) or an Exception (raised). It mirrors
    UserSimulator: the incoming message is appended to the state BEFORE the provider
    call, so a raised exception leaves the state dirty.
    """

    def __init__(self, outcomes: list):
        super().__init__(instructions="scripted user", tools=None)
        self.outcomes = list(outcomes)
        self.call_count = 0

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> ScriptedState:
        return ScriptedState(message_history)

    def generate_next_message(self, message, state: ScriptedState):
        self.call_count += 1
        state.messages.append(message)  # appended before the "provider call"
        outcome = (
            self.outcomes.pop(0)
            if self.outcomes
            else UserMessage(role="user", content="###STOP###", cost=0.0)
        )
        if isinstance(outcome, Exception):
            raise outcome
        state.messages.append(outcome)
        return outcome, state


class ScriptedAgent(HalfDuplexAgent[ScriptedState]):
    """Agent that always replies with the same message."""

    def __init__(self, content: str = "How can I help?"):
        super().__init__(tools=[], domain_policy="")
        self.content = content
        self.call_count = 0

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> ScriptedState:
        return ScriptedState(message_history)

    def generate_next_message(self, message, state: ScriptedState):
        self.call_count += 1
        state.messages.append(message)
        reply = AssistantMessage(role="assistant", content=self.content, cost=0.0)
        state.messages.append(reply)
        return reply, state


def make_orchestrator(
    domain_name: str,
    environment: Environment,
    task: Task,
    agent: ScriptedAgent,
    user: ScriptedUser,
    **kwargs,
) -> Orchestrator:
    return Orchestrator(
        domain=domain_name,
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        **kwargs,
    )


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(transient_error, id="transient_provider_error"),
        pytest.param(invalid_user_message, id="unusable_user_message"),
        pytest.param(
            lambda: RateLimitError(
                message="rate limited", llm_provider="openrouter", model="m"
            ),
            id="rate_limit",
        ),
    ],
)
@pytest.mark.parametrize("num_failures", [1, 2, 3])
def test_user_turn_retried_in_place_preserves_conversation(
    domain_name: str,
    get_environment: Callable[[], Environment],
    base_task: Task,
    failure,
    num_failures: int,
):
    """A user turn that fails then succeeds is retried in place, not restarted.

    Asserts the exact resulting message sequence -- both the orchestrator trajectory
    and the user simulator's own history -- so an incoming agent message that is
    dropped entirely, or replayed once per failed attempt, would fail this test.
    """
    good_msg = UserMessage(role="user", content="Hi, I need help.", cost=0.0)
    user = ScriptedUser([failure() for _ in range(num_failures)] + [good_msg])
    agent = ScriptedAgent()
    orchestrator = make_orchestrator(
        domain_name, get_environment(), base_task, agent, user
    )
    orchestrator.initialize()

    assert orchestrator.to_role == Role.USER
    trajectory_before = list(orchestrator.trajectory)
    greeting = orchestrator.message
    assert isinstance(greeting, AssistantMessage)

    orchestrator.step()  # agent greeting -> user (fails num_failures times, succeeds)

    # The turn was retried, the simulation was not restarted.
    assert user.call_count == num_failures + 1
    assert agent.call_count == 0
    assert not orchestrator.done
    assert orchestrator.termination_reason is None

    # Exact trajectory: the greeting and the one good user message.
    assert orchestrator.trajectory[: len(trajectory_before)] == trajectory_before
    assert len(orchestrator.trajectory) == 2
    assert orchestrator.trajectory[0] is greeting
    assert orchestrator.trajectory[1] is good_msg

    # Exact user history: each failed attempt appended the incoming agent message
    # (before its provider call) and possibly a bad reply; all of that was rolled
    # back, so the greeting appears EXACTLY ONCE, in position.
    user_messages = orchestrator.user_state.messages
    assert len(user_messages) == 2
    assert user_messages[0] is greeting
    assert user_messages[1] is good_msg
    assert sum(1 for m in user_messages if m is greeting) == 1
    assert all(m.has_content() for m in user_messages)


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(transient_error, id="transient_provider_error"),
        pytest.param(invalid_user_message, id="unusable_user_message"),
    ],
)
def test_user_retries_exhausted_ends_simulation_with_scored_result(
    domain_name: str,
    get_environment: Callable[[], Environment],
    base_task: Task,
    failure,
):
    """4 failures (1 attempt + 3 retries) end the simulation with a scored result."""
    user = ScriptedUser([failure() for _ in range(4)])
    agent = ScriptedAgent()
    orchestrator = make_orchestrator(
        domain_name, get_environment(), base_task, agent, user
    )

    simulation = run_simulation(orchestrator)

    # One attempt plus the 3 configured retries, all within the same simulation.
    assert user.call_count == 4
    assert simulation.termination_reason == TerminationReason.USER_ERROR

    # Scored, not null: reward_info populated, and NOT an infrastructure error
    # (only INFRASTRUCTURE_ERROR is dropped from agent_metrics / re-run by checkpoint).
    assert simulation.reward_info is not None
    assert simulation.reward_info.reward == 0.0
    assert simulation.termination_reason != TerminationReason.INFRASTRUCTURE_ERROR
    assert len(simulation.messages) == 1  # the greeting is kept
    assert simulation.duration > 0.0


def test_user_retry_count_is_configurable(
    domain_name: str,
    get_environment: Callable[[], Environment],
    base_task: Task,
):
    """max_user_message_retries controls how many times a turn is re-requested."""
    user = ScriptedUser([transient_error() for _ in range(5)])
    orchestrator = make_orchestrator(
        domain_name,
        get_environment(),
        base_task,
        ScriptedAgent(),
        user,
        max_user_message_retries=1,
    )
    orchestrator.initialize()
    orchestrator.step()

    assert user.call_count == 2
    assert orchestrator.done
    assert orchestrator.termination_reason == TerminationReason.USER_ERROR


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(
            AuthenticationError(
                message="invalid api key", llm_provider="openrouter", model="m"
            ),
            id="auth_error",
        ),
        pytest.param(RuntimeError("something else entirely"), id="runtime_error"),
    ],
)
def test_non_transient_user_error_still_propagates(
    domain_name: str,
    get_environment: Callable[[], Environment],
    base_task: Task,
    error,
):
    """A non-transient error is not swallowed by the turn retry."""
    user = ScriptedUser([error, error, error, error])
    orchestrator = make_orchestrator(
        domain_name, get_environment(), base_task, ScriptedAgent(), user
    )

    with pytest.raises(type(error)):
        orchestrator.run()

    # Failed fast: a single user call, no burning of 4 attempts.
    assert user.call_count == 1


def test_non_transient_user_error_reaches_whole_simulation_retry(
    domain_name: str,
    get_environment: Callable[[], Environment],
    base_task: Task,
):
    """The outer whole-simulation safety net still handles non-transient errors."""
    users: list[ScriptedUser] = []

    def run_fn():
        user = ScriptedUser(
            [
                AuthenticationError(
                    message="invalid api key", llm_provider="openrouter", model="m"
                )
            ]
        )
        users.append(user)
        orchestrator = make_orchestrator(
            domain_name, get_environment(), base_task, ScriptedAgent(), user
        )
        return orchestrator.run()

    simulation = run_with_retry(
        run_fn,
        task=base_task,
        trial=0,
        seed=0,
        max_retries=2,
        retry_delay=0.0,
        console_display=False,
    )

    assert len(users) == 3  # whole simulation restarted, as before
    assert all(u.call_count == 1 for u in users)
    assert simulation.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR


def test_real_user_simulator_state_is_rolled_back(
    domain_name: str,
    get_environment: Callable[[], Environment],
    base_task: Task,
    monkeypatch,
):
    """Same guarantee, against the real UserSimulator rather than a scripted stand-in.

    UserSimulator.generate_next_message appends the incoming message to its state
    BEFORE calling its provider, so a raised exception leaves the state dirty. This
    exercises that real code path (with only the provider call stubbed out) to confirm
    the rollback keeps the incoming message at exactly one occurrence.
    """
    from tau2.user import user_simulator as user_simulator_module

    calls = {"n": 0}

    def fake_generate(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise transient_error()
        return AssistantMessage(role="assistant", content="Hi, I need help.", cost=0.0)

    monkeypatch.setattr(user_simulator_module, "generate", fake_generate)

    user = UserSimulator(
        instructions="You are a user simulator.",
        llm="gpt-4.1",
        llm_args={"temperature": 0.0},
    )
    orchestrator = Orchestrator(
        domain=domain_name,
        agent=ScriptedAgent(),
        user=user,
        environment=get_environment(),
        task=base_task,
    )
    orchestrator.initialize()
    greeting = orchestrator.message

    orchestrator.step()  # agent greeting -> user: raises once, then succeeds

    assert calls["n"] == 2
    assert not orchestrator.done

    # The real UserSimulator appended the greeting before the failed provider call.
    # After rollback + successful retry it must appear exactly once.
    user_messages = orchestrator.user_state.messages
    assert sum(1 for m in user_messages if m is greeting) == 1
    assert len(user_messages) == 2
    assert user_messages[0] is greeting
    assert user_messages[1].content == "Hi, I need help."
    assert orchestrator.trajectory[-1].content == "Hi, I need help."


def test_transient_error_no_longer_restarts_whole_simulation(
    domain_name: str,
    get_environment: Callable[[], Environment],
    base_task: Task,
):
    """The motivating regression: a transient error must not restart the simulation."""
    orchestrators: list[Orchestrator] = []

    def run_fn():
        user = ScriptedUser(
            [
                transient_error(),
                UserMessage(role="user", content="Hi, I need help.", cost=0.0),
                UserMessage(role="user", content="Thanks, ###STOP###", cost=0.0),
            ]
        )
        orchestrator = make_orchestrator(
            domain_name, get_environment(), base_task, ScriptedAgent(), user
        )
        orchestrators.append(orchestrator)
        return orchestrator.run()

    simulation = run_with_retry(
        run_fn,
        task=base_task,
        trial=0,
        seed=0,
        max_retries=2,
        retry_delay=0.0,
        console_display=False,
    )

    # Exactly one simulation was built: the outer retry never fired.
    assert len(orchestrators) == 1
    assert simulation.termination_reason == TerminationReason.USER_STOP
    assert simulation.termination_reason != TerminationReason.INFRASTRUCTURE_ERROR
