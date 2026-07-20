"""Tests for /goal handling in tui_gateway.

The TUI routes ``/goal`` through ``command.dispatch`` (not ``slash.exec``)
because the CLI's ``_handle_goal_command`` queues the kickoff message onto
``_pending_input``, which the slash-worker subprocess has no reader for.
Instead we handle ``/goal`` directly in the server and return a
``{"type": "send", "notice": ..., "message": ...}`` payload the TUI client
uses to render a system line and fire the kickoff prompt.
"""

from __future__ import annotations

import importlib
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    # Bust the goal-module DB cache so it re-resolves HERMES_HOME.
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


@pytest.fixture()
def server(hermes_home):
    with patch.dict(
        "sys.modules",
        {
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
        },
    ):
        mod = importlib.import_module("tui_gateway.server")
        yield mod
        # Reset module-level session state without re-importing. importlib.reload
        # would re-register the module's atexit hooks (ThreadPoolExecutor
        # shutdown, _shutdown_sessions); the duplicates race the stderr
        # buffer at interpreter shutdown and surface as Fatal Python error:
        # _enter_buffered_busy. Clearing the per-session dicts gives the
        # next test a clean slate; _methods is NOT cleared because it's
        # populated at module import time and re-registration only happens
        # via reload (which we don't do).
        mod._sessions.clear()
        mod._pending.clear()
        mod._answers.clear()


@pytest.fixture()
def session(server, request):
    sid = "sid-test"
    session_key = f"tui-goal-{request.node.name}"
    s = {
        "session_key": session_key,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "cols": 120,
    }
    server._sessions[sid] = s
    return sid, session_key, s


def _call(server, method, **params):
    handler = server._methods[method]
    return handler(1, params)


# ── command.dispatch /goal ────────────────────────────────────────────


def test_goal_bare_shows_status_when_none_set(server, session):
    sid, _, _ = session
    r = _call(server, "command.dispatch", name="goal", arg="", session_id=sid)
    assert r["result"]["type"] == "exec"
    assert "No active goal" in r["result"]["output"]


def test_goal_whitespace_only_shows_status(server, session):
    sid, _, _ = session
    r = _call(server, "command.dispatch", name="goal", arg="   ", session_id=sid)
    assert r["result"]["type"] == "exec"
    assert "No active goal" in r["result"]["output"]


def test_goal_status_alias_shows_status(server, session):
    sid, _, _ = session
    r = _call(server, "command.dispatch", name="goal", arg="status", session_id=sid)
    assert r["result"]["type"] == "exec"
    assert "No active goal" in r["result"]["output"]


def test_goal_set_returns_send_with_notice(server, session):
    sid, session_key, _ = session
    r = _call(server, "command.dispatch", name="goal", arg="build a rocket", session_id=sid)
    result = r["result"]
    assert result["type"] == "send"
    assert result["message"] == "build a rocket"
    assert "notice" in result
    assert "Goal set" in result["notice"]
    assert "20-turn budget" in result["notice"]

    # Persisted in SessionDB
    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_key)
    assert mgr.state is not None
    assert mgr.state.goal == "build a rocket"
    assert mgr.state.status == "active"


def test_goal_pause_after_set(server, session):
    sid, session_key, _ = session
    _call(server, "command.dispatch", name="goal", arg="write a story", session_id=sid)
    r = _call(server, "command.dispatch", name="goal", arg="pause", session_id=sid)
    assert r["result"]["type"] == "exec"
    assert "paused" in r["result"]["output"].lower()

    from hermes_cli.goals import GoalManager

    assert GoalManager(session_key).state.status == "paused"


def test_goal_resume_reactivates(server, session):
    sid, session_key, _ = session
    _call(server, "command.dispatch", name="goal", arg="write a story", session_id=sid)
    _call(server, "command.dispatch", name="goal", arg="pause", session_id=sid)
    r = _call(server, "command.dispatch", name="goal", arg="resume", session_id=sid)
    assert r["result"]["type"] == "exec"
    assert "resumed" in r["result"]["output"].lower()

    from hermes_cli.goals import GoalManager

    assert GoalManager(session_key).state.status == "active"


def test_goal_clear_removes_active_goal(server, session):
    sid, session_key, _ = session
    _call(server, "command.dispatch", name="goal", arg="write a story", session_id=sid)
    r = _call(server, "command.dispatch", name="goal", arg="clear", session_id=sid)
    assert r["result"]["type"] == "exec"
    assert "cleared" in r["result"]["output"].lower()

    from hermes_cli.goals import GoalManager

    # After clear the row is marked status=cleared (kept for audit);
    # ``has_goal()`` / ``is_active()`` return False so the goal loop
    # stays off and ``status`` reports "No active goal".
    mgr = GoalManager(session_key)
    assert not mgr.has_goal()
    assert not mgr.is_active()
    assert "No active goal" in mgr.status_line()


def test_goal_stop_and_done_are_clear_aliases(server, session):
    sid, _, _ = session
    _call(server, "command.dispatch", name="goal", arg="first goal", session_id=sid)
    r = _call(server, "command.dispatch", name="goal", arg="stop", session_id=sid)
    assert "cleared" in r["result"]["output"].lower()

    _call(server, "command.dispatch", name="goal", arg="second goal", session_id=sid)
    r = _call(server, "command.dispatch", name="goal", arg="done", session_id=sid)
    assert "cleared" in r["result"]["output"].lower()


def test_goal_requires_session(server):
    r = _call(server, "command.dispatch", name="goal", arg="nope", session_id="unknown")
    assert "error" in r
    assert r["error"]["code"] == 4001


# ── slash.exec /goal routing ──────────────────────────────────────────


def test_slash_exec_routes_goal_to_command_dispatch(server, session):
    """slash.exec must route /goal directly to command.dispatch internally
    instead of returning an error.  Previously the 4018 error required the
    TUI client to retry via command.dispatch, but some clients failed the
    fallback, leaving the command empty ("empty command")."""
    sid, _, _ = session
    r = _call(server, "slash.exec", command="goal status", session_id=sid)
    # Should succeed by routing to command.dispatch internally
    assert "result" in r
    assert r["result"]["type"] == "exec"
    assert "No active goal" in r["result"]["output"]


def test_pending_input_commands_includes_goal(server):
    """Guard: _PENDING_INPUT_COMMANDS must list 'goal' — removing it would
    silently re-break the TUI."""
    assert "goal" in server._PENDING_INPUT_COMMANDS


def test_goal_max_iterations_checkpoints_once_without_judging_or_followup(server, session, monkeypatch):
    """A model iteration cap is a durable handoff, not a normal goal verdict.

    Exercise the real prompt thread so the checkpoint event, no-judge rule,
    lack of recursive follow-up, and released reusable session are one
    behavioral contract rather than implementation-detail assertions.
    """
    sid, session_key, live = session
    from hermes_cli import goals
    from hermes_cli.goals import GoalManager

    GoalManager(session_key).set("finish the bounded task", max_turns=1)
    judge = MagicMock(return_value=("continue", "should not run", False, None, False))
    monkeypatch.setattr(goals, "judge_goal", judge)

    class Agent:
        model = "test-model"
        provider = "test"
        base_url = ""
        api_key = ""
        api_mode = ""
        interim_assistant_callback = None

        def clear_interrupt(self):
            pass

        def run_conversation(self, *_args, **_kwargs):
            return {
                "final_response": "",
                "messages": [],
                "turn_exit_reason": "max_iterations_reached(3/3)",
            }

    emitted = []
    monkeypatch.setattr(server, "_emit", lambda event, event_sid, payload=None: emitted.append((event, event_sid, payload)))
    monkeypatch.setattr(server, "_set_session_context", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_session_cwd", lambda *_args, **_kwargs: ".")
    monkeypatch.setattr(server, "make_stream_renderer", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_get_usage", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(server, "_session_info", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(server, "_load_cfg", lambda: {"goals": {"max_turns": 20}})
    monkeypatch.setattr(server, "_load_interim_assistant_messages", lambda: False)
    monkeypatch.setattr(server, "_voice_tts_enabled", lambda: False)

    live.update(agent=Agent(), session_key=session_key, running=True)
    server._run_prompt_submit("checkpoint-rid", sid, live, "start")
    live["_run_thread"].join(timeout=10)

    assert not live["_run_thread"].is_alive()
    assert live["running"] is False
    assert judge.call_count == 0
    assert GoalManager(session_key).state.status == "needs_continuation"
    checkpoints = [event for event in emitted if event[0] == "goal.checkpointed"]
    assert len(checkpoints) == 1
    assert checkpoints[0][2] == {
        "checkpoint_id": GoalManager(session_key).state.checkpoint_id,
        "checkpoint_at": GoalManager(session_key).state.checkpoint_at,
        "goal_session_id": session_key,
        "status": "needs_continuation",
        "reason": "max_iterations_reached",
        "budget_used": 3,
        "budget_max": 3,
        "summary": GoalManager(session_key).state.checkpoint_summary,
    }
    status_updates = [
        event for event in emitted
        if event[0] == "status.update" and event[2].get("kind") == "goal"
    ]
    assert len(status_updates) == 1
    assert "Use /goal resume" in status_updates[0][2]["text"]
    assert not any(event[0] == "message.start" for event in emitted[1:])

    resume = _call(
        server,
        "command.dispatch",
        name="goal",
        arg="resume",
        session_id=sid,
    )
    assert resume["result"]["type"] == "exec"
    assert "resumed" in resume["result"]["output"].lower()
    resumed = GoalManager(session_key).state
    assert resumed.status == "active"
    assert resumed.checkpoint_id is None


def test_non_goal_max_iterations_keeps_session_reusable(server, session, monkeypatch):
    """Without /goal, an iteration cap remains an ordinary completed TUI turn."""
    sid, _, live = session

    class Agent:
        model = "test-model"
        provider = "test"
        base_url = ""
        api_key = ""
        api_mode = ""
        interim_assistant_callback = None

        def clear_interrupt(self):
            pass

        def run_conversation(self, *_args, **_kwargs):
            return {"final_response": "partial", "messages": [], "turn_exit_reason": "max_iterations_reached(1/1)"}

    emitted = []
    monkeypatch.setattr(server, "_emit", lambda event, event_sid, payload=None: emitted.append((event, event_sid, payload)))
    monkeypatch.setattr(server, "_set_session_context", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_session_cwd", lambda *_args, **_kwargs: ".")
    monkeypatch.setattr(server, "make_stream_renderer", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_get_usage", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(server, "_session_info", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(server, "_load_interim_assistant_messages", lambda: False)
    monkeypatch.setattr(server, "_voice_tts_enabled", lambda: False)

    live.update(agent=Agent(), running=True)
    server._run_prompt_submit("non-goal-rid", sid, live, "start")
    live["_run_thread"].join(timeout=10)

    assert not live["_run_thread"].is_alive()
    assert live["running"] is False
    assert not any(event[0] == "goal.checkpointed" for event in emitted)


def test_goal_checkpoint_persistence_failure_is_visible_and_stops_fallthrough(
    server, monkeypatch
):
    class FailingGoalManager:
        def checkpoint_after_max_iterations(self, _reason):
            return {}

    emitted = []
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, event_sid, payload=None: emitted.append(
            (event, event_sid, payload)
        ),
    )

    handled = server._checkpoint_goal_after_max_iterations(
        FailingGoalManager(),
        sid="browser-session",
        goal_session_id="durable-session",
        turn_exit_reason="max_iterations_reached(3/3)",
    )

    assert handled is True
    assert not any(event[0] == "goal.checkpointed" for event in emitted)
    assert emitted == [
        (
            "status.update",
            "browser-session",
            {
                "kind": "goal",
                "text": (
                    "⚠ Goal continuation checkpoint could not be persisted. "
                    "Automatic continuation was stopped; use /goal status after "
                    "storage is available."
                ),
            },
        )
    ]

    emitted.clear()
    assert not server._checkpoint_goal_after_max_iterations(
        FailingGoalManager(),
        sid="browser-session",
        goal_session_id="durable-session",
        turn_exit_reason="max_iterations_reached(3/3) trailing-data",
    )
    assert emitted == []

    for reason in (
        "max_iterations_reached(4/3)",
        "max_iterations_reached(0/0)",
    ):
        assert not server._checkpoint_goal_after_max_iterations(
            FailingGoalManager(),
            sid="browser-session",
            goal_session_id="durable-session",
            turn_exit_reason=reason,
        )
    assert emitted == []


# ── command.dispatch /moa ────────────────────────────────────────────

def _write_moa_config(home, text):
    cfg_path = home / "config.yaml"
    cfg_path.write_text(text)


def test_moa_bare_returns_usage(server, session, hermes_home):
    _write_moa_config(hermes_home, """
moa:
  default_preset: default
  presets:
    default:
      reference_models:
        - provider: openai-codex
          model: gpt-5.5
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
""")
    sid, _, s = session
    r = _call(server, "command.dispatch", name="moa", arg="", session_id=sid)
    # Bare /moa is usage-only now; switching to a preset is via the model picker.
    assert "error" in r
    assert "model_override" not in s


def test_moa_arg_is_always_one_shot(server, session, hermes_home):
    # Any arg (even a preset name) is a one-shot prompt through the DEFAULT
    # preset; /moa never does a sticky switch anymore.
    _write_moa_config(hermes_home, """
moa:
  default_preset: default
  presets:
    default: {}
    review:
      reference_models:
        - provider: openrouter
          model: deepseek/deepseek-v4-pro
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
""")
    sid, _, s = session
    r = _call(server, "command.dispatch", name="moa", arg="review", session_id=sid)
    result = r["result"]
    assert result["type"] == "send"
    assert result["message"] == "review"
    assert "one-shot" in result["notice"]
    # Lazy session (no live agent) → MoA preset pinned via model_override for
    # the build, and it is the DEFAULT preset, not the "review" arg.
    assert s["model_override"]["provider"] == "moa"
    assert s["model_override"]["model"] == "default"


def test_moa_non_preset_returns_one_shot_send(server, session, hermes_home):
    _write_moa_config(hermes_home, """
moa:
  default_preset: default
  presets:
    default:
      reference_models:
        - provider: openai-codex
          model: gpt-5.5
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
""")
    sid, _, _ = session
    r = _call(server, "command.dispatch", name="moa", arg="inspect this project", session_id=sid)
    result = r["result"]
    assert result["type"] == "send"
    assert result["message"] == "inspect this project"
    assert "one-shot" in result["notice"]


def test_pending_input_commands_includes_moa(server):
    assert "moa" in server._PENDING_INPUT_COMMANDS
