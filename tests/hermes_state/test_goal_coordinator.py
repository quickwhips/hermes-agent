"""Durable goal coordinator persistence contracts (#243)."""

from __future__ import annotations

import sqlite3

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    state = SessionDB(db_path=tmp_path / "state.db")
    yield state
    state.close()


def _create(db: SessionDB, *, goal_id: str = "goal-1", session_id: str = "session-a"):
    return db.create_goal_coordinator(
        goal_id=goal_id,
        profile_name="default",
        goal="Ship the durable coordinator",
        acceptance_contract={"criteria": ["focused tests pass"]},
        session_id=session_id,
    )


def _checkpoint():
    return {
        "milestone": {"id": "m1", "title": "Persistence"},
        "repository": {
            "root": "/work/hermes-agent",
            "worktree": "/work/hermes-agent",
            "branch": "feat/durable-goals",
            "head_sha": "a" * 40,
        },
        "files_changed": ["hermes_state.py"],
        "validation": [
            {
                "command": "pytest tests/hermes_state/test_goal_coordinator.py",
                "status": "passed",
                "summary": "7 passed",
            }
        ],
        "child_sessions": [
            {"id": "worker-1", "status": "completed", "summary": "schema reviewed"}
        ],
        "blockers": [],
        "approvals_needed": [],
        "artifacts": [".hermes/runs/goal-1-summary.json"],
        "next_action": "Resume implementation",
        "continuation_summary": "Schema and atomic checkpoint tests are complete.",
    }


def test_create_survives_session_deletion_and_database_reopen(tmp_path):
    db_path = tmp_path / "state.db"
    first = SessionDB(db_path=db_path)
    first.create_session("session-a", source="cli")

    created = _create(first)

    assert created["id"] == "goal-1"
    assert created["status"] == "created"
    assert created["version"] == 1
    assert first.delete_session("session-a") is True
    assert first.get_goal_coordinator("goal-1")["goal"] == "Ship the durable coordinator"
    first.close()

    reopened = SessionDB(db_path=db_path)
    try:
        record = reopened.get_goal_coordinator("goal-1")
        assert record["profile_name"] == "default"
        assert record["session_id"] == "session-a"
        assert record["acceptance_contract"] == {"criteria": ["focused tests pass"]}
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "delete_path",
    ["single", "if_empty", "bulk", "empty_bulk", "prune", "ghost"],
)
def test_session_deletion_paths_orphan_running_coordinator_for_resume(
    db, delete_path
):
    session_id = f"session-{delete_path}"
    db.create_session(session_id, source="tui" if delete_path == "ghost" else "cli")
    _create(db, session_id=session_id)
    db.transition_goal_coordinator(
        "goal-1", expected_version=1, new_status="running", reason="start"
    )
    if delete_path in {"empty_bulk", "prune", "ghost"}:
        db._conn.execute(
            """UPDATE sessions
               SET ended_at = ?, started_at = ?, message_count = 0
               WHERE id = ?""",
            (1.0, 1.0, session_id),
        )

    if delete_path == "single":
        assert db.delete_session(session_id)
    elif delete_path == "if_empty":
        assert db.delete_session_if_empty(session_id)
    elif delete_path == "bulk":
        assert db.delete_sessions([session_id]) == 1
    elif delete_path == "empty_bulk":
        assert db.delete_empty_sessions() == 1
    elif delete_path == "ghost":
        assert db.prune_empty_ghost_sessions() == 1
    else:
        assert db.prune_sessions(older_than_days=None) == 1

    coordinator = db.get_goal_coordinator("goal-1")
    assert coordinator["status"] == "orphaned"
    assert coordinator["session_id"] == session_id
    assert "goal-1" in {
        item["id"] for item in db.find_resumable_goal_coordinators("default")
    }
    audit = db.list_goal_coordinator_audit("goal-1")[-1]
    assert audit["to_status"] == "orphaned"
    assert audit["reason"].startswith("session-deleted:")


def test_session_reset_promotion_orphans_running_coordinator(db):
    session_id = "session-reset-promotion"
    db.create_session(session_id, source="gateway")
    _create(db, session_id=session_id)
    db.transition_goal_coordinator(
        "goal-1", expected_version=1, new_status="running", reason="start"
    )

    assert db.promote_to_session_reset(session_id, "session_reset")

    assert db.get_goal_coordinator("goal-1")["status"] == "orphaned"


def test_compression_finalization_orphans_running_child_coordinator(db):
    parent_id = "compression-parent"
    child_id = "compression-child"
    db.create_session(parent_id, source="cli")
    db.end_session(parent_id, "compression")
    db.create_session(child_id, source="cli", parent_session_id=parent_id)
    db.append_message(child_id, role="user", content="continuation")
    db._conn.execute(
        "UPDATE sessions SET started_at = ? WHERE id = ?",
        (1.0, child_id),
    )
    _create(db, session_id=child_id)
    db.transition_goal_coordinator(
        "goal-1", expected_version=1, new_status="running", reason="start"
    )

    assert db.finalize_orphaned_compression_sessions() == 1

    assert db.get_goal_coordinator("goal-1")["status"] == "orphaned"


def test_transition_and_checkpoint_are_atomic_and_version_fenced(db):
    _create(db)
    running = db.transition_goal_coordinator(
        "goal-1",
        expected_version=1,
        new_status="running",
        reason="slice-started",
    )
    assert running["version"] == 2

    checkpoint = _checkpoint()
    yielded = db.transition_goal_coordinator(
        "goal-1",
        expected_version=2,
        new_status="needs_continuation",
        reason="max_iterations_reached",
        checkpoint=checkpoint,
    )

    assert yielded["status"] == "needs_continuation"
    assert yielded["version"] == 3
    assert yielded["checkpoint"] == checkpoint

    audit = db.list_goal_coordinator_audit("goal-1")
    assert [(row["accepted"], row["to_status"]) for row in audit] == [
        (True, "created"),
        (True, "running"),
        (True, "needs_continuation"),
    ]
    history = db.list_goal_coordinator_checkpoints("goal-1")
    assert len(history) == 1
    assert history[0]["version"] == 3
    assert history[0]["checkpoint"] == checkpoint


def test_successor_binding_rejects_another_active_coordinator_in_target_session(db):
    _create(db, goal_id="goal-existing", session_id="target-session")
    db.transition_goal_coordinator(
        "goal-existing", expected_version=1, new_status="running", reason="start"
    )
    _create(db, goal_id="goal-orphan", session_id="source-session")
    db.transition_goal_coordinator(
        "goal-orphan", expected_version=1, new_status="orphaned", reason="lost-session"
    )

    with pytest.raises(RuntimeError, match="active goal coordinator"):
        db.transition_goal_coordinator(
            "goal-orphan",
            expected_version=2,
            new_status="running",
            reason="durable-resume",
            session_id="target-session",
        )

    assert db.get_goal_coordinator("goal-existing")["status"] == "running"
    assert db.get_goal_coordinator("goal-orphan")["status"] == "orphaned"
    audit = db.list_goal_coordinator_audit("goal-orphan")[-1]
    assert audit["accepted"] is False
    assert audit["event"] == "transition_rejected"


def test_successor_binding_rejects_yielded_goal_from_live_original_session(db):
    db.create_session("source-session", source="cli")
    _create(db, session_id="source-session")
    db.transition_goal_coordinator(
        "goal-1", expected_version=1, new_status="running", reason="start"
    )
    db.transition_goal_coordinator(
        "goal-1", expected_version=2, new_status="yielded", reason="pause"
    )

    with pytest.raises(RuntimeError, match="original session is still live"):
        db.transition_goal_coordinator(
            "goal-1",
            expected_version=3,
            new_status="running",
            reason="durable-resume",
            session_id="successor-session",
        )

    assert db.get_goal_coordinator("goal-1")["status"] == "yielded"
    assert db.list_goal_coordinator_audit("goal-1")[-1]["accepted"] is False


def test_successor_binding_rejects_orphan_after_original_session_reopens(db):
    db.create_session("source-session", source="cli")
    _create(db, session_id="source-session")
    db.transition_goal_coordinator(
        "goal-1", expected_version=1, new_status="running", reason="start"
    )
    db.end_session("source-session", "agent_close")
    db.reopen_session("source-session")

    with pytest.raises(RuntimeError, match="original session is still live"):
        db.transition_goal_coordinator(
            "goal-1",
            expected_version=3,
            new_status="running",
            reason="durable-resume",
            session_id="successor-session",
        )

    assert db.get_goal_coordinator("goal-1")["status"] == "orphaned"
    assert db.list_goal_coordinator_audit("goal-1")[-1]["accepted"] is False


def test_stale_write_is_rejected_and_audited(db):
    _create(db)
    db.transition_goal_coordinator(
        "goal-1", expected_version=1, new_status="running", reason="start"
    )

    with pytest.raises(RuntimeError, match="stale goal coordinator version"):
        db.transition_goal_coordinator(
            "goal-1",
            expected_version=1,
            new_status="yielded",
            reason="stale-worker",
        )

    record = db.get_goal_coordinator("goal-1")
    assert record["status"] == "running"
    assert record["version"] == 2
    rejected = db.list_goal_coordinator_audit("goal-1")[-1]
    assert rejected["accepted"] is False
    assert rejected["event"] == "transition_rejected"
    assert rejected["reason"] == "stale-worker"


def test_invalid_transition_fails_closed_and_is_audited(db):
    _create(db)

    with pytest.raises(RuntimeError, match="invalid goal coordinator transition"):
        db.transition_goal_coordinator(
            "goal-1",
            expected_version=1,
            new_status="completed",
            reason="invalid-shortcut",
        )

    record = db.get_goal_coordinator("goal-1")
    assert record["status"] == "created"
    assert record["version"] == 1
    rejected = db.list_goal_coordinator_audit("goal-1")[-1]
    assert rejected["accepted"] is False
    assert rejected["event"] == "transition_rejected"


def test_crash_before_commit_rolls_back_record_and_audit(db):
    _create(db)
    db._conn.executescript(
        """
        CREATE TRIGGER abort_accepted_goal_audit
        BEFORE INSERT ON goal_coordinator_audit
        WHEN NEW.accepted = 1
             AND NEW.event = 'transition'
        BEGIN
            SELECT RAISE(ABORT, 'simulated crash before commit');
        END;
        """
    )

    with pytest.raises(sqlite3.IntegrityError, match="simulated crash before commit"):
        db.transition_goal_coordinator(
            "goal-1",
            expected_version=1,
            new_status="running",
            reason="start",
        )

    record = db.get_goal_coordinator("goal-1")
    assert record["status"] == "created"
    assert record["version"] == 1
    assert len(db.list_goal_coordinator_audit("goal-1")) == 1


def test_secret_and_unbounded_checkpoint_payloads_are_rejected(db):
    with pytest.raises(ValueError, match="credential material"):
        db.create_goal_coordinator(
            goal_id="secret-goal-text",
            profile_name="default",
            goal="Use " + "github_" + "pat_" + ("a" * 30) + " to publish",
            acceptance_contract={},
            session_id="session-a",
        )

    with pytest.raises(ValueError, match="acceptance contract field"):
        db.create_goal_coordinator(
            goal_id="secret-goal",
            profile_name="default",
            goal="Do not persist credentials",
            acceptance_contract={"api_key": "should-never-be-stored"},
            session_id="session-a",
        )

    _create(db)
    db.transition_goal_coordinator(
        "goal-1", expected_version=1, new_status="running", reason="start"
    )

    with pytest.raises(ValueError, match="checkpoint field"):
        db.transition_goal_coordinator(
            "goal-1",
            expected_version=2,
            new_status="needs_continuation",
            reason="checkpoint",
            checkpoint={"raw_model_response": "sensitive transcript"},
        )

    with pytest.raises(ValueError, match="unsupported checkpoint field"):
        db.transition_goal_coordinator(
            "goal-1",
            expected_version=2,
            new_status="needs_continuation",
            reason="checkpoint",
            checkpoint={"notes": "raw transcript hidden under a benign key"},
        )

    with pytest.raises(ValueError, match="unsupported validation field"):
        db.transition_goal_coordinator(
            "goal-1",
            expected_version=2,
            new_status="needs_continuation",
            reason="checkpoint",
            checkpoint={"validation": [{"notes": "raw tool output"}]},
        )

    with pytest.raises(ValueError, match="credential material"):
        db.transition_goal_coordinator(
            "goal-1",
            expected_version=2,
            new_status="needs_continuation",
            reason="Use " + "github_" + "pat_" + ("b" * 30),
            checkpoint=_checkpoint(),
        )

    with pytest.raises(ValueError, match="credential material"):
        db.transition_goal_coordinator(
            "goal-1",
            expected_version=2,
            new_status="needs_continuation",
            reason="checkpoint",
            checkpoint={
                "continuation_summary": "Bearer " + ("A" * 32)
            },
        )

    with pytest.raises(ValueError, match="too large"):
        db.transition_goal_coordinator(
            "goal-1",
            expected_version=2,
            new_status="needs_continuation",
            reason="checkpoint",
            checkpoint={"continuation_summary": "x" * 70_000},
        )


def test_legacy_compatibility_payload_is_allowlisted(db):
    with pytest.raises(ValueError, match="unsupported legacy goal field"):
        db.create_goal_coordinator(
            goal_id="legacy-payload-goal",
            profile_name="default",
            goal="Reject disconnected payloads",
            acceptance_contract={},
            session_id="session-a",
            legacy_meta_key="goal:session-a",
            legacy_state_json='{"goal":"x","transcript_copy":"raw output"}',
        )

    with pytest.raises(ValueError, match="unsupported acceptance contract field"):
        db.create_goal_coordinator(
            goal_id="contract-payload-goal",
            profile_name="default",
            goal="Reject arbitrary contract payloads",
            acceptance_contract={"notes": "raw transcript"},
            session_id="session-a",
        )


def test_compact_resume_read_excludes_contract_checkpoint_and_history(db):
    _create(db)
    db.transition_goal_coordinator(
        "goal-1", expected_version=1, new_status="running", reason="start"
    )
    db.transition_goal_coordinator(
        "goal-1",
        expected_version=2,
        new_status="needs_continuation",
        reason="max_iterations_reached",
        checkpoint=_checkpoint(),
    )

    resume = db.get_goal_resume("goal-1")

    assert resume == {
        "id": "goal-1",
        "status": "needs_continuation",
        "version": 3,
        "goal": "Ship the durable coordinator",
        "session_id": "session-a",
        "current_milestone": {"id": "m1", "title": "Persistence"},
        "next_action": "Resume implementation",
        "continuation_summary": "Schema and atomic checkpoint tests are complete.",
        "updated_at": resume["updated_at"],
    }
    assert "acceptance_contract" not in resume
    assert "checkpoint" not in resume
    assert "audit" not in resume
