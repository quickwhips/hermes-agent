from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
import warnings

import pytest
from fastapi import HTTPException

from hermes_cli import web_server
from hermes_cli.sqlite_safe_read import has_live_connection
from hermes_state import SessionDB


def _seed_session(db_path, session_id: str = "session-read") -> None:
    db = SessionDB(db_path=db_path)
    try:
        db.create_session(session_id, source="cli")
        db.append_message(session_id, "user", "read-only contract")
    finally:
        db.close()


def _seed_legacy_session(db_path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, model TEXT, title TEXT, started_at REAL, ended_at REAL);
            CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, timestamp REAL);
            INSERT INTO sessions VALUES ('legacy-session', 'cron', 'legacy-model', 'Legacy title', 1000, NULL);
            INSERT INTO messages VALUES (1, 'legacy-session', 'user', 'legacy needle transcript', 1001);
        """)


def test_read_only_legacy_schema_serves_dashboard_reads_without_schema_writes(tmp_path, monkeypatch):
    import hermes_state

    db_path = tmp_path / "state.db"
    _seed_legacy_session(db_path)
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)
    before = db_path.read_bytes()
    db = SessionDB(db_path=db_path, read_only=True)
    statements: list[str] = []
    db._conn.set_trace_callback(statements.append)
    try:
        assert db.list_sessions_rich(include_archived=True)[0]["id"] == "legacy-session"
        assert db.get_messages("legacy-session")[0]["content"] == "legacy needle transcript"
        assert db.search_messages("needle")[0]["session_id"] == "legacy-session"
        assert db.export_session("legacy-session")["messages"][0]["id"] == 1
    finally:
        db.close()
    assert db_path.read_bytes() == before
    assert not any(token in sql.lower() for sql in statements for token in ("create ", "alter ", "insert ", "update ", "delete "))
    assert web_server.get_sessions()["sessions"][0]["id"] == "legacy-session"
    assert asyncio.run(web_server.get_session_detail("legacy-session"))["title"] == "Legacy title"
    assert asyncio.run(web_server.get_session_messages("legacy-session"))["messages"][0]["id"] == 1
    assert asyncio.run(web_server.search_sessions(q="needle"))["results"][0]["id"] == "legacy-session"
    assert asyncio.run(web_server.export_session_endpoint("legacy-session"))["messages"][0]["id"] == 1
    assert asyncio.run(web_server.get_session_stats())["total"] == 1
    assert web_server._get_usage_analytics(days=7)["daily"] == []
    assert web_server._get_models_analytics(days=7)["models"] == []


def test_search_read_runs_off_event_loop_and_closes(monkeypatch):
    loop_thread = threading.get_ident()
    db_threads: list[int] = []
    closed: list[bool] = []

    class _DB:
        def search_sessions_by_id(self, *args, **kwargs):
            db_threads.append(threading.get_ident())
            time.sleep(0.05)
            return []

        def search_messages(self, *args, **kwargs):
            db_threads.append(threading.get_ident())
            return []

        def close(self):
            db_threads.append(threading.get_ident())
            closed.append(True)

    monkeypatch.setattr(
        web_server,
        "_open_session_db_for_profile",
        lambda profile=None, *, read_only: _DB(),
    )
    monkeypatch.setattr(web_server, "_session_db_exists_for_profile", lambda profile=None: True)

    result = asyncio.run(web_server.search_sessions(q="needle"))

    assert result == {"results": []}
    assert closed == [True]
    assert db_threads
    assert all(thread_id != loop_thread for thread_id in db_threads)


def test_session_list_separates_writable_auto_archive_from_read_only_query(monkeypatch):
    modes: list[bool] = []
    closed: list[str] = []

    class _ReadOnlyDB:
        def list_sessions_rich(self, **kwargs):
            return [{"id": "listed", "started_at": 1, "ended_at": 2}]

        def session_count(self, **kwargs):
            return 1

        def close(self):
            closed.append("read")

    def _open(profile=None, *, read_only):
        modes.append(read_only)
        return _ReadOnlyDB()

    archived: list[object] = []
    monkeypatch.setattr(web_server, "_open_session_db_for_profile", _open)
    monkeypatch.setattr(web_server, "_session_db_exists_for_profile", lambda profile=None: True)
    monkeypatch.setattr(
        web_server,
        "_maybe_auto_archive_for_profile",
        lambda db, profile: archived.append(db),
    )

    result = web_server.get_sessions()

    assert modes == [True]
    assert len(archived) == 1
    assert archived[0] is None
    assert closed == ["read"]
    assert result["total"] == 1
    assert result["sessions"][0]["id"] == "listed"


def test_disabled_auto_archive_does_not_open_a_writable_connection(monkeypatch):
    from hermes_cli import config as config_module

    opened: list[bool] = []
    monkeypatch.setattr(
        web_server,
        "_open_session_db_for_profile",
        lambda profile=None, *, read_only: opened.append(read_only),
    )
    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda: {"sessions": {"auto_archive": False}},
    )
    web_server._last_auto_archive_check.clear()

    web_server._maybe_auto_archive_for_profile(None, None)

    assert opened == []


def test_enabled_auto_archive_lazily_opens_its_own_writable_connection(monkeypatch):
    from hermes_cli import config as config_module

    opened: list[bool] = []
    archived: list[tuple[float, int]] = []
    closed: list[bool] = []

    class _DB:
        def maybe_auto_archive(self, *, idle_days, min_interval_hours):
            archived.append((idle_days, min_interval_hours))

        def close(self):
            closed.append(True)

    monkeypatch.setattr(
        web_server,
        "_open_session_db_for_profile",
        lambda profile=None, *, read_only: opened.append(read_only) or _DB(),
    )
    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda: {
            "sessions": {
                "auto_archive": True,
                "auto_archive_days": 4,
                "min_interval_hours": 12,
            }
        },
    )
    web_server._last_auto_archive_check.clear()

    web_server._maybe_auto_archive_for_profile(None, None)

    assert opened == [False]
    assert archived == [(4.0, 12)]
    assert closed == [True]


def test_read_only_close_never_attempts_wal_checkpoint(tmp_path):
    db_path = tmp_path / "state.db"
    _seed_session(db_path)
    statements: list[str] = []

    db = SessionDB(db_path=db_path, read_only=True)
    db._conn.set_trace_callback(statements.append)
    db.close()

    assert not any("wal_checkpoint" in sql.lower() for sql in statements)


def test_read_only_open_skips_schema_init_and_preserves_metadata(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    _seed_session(db_path)
    with sqlite3.connect(db_path) as conn:
        before = conn.execute("SELECT key, value FROM state_meta ORDER BY key").fetchall()

    def _unexpected_schema_init(self):
        raise AssertionError("read endpoint attempted schema initialization")

    monkeypatch.setattr(SessionDB, "_init_schema", _unexpected_schema_init)
    db = SessionDB(db_path=db_path, read_only=True)
    try:
        assert db.get_session("session-read")["id"] == "session-read"
    finally:
        db.close()

    with sqlite3.connect(db_path) as conn:
        after = conn.execute("SELECT key, value FROM state_meta ORDER BY key").fetchall()
    assert after == before


def test_read_only_open_never_falls_back_to_creating_missing_database(tmp_path):
    db_path = tmp_path / "missing" / "state.db"

    with pytest.raises(sqlite3.OperationalError, match="unable to open database file"):
        SessionDB(db_path=db_path, read_only=True)

    assert not db_path.exists()
    assert not db_path.parent.exists()


def test_missing_database_read_routes_return_empty_or_not_found(tmp_path, monkeypatch):
    import hermes_state

    db_path = tmp_path / "missing" / "state.db"
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)

    assert web_server.get_sessions() == {
        "sessions": [],
        "total": 0,
        "limit": 20,
        "offset": 0,
    }
    assert asyncio.run(web_server.search_sessions(q="needle")) == {"results": []}
    assert asyncio.run(web_server.count_empty_sessions_endpoint()) == {"count": 0}
    assert asyncio.run(web_server.get_session_stats()) == {
        "total": 0,
        "active_store": 0,
        "archived": 0,
        "messages": 0,
        "by_source": {},
    }
    assert web_server._list_cron_job_runs_sync("never-ran") == {
        "runs": [],
        "limit": 20,
    }
    assert web_server._get_usage_analytics(days=7) == {
        "daily": [],
        "by_model": [],
        "by_task": [],
        "totals": {
            "total_input": None,
            "total_output": None,
            "total_cache_read": None,
            "total_reasoning": None,
            "total_estimated_cost": 0,
            "total_actual_cost": 0,
            "total_sessions": 0,
            "total_api_calls": None,
        },
        "period_days": 7,
        "skills": {
            "summary": {
                "total_skill_loads": 0,
                "total_skill_edits": 0,
                "total_skill_actions": 0,
                "distinct_skills_used": 0,
            },
            "top_skills": [],
        },
        "tools": [],
    }
    assert web_server._get_models_analytics(days=7) == {
        "models": [],
        "totals": {
            "distinct_models": 0,
            "total_input": None,
            "total_output": None,
            "total_cache_read": None,
            "total_reasoning": None,
            "total_estimated_cost": 0,
            "total_actual_cost": 0,
            "total_sessions": 0,
            "total_api_calls": None,
        },
        "period_days": 7,
    }

    for read in (
        lambda: web_server.get_session_detail("missing"),
        lambda: web_server.get_session_latest_descendant("missing"),
        lambda: web_server.get_session_messages("missing"),
        lambda: web_server.export_session_endpoint("missing"),
    ):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(read())
        assert exc_info.value.status_code == 404

    assert not db_path.exists()
    assert not db_path.parent.exists()


def test_missing_named_profile_analytics_preserve_empty_contract(tmp_path, monkeypatch):
    profile_home = tmp_path / "missing-profile"
    monkeypatch.setattr(
        web_server,
        "_cron_profile_home",
        lambda profile: (profile, profile_home),
    )

    assert web_server._get_usage_analytics(days=3, profile="fresh") == {
        "daily": [],
        "by_model": [],
        "by_task": [],
        "totals": {
            "total_input": None,
            "total_output": None,
            "total_cache_read": None,
            "total_reasoning": None,
            "total_estimated_cost": 0,
            "total_actual_cost": 0,
            "total_sessions": 0,
            "total_api_calls": None,
        },
        "period_days": 3,
        "skills": {
            "summary": {
                "total_skill_loads": 0,
                "total_skill_edits": 0,
                "total_skill_actions": 0,
                "distinct_skills_used": 0,
            },
            "top_skills": [],
        },
        "tools": [],
    }
    assert web_server._get_models_analytics(days=3, profile="fresh") == {
        "models": [],
        "totals": {
            "distinct_models": 0,
            "total_input": None,
            "total_output": None,
            "total_cache_read": None,
            "total_reasoning": None,
            "total_estimated_cost": 0,
            "total_actual_cost": 0,
            "total_sessions": 0,
            "total_api_calls": None,
        },
        "period_days": 3,
    }
    assert not profile_home.exists()


def test_missing_database_resume_lookup_preserves_requested_session(tmp_path, monkeypatch):
    import hermes_state

    db_path = tmp_path / "missing" / "state.db"
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr(
        "hermes_cli.main._make_tui_argv",
        lambda root, tui_dev=False: (["node", "fake-tui.js"], None),
    )

    _argv, _cwd, env = web_server._resolve_chat_argv(resume="missing-session")

    assert env is not None
    assert env["HERMES_TUI_RESUME"] == "missing-session"
    assert not db_path.exists()
    assert not db_path.parent.exists()


def test_authenticated_detail_reads_through_writer_lock_and_keeps_contract(
    tmp_path, monkeypatch
):
    import hermes_state

    db_path = tmp_path / "state.db"
    _seed_session(db_path, "locked-session")
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)

    lock = sqlite3.connect(db_path, isolation_level=None)
    lock.execute("BEGIN IMMEDIATE")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Using `httpx` with `starlette.testclient` is deprecated",
        )
        from starlette.testclient import TestClient

        client = TestClient(web_server.app)
    try:
        unauthorized = client.get("/api/sessions/locked-session")
        assert unauthorized.status_code == 401

        response = client.get(
            "/api/sessions/locked-session",
            headers={web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN},
        )
        assert response.status_code == 200
        assert response.json()["id"] == "locked-session"
        assert response.json()["profile"] == "default"
        assert response.json()["is_default_profile"] is True
    finally:
        client.close()
        lock.execute("ROLLBACK")
        lock.close()

    assert not has_live_connection(db_path)


def test_large_wal_repeated_reads_complete_during_concurrent_writes(
    tmp_path, monkeypatch
):
    import hermes_state

    monkeypatch.setattr(
        hermes_state,
        "is_sqlite_wal_reset_vulnerable",
        lambda version_info=None: False,
    )
    db_path = tmp_path / "state.db"
    writer = SessionDB(db_path=db_path)
    writer.create_session("busy-session", source="cli")
    writer._conn.execute("PRAGMA wal_autocheckpoint=0")

    pin = sqlite3.connect(db_path, isolation_level=None)
    pin.execute("BEGIN")
    pin.execute("SELECT COUNT(*) FROM messages").fetchone()

    payload = "x" * 8192
    for index in range(96):
        writer.append_message("busy-session", "user", f"{index}:{payload}")

    wal_path = db_path.with_name(f"{db_path.name}-wal")
    assert wal_path.stat().st_size >= 512 * 1024

    errors: list[Exception] = []

    def _write_more() -> None:
        try:
            for index in range(24):
                writer.append_message("busy-session", "assistant", f"live-{index}:{payload}")
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    thread = threading.Thread(target=_write_more)
    thread.start()
    latencies: list[float] = []
    try:
        for _ in range(12):
            started = time.monotonic()
            reader = SessionDB(db_path=db_path, read_only=True)
            try:
                assert reader.get_session("busy-session")["id"] == "busy-session"
                assert reader.get_messages("busy-session", limit=1)
            finally:
                reader.close()
            latencies.append(time.monotonic() - started)
    finally:
        thread.join(timeout=10)
        pin.execute("ROLLBACK")
        pin.close()
        writer.close()

    assert not thread.is_alive()
    assert errors == []
    assert max(latencies) < 2.0
    assert not has_live_connection(db_path)
