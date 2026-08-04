"""Focused tests for dashboard PTY reconnect breadcrumbs."""

import json
import sqlite3
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import pytest


pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"), reason="PTY bridge is POSIX-only"
)


class _OneFrameBridge:
    def __init__(self):
        self._sent = False
        self.closed = False

    @classmethod
    def spawn(cls, *args, **kwargs):
        return cls()

    def read(self, timeout):
        if not self._sent:
            self._sent = True
            return b"ready"
        return None

    def resize(self, *, cols, rows):
        pass

    def write(self, raw):
        pass

    def close(self):
        self.closed = True


@pytest.fixture
def pty_client(monkeypatch, _isolate_hermes_home):
    from starlette.testclient import TestClient

    import hermes_cli.web_server as ws

    monkeypatch.setattr(ws, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)
    monkeypatch.setattr(ws.PtyBridge, "spawn", _OneFrameBridge.spawn)
    ws.app.state.pty_active_session_files = {}

    client = TestClient(ws.app)
    return ws, client, ws._SESSION_TOKEN


def _url(token: str, **params: str) -> str:
    return f"/api/pty?{urlencode({'token': token, **params})}"






def test_fresh_param_ignores_channel_active_session_file(pty_client, monkeypatch):
    """Explicit fresh starts must not resurrect the prior channel session."""
    ws, client, token = pty_client
    channel = "fresh-chan"
    active_file = ws._active_session_file_for_channel(ws.app, channel)
    active_file.write_text(json.dumps({"session_id": "sess-old"}), encoding="utf-8")
    captured = {}

    def fake_resolve(resume=None, sidecar_url=None, profile=None, active_session_file=None):
        captured["active_session_file"] = active_session_file
        captured["resume"] = resume
        return (["fake-hermes-tui"], None, None)

    monkeypatch.setattr(ws, "_resolve_chat_argv", fake_resolve)

    with client.websocket_connect(_url(token, channel=channel, fresh="1")) as conn:
        assert conn.receive_bytes() == b"ready"

    assert captured["resume"] is None
    assert captured["active_session_file"] == str(active_file)
    assert not active_file.exists()


def test_child_eof_closes_socket_and_bridge(pty_client, monkeypatch):
    """Child EOF must close the WS server-side and reap the PTY.

    Regression for the FD leak (#54028): the reader task hits EOF when the
    PTY child exits, but if the browser's socket is half-open (no FIN), the
    writer loop's ``ws.receive()`` would block forever and the PTY fds would
    never be closed. The reader now closes the WebSocket on EOF so the
    handler's ``finally`` runs ``bridge.close()``.
    """
    ws, client, token = pty_client
    bridges = []

    class _RecordingBridge(_OneFrameBridge):
        @classmethod
        def spawn(cls, *args, **kwargs):
            b = cls()
            bridges.append(b)
            return b

    monkeypatch.setattr(ws.PtyBridge, "spawn", _RecordingBridge.spawn)
    monkeypatch.setattr(
        ws, "_resolve_chat_argv", lambda **kw: (["fake-hermes-tui"], None, None)
    )

    # The client never sends a disconnect of its own — it only reads the one
    # frame then the server side must tear everything down on child EOF.
    with client.websocket_connect(_url(token, channel="eof-chan")) as conn:
        assert conn.receive_bytes() == b"ready"
        # Server closes the socket after the child EOFs; receiving again
        # surfaces the close rather than hanging.
        with pytest.raises(Exception):
            conn.receive_bytes()

    assert len(bridges) == 1
    # bridge.close() runs in the handler's `finally` via asyncio.to_thread,
    # which can lag the client-side context exit by a tick or two. Poll briefly
    # instead of asserting immediately so the teardown isn't a race.
    import time

    deadline = time.monotonic() + 5.0
    while not bridges[0].closed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert bridges[0].closed is True


class _IdleBridge:
    """One deterministic frame followed by realistic blocking idle reads."""

    def __init__(self):
        self.closed = False
        self._ready = True

    def read(self, timeout):
        if self._ready:
            self._ready = False
            return b"ready"
        time.sleep(timeout)
        return b""

    def write(self, raw):
        pass

    def resize(self, *, cols, rows):
        pass

    def close(self):
        self.closed = True


@pytest.fixture
def lifecycle_client(monkeypatch, _isolate_hermes_home):
    """Fresh bounded registry plus a real lifespan and temporary Hermes home."""
    from starlette.testclient import TestClient

    import hermes_cli.web_server as ws
    from hermes_cli.pty_session import PtySessionRegistry

    registry = PtySessionRegistry(
        ttl=60.0, max_sessions=4, buffer_cap=1024, read_timeout=0.01
    )
    monkeypatch.setattr(ws, "PTY_REGISTRY", registry)
    monkeypatch.setattr(ws, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)
    monkeypatch.setattr(
        ws, "_resolve_chat_argv", lambda **_kwargs: (["unused"], None, None)
    )
    with TestClient(ws.app) as client:
        yield ws, client, ws._SESSION_TOKEN, registry


def test_fresh_attach_rotates_pty_and_status_is_allowlisted(lifecycle_client, monkeypatch):
    ws, client, token, registry = lifecycle_client
    bridges = []

    def spawn(*_args, **_kwargs):
        bridge = _IdleBridge()
        bridges.append(bridge)
        return bridge

    monkeypatch.setattr(ws.PtyBridge, "spawn", spawn)
    attach = "not-status-telemetry"
    with client.websocket_connect(_url(token, attach=attach)) as first:
        assert first.receive_bytes() == b"ready"
    with client.websocket_connect(_url(token, attach=attach, fresh="1")) as second:
        assert second.receive_bytes() == b"ready"

    status = client.get(
        "/api/status", headers={ws._SESSION_HEADER_NAME: token}
    ).json()["components"]["pty"]
    assert len(bridges) == 2
    assert bridges[0].closed is True
    assert bridges[1].closed is False
    assert status["sessions"]["total"] == 1
    assert status["events"]["forced_fresh"] == 1
    assert set(status) == {"status", "sessions", "events", "websocket_failures"}
    assert attach not in repr(status)
    assert registry.snapshot()["sessions"]["total"] == 1


def test_reconnect_storm_bounds_pty_handles_and_durable_session_rows(
    lifecycle_client, monkeypatch
):
    ws, client, token, registry = lifecycle_client
    bridges = []
    channel = "row-bound-channel"
    active_file = ws._active_session_file_for_channel(ws.app, channel)
    active_file.write_text(json.dumps({"session_id": "pty-1"}), encoding="utf-8")

    from hermes_state import SessionDB

    db = SessionDB()
    try:
        db.create_session("pty-1", "dashboard_pty_test")
    finally:
        db.close()

    def spawn(*_args, **_kwargs):
        bridge = _IdleBridge()
        bridges.append(bridge)
        return bridge

    monkeypatch.setattr(ws.PtyBridge, "spawn", spawn)
    for _ in range(8):
        with client.websocket_connect(
            _url(token, attach="same-channel", channel=channel)
        ) as conn:
            assert conn.receive_bytes() == b"ready"

    db = SessionDB()
    try:
        with sqlite3.connect(db.db_path) as conn:
            rows = conn.execute(
                "SELECT count(*) FROM sessions WHERE source = ?",
                ("dashboard_pty_test",),
            ).fetchone()[0]
    finally:
        db.close()

    assert len(bridges) == 1
    assert rows == 1
    assert registry.snapshot()["sessions"]["total"] == 1


def test_rotation_race_closes_stale_attach_without_server_error(
    lifecycle_client, monkeypatch
):
    from starlette.websockets import WebSocketDisconnect

    _ws, client, token, registry = lifecycle_client

    class StaleSession:
        async def attach(self, _socket):
            raise RuntimeError("cannot attach to closed PTY session")

    async def stale_lookup(_key, *, spawn):
        del spawn
        return StaleSession(), False

    monkeypatch.setattr(registry, "attach_or_spawn", stale_lookup)

    with client.websocket_connect(_url(token, attach="rotated")) as conn:
        with pytest.raises(WebSocketDisconnect) as closed:
            conn.receive_bytes()

    assert closed.value.code == 4409
