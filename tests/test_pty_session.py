import asyncio
import threading
import time

import pytest

from hermes_cli.pty_session import RingBuffer


def test_ringbuffer_keeps_everything_under_capacity():
    rb = RingBuffer(10)
    rb.append(b"abc")
    rb.append(b"def")
    assert rb.snapshot() == b"abcdef"
    assert rb.truncated is False


def test_ringbuffer_drops_oldest_over_capacity():
    rb = RingBuffer(4)
    rb.append(b"abcdef")          # 6 bytes into a 4-byte buffer
    assert rb.snapshot() == b"cdef"
    assert rb.truncated is True


def test_ringbuffer_truncation_across_appends():
    rb = RingBuffer(3)
    rb.append(b"ab")
    rb.append(b"cd")             # now "abcd" -> keep "bcd"
    assert rb.snapshot() == b"bcd"
    assert rb.truncated is True


class FakeBridge:
    """Implements the bridge contract PtySession depends on."""

    def __init__(self, chunks):
        self._chunks = list(chunks)   # bytes; b"" = idle tick; None = EOF
        self.written = bytearray()
        self.closed = False
        self.resized = None

    def read(self, timeout):
        if not self._chunks:
            return b""                # idle
        return self._chunks.pop(0)

    def write(self, data):
        self.written.extend(data)

    def resize(self, cols, rows):
        self.resized = (cols, rows)

    def close(self):
        self.closed = True


class FakeWS:
    def __init__(self):
        self.sent = []               # list of ("bytes"|"text", payload)
        self.close_code = None

    async def send_bytes(self, data):
        self.sent.append(("bytes", bytes(data)))

    async def send_text(self, text):
        self.sent.append(("text", text))

    async def close(self, code=1000, reason=""):
        self.close_code = code


class SlowCloseBridge(FakeBridge):
    def __init__(self, chunks, close_started, release_close):
        super().__init__(chunks)
        self._close_started = close_started
        self._release_close = release_close

    def close(self):
        self._close_started.set()
        self._release_close.wait(timeout=2)
        self.closed = True


@pytest.mark.asyncio
async def test_attach_replays_buffer_then_streams_live():
    from hermes_cli.pty_session import PtySession
    bridge = FakeBridge([b"hello ", b"world", None])
    s = PtySession("k", bridge, buffer_cap=1024, read_timeout=0.01)
    await s.start()
    await asyncio.sleep(0.05)                      # drain consumes "hello world"
    ws = FakeWS()
    await s.attach(ws)
    replay = b"".join(p for kind, p in ws.sent if kind == "bytes")
    assert replay == b"hello world"
    await s.close()


@pytest.mark.asyncio
async def test_detach_keeps_draining_into_buffer():
    from hermes_cli.pty_session import PtySession
    bridge = FakeBridge([b"one", b"", b"two"])
    s = PtySession("k", bridge, buffer_cap=1024, read_timeout=0.01)
    await s.start()
    ws = FakeWS()
    await s.attach(ws)
    s.detach(ws)
    assert s.attached is False
    assert s.last_detached_at is not None
    await asyncio.sleep(0.05)                      # "two" drains while detached
    ws2 = FakeWS()
    await s.attach(ws2)
    replay = b"".join(p for kind, p in ws2.sent if kind == "bytes")
    assert replay == b"onetwo"
    await s.close()


@pytest.mark.asyncio
async def test_eof_marks_dead_and_closes_socket_4410():
    from hermes_cli.pty_session import PtySession
    bridge = FakeBridge([b"bye", None])
    s = PtySession("k", bridge, buffer_cap=1024, read_timeout=0.01)
    await s.start()
    ws = FakeWS()
    await s.attach(ws)
    await asyncio.sleep(0.05)                      # drain hits None (EOF)
    assert s.alive is False
    assert ws.close_code == 4410
    await s.close()


@pytest.mark.asyncio
async def test_attach_supersedes_previous_socket_without_detaching_new_viewer():
    from hermes_cli.pty_session import PtySession

    bridge = FakeBridge([b"ready", b"", b""])
    s = PtySession("k", bridge, buffer_cap=1024, read_timeout=0.01)
    await s.start()
    first = FakeWS()
    second = FakeWS()

    await s.attach(first)
    await s.attach(second)
    s.detach(first)

    assert first.close_code == 4409
    assert s.attached is True
    assert s.last_detached_at is None
    s.detach(second)
    assert s.attached is False
    await s.close()


from hermes_cli.pty_session import PtySessionRegistry, RegistryFull


def make_registry(ttl=1800.0, max_sessions=16):
    return PtySessionRegistry(ttl=ttl, max_sessions=max_sessions,
                              buffer_cap=1024, read_timeout=0.01)


@pytest.mark.asyncio
async def test_same_key_reattaches_same_session():
    reg = make_registry()
    b1 = FakeBridge([b"", b"", b""])
    s1, created1 = await reg.attach_or_spawn("tok", spawn=lambda: b1)
    s2, created2 = await reg.attach_or_spawn("tok", spawn=lambda: FakeBridge([]))
    assert created1 is True and created2 is False
    assert s1 is s2
    assert s2.bridge is b1                     # second spawn callable was NOT used
    await reg.close_all()


@pytest.mark.asyncio
async def test_concurrent_same_key_attach_spawns_one_living_session():
    reg = make_registry()
    spawned = []

    async def spawn():
        spawned.append(FakeBridge([b"", b""]))
        await asyncio.sleep(0.02)
        return spawned[-1]

    results = await asyncio.gather(
        *(reg.attach_or_spawn("storm", spawn=spawn) for _ in range(8))
    )

    sessions = {id(session) for session, _created in results}
    assert len(spawned) == 1
    assert len(sessions) == 1
    assert sum(1 for _session, created in results if created) == 1
    assert reg.snapshot()["sessions"]["alive"] == 1
    await reg.close_all()


@pytest.mark.asyncio
async def test_rotate_forces_fresh_session_and_closes_existing():
    reg = make_registry()
    old_bridge = FakeBridge([b"", b""])
    new_bridge = FakeBridge([b"", b""])

    old_session, _ = await reg.attach_or_spawn("tok", spawn=lambda: old_bridge)
    new_session, created = await reg.rotate("tok", spawn=lambda: new_bridge)

    assert created is True
    assert new_session is not old_session
    assert old_bridge.closed is True
    assert reg.snapshot()["events"]["forced_fresh"] == 1
    await reg.close_all()


@pytest.mark.asyncio
async def test_dead_session_is_replaced_and_counted():
    reg = make_registry()
    old_bridge = FakeBridge([None])
    new_bridge = FakeBridge([b"", b""])
    old_session, _ = await reg.attach_or_spawn("tok", spawn=lambda: old_bridge)
    await asyncio.sleep(0.05)

    new_session, created = await reg.attach_or_spawn("tok", spawn=lambda: new_bridge)

    assert created is True
    assert new_session is not old_session
    assert old_bridge.closed is True
    assert reg.snapshot()["events"]["dead_replace"] == 1
    await reg.close_all()


@pytest.mark.asyncio
async def test_reap_idle_closes_sessions_past_ttl():
    reg = make_registry(ttl=10.0)
    b = FakeBridge([b"", b""])
    s, _ = await reg.attach_or_spawn("tok", spawn=lambda: b)
    ws = FakeWS()
    await s.attach(ws)
    s.detach(ws)
    s.last_detached_at = time.monotonic() - 11.0   # detached 11s ago, ttl 10s
    await reg.reap_idle()
    assert b.closed is True
    s2, created = await reg.attach_or_spawn("tok", spawn=lambda: FakeBridge([]))
    assert created is True
    await reg.close_all()


@pytest.mark.asyncio
async def test_reap_idle_closes_sessions_at_ttl_boundary():
    reg = make_registry(ttl=10.0)
    b = FakeBridge([b"", b""])
    s, _ = await reg.attach_or_spawn("tok", spawn=lambda: b)
    ws = FakeWS()
    await s.attach(ws)
    s.detach(ws)
    s.last_detached_at = time.monotonic() - 10.0
    await reg.reap_idle(now=s.last_detached_at + 10.0)
    assert b.closed is True


@pytest.mark.asyncio
async def test_registry_full_eviction_closes_oldest_idle_before_returning():
    reg = make_registry(max_sessions=2)
    b1 = FakeBridge([b"", b""])
    b2 = FakeBridge([b"", b""])
    b3 = FakeBridge([b"", b""])
    s1, _ = await reg.attach_or_spawn("a", spawn=lambda: b1)
    s2, _ = await reg.attach_or_spawn("b", spawn=lambda: b2)
    ws1 = FakeWS()
    ws2 = FakeWS()
    await s1.attach(ws1)
    await s2.attach(ws2)
    s1.detach(ws1)
    s2.detach(ws2)
    s1.last_detached_at = time.monotonic() - 20.0
    s2.last_detached_at = time.monotonic() - 10.0

    await reg.attach_or_spawn("c", spawn=lambda: b3)

    assert b1.closed is True
    snap = reg.snapshot()
    assert snap["sessions"]["total"] == 2
    assert snap["events"]["registry_full"] == 1
    await reg.close_all()


@pytest.mark.asyncio
async def test_new_key_at_capacity_raises_when_none_reapable():
    reg = make_registry(max_sessions=1)
    b = FakeBridge([b"", b""])
    s, _ = await reg.attach_or_spawn("a", spawn=lambda: b)
    await s.attach(FakeWS())                    # attached → not reapable
    with pytest.raises(RegistryFull):
        await reg.attach_or_spawn("b", spawn=lambda: FakeBridge([]))
    assert reg.snapshot()["events"]["registry_full"] == 1
    await reg.close_all()


def test_snapshot_reports_secret_free_counts_and_failures():
    reg = make_registry(ttl=15.0, max_sessions=3)

    reg.record_ws_failure("registry_full")
    reg.record_ws_failure("not-a-public-detail")

    snap = reg.snapshot(now=time.monotonic())
    assert snap["sessions"]["total"] == 0
    assert snap["sessions"]["max"] == 3
    assert snap["sessions"]["ttl_seconds"] == 15
    assert snap["websocket_failures"]["registry_full"] == 1
    assert snap["websocket_failures"]["unexpected"] == 1
    assert "tok" not in repr(snap)
    assert "not-a-public-detail" not in repr(snap)


@pytest.mark.asyncio
async def test_close_all_finishes_bounded_cleanup_when_cancelled():
    reg = make_registry()
    close_started = threading.Event()
    release_close = threading.Event()
    bridge = SlowCloseBridge([b"", b""], close_started, release_close)
    await reg.attach_or_spawn("tok", spawn=lambda: bridge)
    task = asyncio.create_task(reg.close_all())
    await asyncio.to_thread(close_started.wait, 2)

    task.cancel()
    release_close.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert bridge.closed is True
    with pytest.raises(RuntimeError):
        await reg.attach_or_spawn("new", spawn=lambda: FakeBridge([]))


@pytest.mark.asyncio
async def test_reaper_loop_invokes_reap(monkeypatch):
    from hermes_cli.pty_session import run_reaper
    reg = make_registry()
    calls = {"n": 0}

    async def fake_reap(now=None):
        calls["n"] += 1

    monkeypatch.setattr(reg, "reap_idle", fake_reap)
    task = asyncio.create_task(run_reaper(reg, interval=0.01))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert calls["n"] >= 2
