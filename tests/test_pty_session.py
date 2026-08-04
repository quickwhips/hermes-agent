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
    assert ws.close_code == 4410
    assert s.attached is False
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
async def test_eof_closes_bridge_before_periodic_reap():
    from hermes_cli.pty_session import PtySession

    bridge = FakeBridge([None])
    session = PtySession("k", bridge, buffer_cap=1024, read_timeout=0.01)

    await session.start()
    await asyncio.sleep(0.05)

    assert session.alive is False
    assert bridge.closed is True
    await session.close()


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
async def test_new_key_at_capacity_raises_when_none_reapable():
    reg = make_registry(max_sessions=1)
    b = FakeBridge([b"", b""])
    s, _ = await reg.attach_or_spawn("a", spawn=lambda: b)
    await s.attach(FakeWS())                    # attached → not reapable
    with pytest.raises(RegistryFull):
        await reg.attach_or_spawn("b", spawn=lambda: FakeBridge([]))
    await reg.close_all()


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


@pytest.mark.asyncio
async def test_concurrent_same_key_reconnect_storm_spawns_one_session():
    reg = make_registry()
    bridges = []

    async def spawn():
        bridge = FakeBridge([b"", b""])
        bridges.append(bridge)
        await asyncio.sleep(0.01)
        return bridge

    results = await asyncio.gather(
        *(reg.attach_or_spawn("storm", spawn=spawn) for _ in range(12))
    )

    assert len(bridges) == 1
    assert len({id(session) for session, _created in results}) == 1
    assert sum(created for _session, created in results) == 1
    await reg.close_all()


@pytest.mark.asyncio
async def test_rotate_replaces_live_session_and_closes_old_bridge():
    reg = make_registry()
    old_bridge = FakeBridge([b"", b""])
    new_bridge = FakeBridge([b"", b""])
    old_session, _ = await reg.attach_or_spawn("tok", spawn=lambda: old_bridge)

    new_session, created = await reg.rotate("tok", spawn=lambda: new_bridge)

    assert created is True
    assert new_session is not old_session
    assert old_bridge.closed is True
    assert len(reg._sessions) == 1
    await reg.close_all()


@pytest.mark.asyncio
async def test_session_removed_by_rotation_rejects_late_attach():
    reg = make_registry()
    old_session, _ = await reg.attach_or_spawn(
        "tok", spawn=lambda: FakeBridge([b"", b""])
    )
    await reg.rotate("tok", spawn=lambda: FakeBridge([b"", b""]))

    ws = FakeWS()
    with pytest.raises(RuntimeError, match="closed PTY session"):
        await old_session.attach(ws)

    assert old_session.attached is False
    assert ws.sent == []
    await reg.close_all()


@pytest.mark.asyncio
async def test_rotation_closes_attached_socket_before_replacing_bridge():
    reg = make_registry()
    old_session, _ = await reg.attach_or_spawn(
        "tok", spawn=lambda: FakeBridge([b"", b""])
    )
    old_ws = FakeWS()
    await old_session.attach(old_ws)

    await reg.rotate("tok", spawn=lambda: FakeBridge([b"", b""]))

    assert old_ws.close_code == 4409
    assert old_session.attached is False
    await reg.close_all()


@pytest.mark.asyncio
async def test_eof_during_supersede_closes_new_socket_without_dead_attach():
    from hermes_cli.pty_session import PtySession

    eof_enabled = threading.Event()
    eof_read = threading.Event()

    class GatedEofBridge(FakeBridge):
        def read(self, _timeout):
            if not eof_enabled.is_set():
                return b""
            eof_read.set()
            return None

    supersede_started = asyncio.Event()
    release_supersede = asyncio.Event()

    class BlockingOldWS(FakeWS):
        async def close(self, code=1000, reason=""):
            if code == 4409:
                supersede_started.set()
                await release_supersede.wait()
            await super().close(code=code, reason=reason)

    session = PtySession(
        "tok", GatedEofBridge([]), buffer_cap=1024, read_timeout=0.01
    )
    await session.start()
    old_ws = BlockingOldWS()
    new_ws = FakeWS()
    await session.attach(old_ws)

    attach_task = asyncio.create_task(session.attach(new_ws))
    await asyncio.wait_for(supersede_started.wait(), 2)
    eof_enabled.set()
    assert await asyncio.to_thread(eof_read.wait, 2)
    release_supersede.set()
    await attach_task

    for _ in range(100):
        if new_ws.close_code is not None:
            break
        await asyncio.sleep(0.01)

    assert new_ws.close_code == 4410
    assert session.alive is False
    assert session.attached is False
    await session.close()


@pytest.mark.asyncio
async def test_failed_buffer_replay_rolls_back_attach_and_allows_capacity_eviction():
    reg = make_registry(max_sessions=1)
    session, _ = await reg.attach_or_spawn(
        "first", spawn=lambda: FakeBridge([b"", b""])
    )
    session.buffer.append(b"history")

    class FailingReplayWS(FakeWS):
        async def send_bytes(self, _data):
            raise RuntimeError("socket disconnected during replay")

    with pytest.raises(RuntimeError, match="disconnected during replay"):
        await session.attach(FailingReplayWS())

    assert session.attached is False
    assert session.last_detached_at is not None
    replacement, created = await reg.attach_or_spawn(
        "second", spawn=lambda: FakeBridge([b"", b""])
    )
    assert created is True
    assert replacement.key == "second"
    await reg.close_all()


@pytest.mark.asyncio
async def test_cancelled_buffer_replay_rolls_back_partial_attach():
    reg = make_registry()
    session, _ = await reg.attach_or_spawn(
        "tok", spawn=lambda: FakeBridge([b"", b""])
    )
    session.buffer.append(b"history")
    replay_started = asyncio.Event()

    class BlockingReplayWS(FakeWS):
        async def send_bytes(self, _data):
            replay_started.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(session.attach(BlockingReplayWS()))
    await asyncio.wait_for(replay_started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert session.attached is False
    assert session.last_detached_at is not None
    await reg.close_all()


@pytest.mark.asyncio
async def test_cancelled_socket_supersede_detaches_old_socket():
    reg = make_registry()
    session, _ = await reg.attach_or_spawn(
        "tok", spawn=lambda: FakeBridge([b"", b""])
    )
    close_started = asyncio.Event()

    class BlockingCloseWS(FakeWS):
        async def close(self, code=1000, reason=""):
            close_started.set()
            await asyncio.Event().wait()

    await session.attach(BlockingCloseWS())
    task = asyncio.create_task(session.attach(FakeWS()))
    await asyncio.wait_for(close_started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert session.attached is False
    assert session.last_detached_at is not None
    await reg.close_all()


@pytest.mark.asyncio
async def test_failed_live_send_detaches_socket_and_allows_capacity_eviction():
    send_enabled = threading.Event()
    send_attempted = threading.Event()

    class GatedLiveBridge(FakeBridge):
        def __init__(self):
            super().__init__([])
            self._sent = False

        def read(self, _timeout):
            if send_enabled.is_set() and not self._sent:
                self._sent = True
                return b"live output"
            return b""

    class BrokenLiveWS(FakeWS):
        async def send_bytes(self, _data):
            send_attempted.set()
            raise RuntimeError("peer disconnected during live send")

    reg = make_registry(max_sessions=1)
    old_bridge = GatedLiveBridge()
    session, _ = await reg.attach_or_spawn("first", spawn=lambda: old_bridge)
    await session.attach(BrokenLiveWS())
    send_enabled.set()
    assert await asyncio.to_thread(send_attempted.wait, 2)

    for _ in range(100):
        if not session.attached:
            break
        await asyncio.sleep(0.01)

    assert session.attached is False
    assert session.last_detached_at is not None
    replacement, created = await reg.attach_or_spawn(
        "second", spawn=lambda: FakeBridge([b"", b""])
    )
    assert created is True
    assert replacement.key == "second"
    assert old_bridge.closed is True
    await reg.close_all()


@pytest.mark.asyncio
async def test_dead_child_is_closed_and_replaced_for_same_key():
    reg = make_registry()
    old_bridge = FakeBridge([None])
    old_session, _ = await reg.attach_or_spawn("tok", spawn=lambda: old_bridge)
    await asyncio.sleep(0.02)

    new_session, created = await reg.attach_or_spawn(
        "tok", spawn=lambda: FakeBridge([b"", b""])
    )

    assert created is True
    assert new_session is not old_session
    assert old_bridge.closed is True
    await reg.close_all()


@pytest.mark.asyncio
async def test_idle_eviction_closes_oldest_handle_before_new_spawn():
    reg = make_registry(max_sessions=2)
    first = FakeBridge([b"", b""])
    second = FakeBridge([b"", b""])
    first_session, _ = await reg.attach_or_spawn("first", spawn=lambda: first)
    second_session, _ = await reg.attach_or_spawn("second", spawn=lambda: second)
    first_ws, second_ws = FakeWS(), FakeWS()
    await first_session.attach(first_ws)
    await second_session.attach(second_ws)
    first_session.detach(first_ws)
    second_session.detach(second_ws)
    first_session.last_detached_at = time.monotonic() - 2
    second_session.last_detached_at = time.monotonic() - 1

    await reg.attach_or_spawn("third", spawn=lambda: FakeBridge([b"", b""]))

    assert first.closed is True
    assert len(reg._sessions) == 2
    await reg.close_all()


def test_status_snapshot_is_bounded_and_secret_safe():
    reg = make_registry(ttl=15.0, max_sessions=3)

    snapshot = reg.snapshot()

    assert snapshot["sessions"] == {
        "total": 0,
        "alive": 0,
        "attached": 0,
        "detached": 0,
        "max": 3,
        "ttl_seconds": 15,
        "oldest_detached_age_seconds": 0,
    }
    assert "tok" not in repr(snapshot)


@pytest.mark.asyncio
async def test_cancelled_spawn_closes_bridge_created_after_cancellation():
    reg = make_registry()
    spawn_started = threading.Event()
    release_spawn = threading.Event()
    bridge = FakeBridge([b"", b""])

    def spawn():
        spawn_started.set()
        release_spawn.wait(2)
        return bridge

    task = asyncio.create_task(reg.attach_or_spawn("tok", spawn=spawn))
    assert await asyncio.to_thread(spawn_started.wait, 2)

    task.cancel()
    release_spawn.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert bridge.closed is True
    assert reg.snapshot()["sessions"]["total"] == 0
    await reg.close_all()


@pytest.mark.asyncio
async def test_cancelled_close_all_finishes_remaining_child_cleanup():
    reg = make_registry()
    close_started = threading.Event()
    release_close = threading.Event()

    class SlowCloseBridge(FakeBridge):
        def close(self):
            close_started.set()
            release_close.wait(2)
            super().close()

    first = SlowCloseBridge([b"", b""])
    second = FakeBridge([b"", b""])
    await reg.attach_or_spawn("first", spawn=lambda: first)
    await reg.attach_or_spawn("second", spawn=lambda: second)

    task = asyncio.create_task(reg.close_all())
    assert await asyncio.to_thread(close_started.wait, 2)
    task.cancel()
    release_close.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert first.closed is True
    assert second.closed is True
    with pytest.raises(RuntimeError):
        await reg.attach_or_spawn("new", spawn=lambda: FakeBridge([]))
