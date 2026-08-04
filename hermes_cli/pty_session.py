"""Keep-alive PTY sessions for dashboard terminals.

A PTY process outlives the WebSocket that created it: a single drain task
always reads the PTY into a bounded RingBuffer and forwards to the attached
socket when present. Reconnecting with the same opaque token replays the
buffer and resumes live. See
docs/superpowers/specs/2026-06-20-pty-keepalive-reattach-design.md.
"""
from __future__ import annotations

import asyncio
import inspect
import time
from typing import Callable, Dict, Optional, Tuple

WS_CLOSE_PROCESS_EXITED = 4410
WS_CLOSE_SUPERSEDED = 4409


class RingBuffer:
    """Keeps only the most recent ``capacity`` bytes appended to it."""

    def __init__(self, capacity: int) -> None:
        self._cap = capacity
        self._buf = bytearray()
        self._truncated = False

    def append(self, data: bytes) -> None:
        self._buf.extend(data)
        overflow = len(self._buf) - self._cap
        if overflow > 0:
            del self._buf[:overflow]
            self._truncated = True

    def snapshot(self) -> bytes:
        return bytes(self._buf)

    @property
    def truncated(self) -> bool:
        return self._truncated


_EVENT_KEYS = (
    "create",
    "reattach",
    "detach",
    "supersede",
    "process_exit",
    "idle_reap",
    "dead_reap",
    "dead_replace",
    "forced_fresh",
    "registry_full",
    "ws_failure",
    "close_all",
)

_WS_FAILURE_KEYS = (
    "pty_unavailable",
    "spawn_failed",
    "registry_full",
    "unexpected",
)


class PtySession:
    def __init__(
        self,
        key: str,
        bridge,
        *,
        buffer_cap: int,
        read_timeout: float,
        record_event: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.key = key
        self.bridge = bridge
        self.buffer = RingBuffer(buffer_cap)
        self.alive = True
        self.attached = False
        self.last_detached_at: Optional[float] = None
        self._read_timeout = read_timeout
        self._ws = None
        self._drain_task: Optional[asyncio.Task] = None
        self._record_event = record_event
        self._close_lock = asyncio.Lock()
        self._closed = False

    async def start(self) -> None:
        self._drain_task = asyncio.create_task(self._drain())

    async def _drain(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            chunk = await loop.run_in_executor(None, self.bridge.read, self._read_timeout)
            if chunk is None:                       # EOF — the agent process exited
                self.alive = False
                self._record("process_exit")
                ws = self._ws
                if ws is not None:
                    try:
                        await ws.close(code=WS_CLOSE_PROCESS_EXITED)
                    except Exception:
                        pass
                return
            if not chunk:                            # idle tick
                await asyncio.sleep(0)
                continue
            self.buffer.append(chunk)
            ws = self._ws
            if ws is not None:
                try:
                    await ws.send_bytes(chunk)
                except Exception:
                    pass                             # detached mid-send; keep buffering

    async def attach(self, ws) -> None:
        old = self._ws
        if old is not None and old is not ws:
            self._record("supersede")
            try:
                await old.close(code=WS_CLOSE_SUPERSEDED)
            except Exception:
                pass
        self._ws = ws
        self.attached = True
        self.last_detached_at = None
        snap = self.buffer.snapshot()
        if snap:
            await ws.send_bytes(snap)

    def detach(self, ws) -> None:
        # Only the currently-attached socket may mark the session detached.
        # A superseded socket's handler also calls detach on its way out
        # (its ``finally`` runs after the new tab attached); flipping
        # ``attached`` then would make a session with a live viewer look
        # idle and reapable.
        if self._ws is not ws:
            return
        self._ws = None
        self.attached = False
        self.last_detached_at = time.monotonic()

    def _record(self, event: str) -> None:
        if self._record_event is not None:
            self._record_event(event)

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self.alive = False
            if self._drain_task is not None:
                self._drain_task.cancel()
                try:
                    await self._drain_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
            try:
                # bridge.close() joins the child — blocking; keep it off the
                # event loop (#53227).
                await asyncio.to_thread(self.bridge.close)
            except Exception:
                pass


class RegistryFull(Exception):
    pass


async def run_reaper(registry: "PtySessionRegistry", *, interval: float = 60.0) -> None:
    """Periodically reap idle/dead keep-alive sessions. Cancelled on shutdown."""
    while True:
        await asyncio.sleep(interval)
        try:
            await registry.reap_idle()
        except asyncio.CancelledError:
            raise
        except Exception:
            pass


class PtySessionRegistry:
    def __init__(self, *, ttl: float, max_sessions: int,
                 buffer_cap: int, read_timeout: float) -> None:
        self._ttl = ttl
        self._max = max_sessions
        self._buffer_cap = buffer_cap
        self._read_timeout = read_timeout
        self._sessions: Dict[str, PtySession] = {}
        self._lock = asyncio.Lock()
        self._closed = False
        self._events = {key: 0 for key in _EVENT_KEYS}
        self._ws_failures = {key: 0 for key in _WS_FAILURE_KEYS}

    async def attach_or_spawn(self, key: str, *, spawn: Callable[[], object]
                              ) -> Tuple[PtySession, bool]:
        async with self._lock:
            self._raise_if_closed()
            existing = self._sessions.get(key)
            if existing is not None and existing.alive:
                self._record("reattach")
                return existing, False
            if existing is not None:                       # dead remnant
                self._record("dead_replace")
                await existing.close()
                self._sessions.pop(key, None)
            await self._reap_idle_locked()
            if len(self._sessions) >= self._max:
                await self._reap_one_idle_or_raise_locked()
            session = await self._spawn_session_locked(key, spawn)
            self._sessions[key] = session
            self._record("create")
            return session, True

    async def rotate(self, key: str, *, spawn: Callable[[], object]) -> Tuple[PtySession, bool]:
        async with self._lock:
            self._raise_if_closed()
            old = self._sessions.pop(key, None)
            if old is not None:
                self._record("forced_fresh")
                await old.close()
            await self._reap_idle_locked()
            if len(self._sessions) >= self._max:
                await self._reap_one_idle_or_raise_locked()
            session = await self._spawn_session_locked(key, spawn)
            self._sessions[key] = session
            self._record("create")
            return session, True

    def detach(self, key: str, ws) -> None:
        s = self._sessions.get(key)
        if s is not None:
            was_attached = s.attached and s._ws is ws
            s.detach(ws)
            if was_attached:
                self._record("detach")

    async def reap_idle(self, now: Optional[float] = None) -> None:
        async with self._lock:
            await self._reap_idle_locked(now=now)

    async def _reap_idle_locked(self, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        doomed = []
        for key, s in self._sessions.items():
            if not s.alive:
                doomed.append((key, "dead_reap"))
            elif (
                not s.attached
                and s.last_detached_at is not None
                and (now - s.last_detached_at) >= self._ttl
            ):
                doomed.append((key, "idle_reap"))
        for key, event in doomed:
            session = self._sessions.pop(key, None)
            if session is not None:
                self._record(event)
                await session.close()

    async def _reap_one_idle_or_raise_locked(self) -> None:
        idle = [s for s in self._sessions.values()
                if not s.attached and s.last_detached_at is not None]
        if not idle:
            self._record("registry_full")
            raise RegistryFull()
        oldest = min(idle, key=lambda s: s.last_detached_at or 0.0)
        self._sessions.pop(oldest.key, None)
        self._record("registry_full")
        await oldest.close()

    async def close_all(self) -> None:
        async with self._lock:
            self._closed = True
            sessions = list(self._sessions.values())
            self._sessions.clear()
            self._record("close_all")
        cleanup = asyncio.create_task(self._close_sessions(sessions))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            # Shutdown cancellation must not orphan children that have already
            # been removed from the registry. Finish the bounded cleanup, then
            # preserve the caller's cancellation signal.
            await cleanup
            raise

    async def _close_sessions(self, sessions) -> None:
        for session in sessions:
            await session.close()

    def record_ws_failure(self, outcome: str) -> None:
        safe = outcome if outcome in self._ws_failures else "unexpected"
        self._ws_failures[safe] += 1
        self._record("ws_failure")

    def snapshot(self, now: Optional[float] = None) -> Dict[str, object]:
        now = time.monotonic() if now is None else now
        sessions = list(self._sessions.values())
        detached_ages = [
            max(0.0, now - s.last_detached_at)
            for s in sessions
            if not s.attached and s.last_detached_at is not None
        ]
        oldest_age = min(max(detached_ages, default=0.0), self._ttl)
        return {
            "status": "ok",
            "sessions": {
                "total": len(sessions),
                "alive": sum(1 for s in sessions if s.alive),
                "attached": sum(1 for s in sessions if s.attached),
                "detached": sum(1 for s in sessions if not s.attached),
                "max": self._max,
                "ttl_seconds": int(self._ttl),
                "oldest_detached_age_seconds": int(oldest_age),
            },
            "events": dict(self._events),
            "websocket_failures": dict(self._ws_failures),
        }

    async def _spawn_session_locked(self, key: str, spawn: Callable[[], object]) -> PtySession:
        bridge = None
        spawn_task = asyncio.ensure_future(self._call_spawn(spawn))
        try:
            bridge = await asyncio.shield(spawn_task)
            session = PtySession(
                key,
                bridge,
                buffer_cap=self._buffer_cap,
                read_timeout=self._read_timeout,
                record_event=self._record,
            )
            await session.start()
            return session
        except asyncio.CancelledError:
            try:
                bridge = await asyncio.shield(spawn_task)
            except Exception:
                bridge = None
            if bridge is not None:
                await self._close_bridge(bridge)
            raise
        except Exception:
            if bridge is not None:
                await self._close_bridge(bridge)
            raise

    async def _call_spawn(self, spawn: Callable[[], object]) -> object:
        if asyncio.iscoroutinefunction(spawn):
            return await spawn()
        # PTY spawn does blocking fork/exec work — keep it off the event
        # loop (#53227).
        result = await asyncio.to_thread(spawn)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _close_bridge(self, bridge) -> None:
        try:
            await asyncio.to_thread(bridge.close)
        except Exception:
            pass

    def _record(self, event: str) -> None:
        if event in self._events:
            self._events[event] += 1

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError("PTY session registry is closed")