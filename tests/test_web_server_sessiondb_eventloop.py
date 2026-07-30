from __future__ import annotations

import asyncio
import inspect
import threading

import pytest

from hermes_cli import web_server


class _OpenMarker(RuntimeError):
    pass


def _session_handler_cases():
    return [
        ("search", True, lambda: web_server.search_sessions(q="needle")),
        (
            "bulk delete",
            False,
            lambda: web_server.bulk_delete_sessions_endpoint(
                web_server.BulkDeleteSessions(ids=["session-id"])
            ),
        ),
        ("count empty", True, lambda: web_server.count_empty_sessions_endpoint()),
        ("delete empty", False, lambda: web_server.delete_empty_sessions_endpoint()),
        ("stats", True, lambda: web_server.get_session_stats()),
        ("detail", True, lambda: web_server.get_session_detail("session-id")),
        (
            "latest descendant",
            True,
            lambda: web_server.get_session_latest_descendant("session-id"),
        ),
        (
            "messages",
            True,
            lambda: web_server.get_session_messages("session-id"),
        ),
        (
            "delete",
            False,
            lambda: web_server.delete_session_endpoint("session-id"),
        ),
        (
            "rename",
            False,
            lambda: web_server.rename_session_endpoint(
                "session-id", web_server.SessionRename(title="new title")
            ),
        ),
        (
            "export",
            True,
            lambda: web_server.export_session_endpoint("session-id"),
        ),
        (
            "prune",
            False,
            lambda: web_server.prune_sessions_endpoint(web_server.SessionPrune()),
        ),
        ("usage analytics", True, lambda: web_server.get_usage_analytics()),
        ("model analytics", True, lambda: web_server.get_models_analytics()),
    ]


def test_sessiondb_profile_helper_requires_explicit_keyword_only_access_mode():
    parameter = inspect.signature(
        web_server._open_session_db_for_profile
    ).parameters["read_only"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


@pytest.mark.parametrize(
    ("name", "expected_read_only", "invoke"),
    _session_handler_cases(),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_async_sessiondb_handlers_offload_declared_access_mode(
    monkeypatch, name, expected_read_only, invoke
):
    loop_thread = threading.get_ident()
    observed: list[tuple[int, bool]] = []

    def _open(profile=None, *, read_only):
        observed.append((threading.get_ident(), read_only))
        raise _OpenMarker(name)

    monkeypatch.setattr(web_server, "_open_session_db_for_profile", _open)
    monkeypatch.setattr(web_server, "_session_db_exists_for_profile", lambda profile=None: True)

    try:
        asyncio.run(invoke())
    except (Exception, BaseExceptionGroup):
        pass

    assert observed, f"{name} did not open SessionDB"
    assert {mode for _, mode in observed} == {expected_read_only}
    assert all(thread_id != loop_thread for thread_id, _ in observed)
