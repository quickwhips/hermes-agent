from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STATUS_HELPER = REPO_ROOT / "docker" / "init_status.sh"


def _run_shell(script: str, *args: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["dash", "-c", script, "test", *(str(arg) for arg in args)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_status_commit_is_current_boot_bound_and_owner_only(tmp_path: Path) -> None:
    marker = tmp_path / "status"
    result = _run_shell(
        f'. "{STATUS_HELPER}"; token=$(hermes_current_boot_token); '
        'hermes_mark_status "$1" ready "$token"; '
        'hermes_status_is_ready "$1" "$token"',
        marker,
    )
    assert result.returncode == 0, result.stderr
    assert marker.read_text().startswith("ready:pid:[")
    assert stat.S_IMODE(marker.stat().st_mode) == 0o600


def test_stale_boot_token_does_not_authorize(tmp_path: Path) -> None:
    marker = tmp_path / "status"
    marker.write_text("ready:pid:[stale]:0\n")
    result = _run_shell(
        f'. "{STATUS_HELPER}"; token=$(hermes_current_boot_token); '
        '! hermes_status_is_ready "$1" "$token"',
        marker,
    )
    assert result.returncode == 0, result.stderr


def test_status_commit_replaces_symlink_without_writing_target(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.write_text("unchanged")
    marker = tmp_path / "status"
    marker.symlink_to(external)
    result = _run_shell(
        f'. "{STATUS_HELPER}"; token=$(hermes_current_boot_token); '
        'hermes_mark_status "$1" failed "$token"',
        marker,
    )
    assert result.returncode == 0, result.stderr
    assert not marker.is_symlink()
    assert marker.read_text().startswith("failed:pid:[")
    assert external.read_text() == "unchanged"


@pytest.mark.parametrize("state", ["", "READY", "ready\nforged"])
def test_invalid_status_state_is_rejected(tmp_path: Path, state: str) -> None:
    marker = tmp_path / "status"
    env = os.environ | {"TEST_STATE": state}
    result = subprocess.run(
        [
            "dash",
            "-c",
            f'. "{STATUS_HELPER}"; token=$(hermes_current_boot_token); '
            '! hermes_mark_status "$1" "$TEST_STATE" "$token"',
            "test",
            str(marker),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not marker.exists()
