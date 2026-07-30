"""Regression tests for symlink-safe Docker stage2 ownership repair."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE2_HOOK = REPO_ROOT / "docker" / "stage2-hook.sh"


@pytest.fixture(scope="module")
def stage2_text() -> str:
    if not STAGE2_HOOK.exists():
        pytest.skip("docker/stage2-hook.sh not present in this checkout")
    return STAGE2_HOOK.read_text()


def _chown_hermes_tree_function(text: str) -> str:
    start = text.index("path_has_symlink_component() {")
    end = text.index("\n\nneeds_chown=false", start)
    return text[start:end]


def _run_helper(
    text: str,
    target: Path,
    log_path: Path,
    *,
    hermes_home: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("sh not available")
    hermes_home = target if hermes_home is None else hermes_home
    script = (
        "set -eu\n"
        f'HERMES_HOME="{hermes_home}"\n'
        "actual_hermes_uid=99\n"
        "actual_hermes_gid=100\n"
        f"{_chown_hermes_tree_function(text)}\n"
        f'find() {{ printf "%s\\n" "$*" >> "{log_path}"; }}\n'
        f'chown_hermes_tree "{target}"\n'
    )
    return subprocess.run([shell, "-c", script], capture_output=True, text=True)


def _run_real_find_helper(
    text: str,
    target: Path,
    log_path: Path,
    mountpoint: Path,
    tmp_path: Path,
    *,
    expected_uid: int | None = None,
    expected_gid: int | None = None,
    chown_exit: int = 0,
) -> subprocess.CompletedProcess[str]:
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("sh not available")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_mountpoint = bin_dir / "mountpoint"
    fake_mountpoint.write_text(
        '#!/bin/sh\n[ "$1" = "-q" ] && [ "$2" = "$FAKE_MOUNTPOINT" ]\n'
    )
    fake_mountpoint.chmod(0o755)
    fake_chown = bin_dir / "chown"
    fake_chown.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$CHOWN_LOG"\n'
        f"exit {chown_exit}\n"
    )
    fake_chown.chmod(0o755)
    expected_uid = os.getuid() + 1 if expected_uid is None else expected_uid
    expected_gid = os.getgid() + 1 if expected_gid is None else expected_gid
    script = (
        "set -eu\n"
        f'HERMES_HOME="{target}"\n'
        f"actual_hermes_uid={expected_uid}\n"
        f"actual_hermes_gid={expected_gid}\n"
        f"{_chown_hermes_tree_function(text)}\n"
        f'chown_hermes_tree "{target}"\n'
    )
    environment = os.environ | {
        "CHOWN_LOG": str(log_path),
        "FAKE_MOUNTPOINT": str(mountpoint),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
    }
    return subprocess.run(
        [shell, "-c", script], capture_output=True, text=True, env=environment
    )


def test_chown_helper_repairs_real_directories(stage2_text: str, tmp_path: Path) -> None:
    target = tmp_path / "home"
    target.mkdir()
    log_path = tmp_path / "chown.log"

    proc = _run_helper(stage2_text, target, log_path)

    assert proc.returncode == 0, proc.stderr
    assert log_path.read_text().splitlines() == [
        f"{target} -xdev ( ! -path {target} -type d -exec mountpoint -q {{}} ; "
        "-prune ) -o ( ( ! -uid 99 -o ! -gid 100 ) "
        "-exec chown -h 99:100 {} + )",
    ]


def test_chown_helper_prunes_nested_mountpoints(
    stage2_text: str, tmp_path: Path
) -> None:
    target = tmp_path / "home"
    nested_mount = target / "external"
    nested_mount.mkdir(parents=True)
    (nested_mount / "foreign-state").write_text("do not touch")
    (target / "managed-state").write_text("repair")
    log_path = tmp_path / "chown.log"

    proc = _run_real_find_helper(
        stage2_text, target, log_path, nested_mount, tmp_path
    )

    assert proc.returncode == 0, proc.stderr
    changed_paths = log_path.read_text()
    assert str(target / "managed-state") in changed_paths
    assert str(nested_mount) not in changed_paths


def test_chown_helper_selects_gid_only_drift(
    stage2_text: str, tmp_path: Path
) -> None:
    target = tmp_path / "home"
    target.mkdir()
    state = target / "state"
    state.write_text("repair group")
    log_path = tmp_path / "chown.log"

    proc = _run_real_find_helper(
        stage2_text,
        target,
        log_path,
        tmp_path / "not-mounted",
        tmp_path,
        expected_uid=os.getuid(),
        expected_gid=os.getgid() + 1,
    )

    assert proc.returncode == 0, proc.stderr
    assert str(state) in log_path.read_text()


def test_chown_helper_does_not_follow_final_symlink(
    stage2_text: str, tmp_path: Path
) -> None:
    target = tmp_path / "home"
    target.mkdir()
    external = tmp_path / "external-state"
    external.write_text("do not follow")
    link = target / "state-link"
    try:
        link.symlink_to(external)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are not available on this platform")
    log_path = tmp_path / "chown.log"

    proc = _run_real_find_helper(
        stage2_text, target, log_path, tmp_path / "not-mounted", tmp_path
    )

    assert proc.returncode == 0, proc.stderr
    changed_paths = log_path.read_text()
    assert str(link) in changed_paths
    assert str(external) not in changed_paths


def test_chown_helper_is_fail_soft_when_chown_fails(
    stage2_text: str, tmp_path: Path
) -> None:
    target = tmp_path / "home"
    target.mkdir()
    (target / "state").write_text("repair")

    proc = _run_real_find_helper(
        stage2_text,
        target,
        tmp_path / "chown.log",
        tmp_path / "not-mounted",
        tmp_path,
        chown_exit=1,
    )

    assert proc.returncode == 0, proc.stderr


def test_chown_helper_warm_tree_executes_no_chown(
    stage2_text: str, tmp_path: Path
) -> None:
    target = tmp_path / "home"
    target.mkdir()
    (target / "state").write_text("already owned")
    log_path = tmp_path / "chown.log"

    proc = _run_real_find_helper(
        stage2_text,
        target,
        log_path,
        tmp_path / "not-mounted",
        tmp_path,
        expected_uid=os.getuid(),
        expected_gid=os.getgid(),
    )

    assert proc.returncode == 0, proc.stderr
    assert not log_path.exists()


def test_chown_helper_refuses_symlinked_directories(stage2_text: str, tmp_path: Path) -> None:
    real_home = tmp_path / "real-home"
    real_home.mkdir()
    symlinked_home = tmp_path / "hermes-home"
    try:
        symlinked_home.symlink_to(real_home, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are not available on this platform")
    log_path = tmp_path / "chown.log"

    proc = _run_helper(stage2_text, symlinked_home, log_path)

    assert proc.returncode == 0, proc.stderr
    assert not log_path.exists()
    assert "refusing recursive chown through symlinked path" in proc.stdout


def test_chown_helper_refuses_target_under_symlinked_home(
    stage2_text: str,
    tmp_path: Path,
) -> None:
    real_home = tmp_path / "real-home"
    (real_home / "cron").mkdir(parents=True)
    linked_home = tmp_path / "linked-home"
    try:
        linked_home.symlink_to(real_home, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are not available on this platform")
    log_path = tmp_path / "chown.log"

    proc = _run_helper(
        stage2_text,
        linked_home / "cron",
        log_path,
        hermes_home=linked_home,
    )

    assert proc.returncode == 0, proc.stderr
    assert not log_path.exists(), "must not chown through a symlinked HERMES_HOME"
    assert "refusing recursive chown through symlinked path" in proc.stdout


def test_stage2_uses_symlink_safe_helper_for_hermes_home_trees(stage2_text: str) -> None:
    helper = _chown_hermes_tree_function(stage2_text)

    assert 'chown_hermes_tree "$HERMES_HOME/$sub"' in stage2_text
    assert 'chown_hermes_tree "$HERMES_HOME/profiles"' in stage2_text
    assert 'chown_hermes_tree "$HERMES_HOME/cron"' in stage2_text
    assert 'find "$target" -xdev' in helper
    assert '! -path "$target" -type d -exec mountpoint -q {} \\; -prune' in helper
    assert '! -uid "$actual_hermes_uid" -o ! -gid "$actual_hermes_gid"' in helper
    assert '-exec chown -h "$actual_hermes_uid:$actual_hermes_gid" {} +' in helper
    assert "chown -R" not in helper
    assert 'chown -R hermes:hermes "$HERMES_HOME/$sub"' not in stage2_text
    assert 'chown -R hermes:hermes "$HERMES_HOME/profiles"' not in stage2_text
    assert 'chown -R hermes:hermes "$HERMES_HOME/cron"' not in stage2_text
    assert "xz-utils util-linux" in (REPO_ROOT / "Dockerfile").read_text()


def test_stage2_skips_top_level_chown_for_symlinked_hermes_home(
    stage2_text: str,
) -> None:
    assert 'refuse_symlinked_path "chown" "$HERMES_HOME"' in stage2_text
