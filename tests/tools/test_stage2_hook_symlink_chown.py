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


def _managed_tree_needs_chown_function(text: str) -> str:
    match = re.search(r"(managed_tree_needs_chown\(\) \{.*?\n\})", text, re.DOTALL)
    assert match is not None, "managed_tree_needs_chown helper missing"
    return match.group(1)


def _run_helper(
    text: str,
    target: Path,
    log_path: Path,
    *,
    hermes_home: Path | None = None,
    python_exit: int = 0,
) -> subprocess.CompletedProcess[str]:
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("sh not available")
    hermes_home = target if hermes_home is None else hermes_home
    install_dir = log_path.parent / "install"
    python = install_dir / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(
        f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{log_path}"\nexit {python_exit}\n'
    )
    python.chmod(0o755)
    (install_dir / "docker").mkdir()
    (install_dir / "docker" / "chown_hermes_tree.py").touch()
    script = (
        "set -eu\n"
        f'HERMES_HOME="{hermes_home}"\n'
        f'INSTALL_DIR="{install_dir}"\n'
        "actual_hermes_uid=99\n"
        "actual_hermes_gid=100\n"
        f"{_chown_hermes_tree_function(text)}\n"
        f'find() {{ printf "%s\\n" "$*" >> "{log_path}"; }}\n'
        f'chown_hermes_tree "{target}"\n'
    )
    return subprocess.run([shell, "-c", script], capture_output=True, text=True)


def test_chown_helper_repairs_real_directories(stage2_text: str, tmp_path: Path) -> None:
    target = tmp_path / "home"
    target.mkdir()
    log_path = tmp_path / "chown.log"

    proc = _run_helper(stage2_text, target, log_path)

    assert proc.returncode == 0, proc.stderr
    assert log_path.read_text().splitlines() == [
        f"{tmp_path}/install/docker/chown_hermes_tree.py {target} 99 100",
    ]
    assert "mountpoint" not in log_path.read_text()


def test_chown_helper_is_fail_soft_when_python_helper_fails(
    stage2_text: str, tmp_path: Path
) -> None:
    target = tmp_path / "home"
    target.mkdir()
    proc = _run_helper(
        stage2_text, target, tmp_path / "chown.log", python_exit=1
    )

    assert proc.returncode == 0, proc.stderr
    assert "chown" in proc.stdout
    assert "continuing" in proc.stdout


def test_gid_only_drift_activates_managed_tree_repair(
    stage2_text: str, tmp_path: Path
) -> None:
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("sh not available")
    target = tmp_path / "home"
    target.mkdir()
    script = (
        "set -eu\n"
        f"actual_hermes_uid={os.getuid()}\n"
        f"actual_hermes_gid={os.getgid() + 1}\n"
        f"{_managed_tree_needs_chown_function(stage2_text)}\n"
        f'managed_tree_needs_chown "{target}"\n'
    )

    proc = subprocess.run([shell, "-c", script], capture_output=True, text=True)

    assert proc.returncode == 0, proc.stderr


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
    assert '"$INSTALL_DIR/docker/chown_hermes_tree.py"' in helper
    assert '"$target" "$actual_hermes_uid" "$actual_hermes_gid"' in helper
    assert "mountpoint" not in helper
    assert "find " not in helper
    assert "chown -R" not in helper
    assert 'chown -R hermes:hermes "$HERMES_HOME/$sub"' not in stage2_text
    assert 'chown -R hermes:hermes "$HERMES_HOME/profiles"' not in stage2_text
    assert 'chown -R hermes:hermes "$HERMES_HOME/cron"' not in stage2_text
    assert "xz-utils util-linux" not in (REPO_ROOT / "Dockerfile").read_text()


def test_stage2_skips_top_level_chown_for_symlinked_hermes_home(
    stage2_text: str,
) -> None:
    assert 'chown_hermes_path "$HERMES_HOME"' in stage2_text


def test_stage2_routes_every_mutable_home_ownership_change_through_helper(
    stage2_text: str,
) -> None:
    assert 'chown_hermes_path "$HERMES_HOME/logs/gateways"' in stage2_text
    assert 'chown_hermes_path "$HERMES_HOME/$f"' in stage2_text
    assert 'chown_hermes_path "$HERMES_HOME/config.yaml" 640' in stage2_text
    assert "chown hermes:hermes" not in stage2_text
    assert 'chmod 640 "$HERMES_HOME/config.yaml"' not in stage2_text
