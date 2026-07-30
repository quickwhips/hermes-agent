"""Behavioral tests for descriptor-relative persistent-tree ownership repair."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "docker" / "chown_hermes_tree.py"


def _load_helper() -> ModuleType:
    spec = importlib.util.spec_from_file_location("stage2_chown_hermes_tree", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _encode_mount_path(path: Path) -> str:
    return (
        str(path)
        .replace("\\", r"\134")
        .replace(" ", r"\040")
        .replace("\t", r"\011")
        .replace("\n", r"\012")
    )


def _mountinfo(path: Path, *mountpoints: Path) -> Path:
    lines = [
        f"{number} 1 0:{number} / {_encode_mount_path(point)} rw - tmpfs tmpfs rw\n"
        for number, point in enumerate(mountpoints, start=2)
    ]
    path.write_text("".join(lines))
    return path


def _record_chown(monkeypatch: pytest.MonkeyPatch, module: ModuleType) -> list[tuple[object, int | None, bool]]:
    calls: list[tuple[object, int | None, bool]] = []

    def fake_chown(
        path: object,
        _uid: int,
        _gid: int,
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        calls.append((path, dir_fd, follow_symlinks))

    monkeypatch.setattr(module.os, "chown", fake_chown)
    return calls


def test_repair_tree_prunes_nested_mount_root_and_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_helper()
    target = tmp_path / "managed home"
    nested_mount = target / "external"
    nested_mount.mkdir(parents=True)
    (nested_mount / "foreign-state").write_text("do not touch")
    (target / "managed-state").write_text("repair")
    mountinfo = _mountinfo(tmp_path / "mountinfo", target, nested_mount)
    calls = _record_chown(monkeypatch, module)

    module.repair_tree(
        target,
        os.getuid() + 1,
        os.getgid() + 1,
        mountinfo_path=mountinfo,
    )

    changed = {str(path) for path, _dir_fd, _follow in calls}
    assert "managed-state" in changed
    assert "external" not in changed
    assert "foreign-state" not in changed
    assert all(follow is False for _path, _dir_fd, follow in calls)


def test_repair_tree_selects_gid_only_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_helper()
    target = tmp_path / "home"
    target.mkdir()
    (target / "state").write_text("repair group")
    calls = _record_chown(monkeypatch, module)

    module.repair_tree(
        target,
        os.getuid(),
        os.getgid() + 1,
        mountinfo_path=_mountinfo(tmp_path / "mountinfo", target),
    )

    state_calls = [call for call in calls if call[0] == "state"]
    assert len(state_calls) == 1
    assert state_calls[0][1] is not None
    assert state_calls[0][2] is False


def test_repair_tree_chowns_final_symlink_without_following_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_helper()
    target = tmp_path / "home"
    target.mkdir()
    external = tmp_path / "external-state"
    external.write_text("do not follow")
    link = target / "state-link"
    try:
        link.symlink_to(external)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are not available on this platform")
    calls = _record_chown(monkeypatch, module)

    module.repair_tree(
        target,
        os.getuid() + 1,
        os.getgid() + 1,
        mountinfo_path=_mountinfo(tmp_path / "mountinfo", target),
    )

    changed = {str(path) for path, _dir_fd, _follow in calls}
    assert "state-link" in changed
    assert str(external) not in changed
    assert all(follow is False for _path, _dir_fd, follow in calls)


def test_repair_tree_warm_tree_executes_no_chown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_helper()
    target = tmp_path / "home"
    target.mkdir()
    (target / "state").write_text("already owned")
    calls = _record_chown(monkeypatch, module)

    module.repair_tree(
        target,
        os.getuid(),
        os.getgid(),
        mountinfo_path=_mountinfo(tmp_path / "mountinfo", target),
    )

    assert calls == []


def test_repair_tree_rejects_malformed_mountinfo_before_chown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_helper()
    target = tmp_path / "home"
    target.mkdir()
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text("malformed\n")
    calls = _record_chown(monkeypatch, module)

    with pytest.raises(ValueError, match="malformed mountinfo"):
        module.repair_tree(
            target,
            os.getuid() + 1,
            os.getgid() + 1,
            mountinfo_path=mountinfo,
        )

    assert calls == []


def test_helper_has_no_per_entry_process_spawn() -> None:
    source = HELPER.read_text()

    assert "subprocess" not in source
    assert "mountpoint -q" not in source
    assert "os.system" not in source
    assert "os.fwalk" in source
    assert "dir_fd=directory_fd" in source
    assert "follow_symlinks=False" in source
