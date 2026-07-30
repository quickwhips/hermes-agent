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

    def fake_fchown(descriptor: int, _uid: int, _gid: int) -> None:
        calls.append((f"fd:{descriptor}", descriptor, False))

    monkeypatch.setattr(module.os, "chown", fake_chown)
    monkeypatch.setattr(module.os, "fchown", fake_fchown)
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


def test_repair_tree_prunes_nested_file_mount_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_helper()
    target = tmp_path / "home"
    target.mkdir()
    nested_mount = target / "external-file"
    nested_mount.write_text("do not touch")
    managed = target / "managed-state"
    managed.write_text("repair")
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
    assert "external-file" not in changed


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


@pytest.mark.parametrize(
    "record",
    [
        "1 2 0:1 / /fake rw\n",
        "1 2 0:1 / /fake rw - ext4\n",
        "1  2 0:1 / /fake rw - ext4 /dev/root rw\n",
        "\n",
    ],
    ids=["missing_separator", "missing_post_fields", "double_space", "blank"],
)
def test_structurally_malformed_mountinfo_fails_before_any_chown(
    record: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_helper()
    target = tmp_path / "home"
    target.mkdir()
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(record)
    calls = _record_chown(monkeypatch, module)

    with pytest.raises(ValueError, match="malformed mountinfo"):
        module.repair_tree(
            target,
            os.getuid() + 1,
            os.getgid() + 1,
            mountinfo_path=mountinfo,
        )
    assert calls == []


def test_intermediate_symlink_cannot_redirect_anchored_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_helper()
    foreign_parent = tmp_path / "foreign"
    target = foreign_parent / "managed"
    target.mkdir(parents=True)
    (target / "victim").write_text("do not touch")
    link_parent = tmp_path / "link-home"
    link_parent.symlink_to(foreign_parent, target_is_directory=True)
    redirected_target = link_parent / "managed"
    mountinfo = _mountinfo(tmp_path / "mountinfo", redirected_target)
    calls = _record_chown(monkeypatch, module)

    with pytest.raises(OSError):
        module.repair_tree(
            redirected_target,
            os.getuid() + 1,
            os.getgid() + 1,
            mountinfo_path=mountinfo,
        )

    assert calls == []


def test_repair_path_repairs_directory_and_regular_file_by_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_helper()
    directory = tmp_path / "gateways"
    directory.mkdir()
    state_file = tmp_path / "state.db"
    state_file.write_text("state")
    calls = _record_chown(monkeypatch, module)

    module.repair_path(directory, os.getuid() + 1, os.getgid() + 1)
    module.repair_path(state_file, os.getuid() + 1, os.getgid() + 1)

    assert len(calls) == 2
    assert all(str(path).startswith("fd:") for path, _fd, _follow in calls)


def test_repair_path_rejects_intermediate_and_final_symlinks_before_chown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_helper()
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    target = real_parent / "state.db"
    target.write_text("state")
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    linked_file = tmp_path / "linked-state"
    linked_file.symlink_to(target)
    calls = _record_chown(monkeypatch, module)

    with pytest.raises(OSError):
        module.repair_path(linked_parent / "state.db", os.getuid() + 1, os.getgid())
    with pytest.raises(OSError):
        module.repair_path(linked_file, os.getuid() + 1, os.getgid())

    assert calls == []


def test_repair_path_applies_mode_through_open_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_helper()
    target = tmp_path / "config.yaml"
    target.write_text("model: test")
    modes: list[tuple[int, int]] = []
    _record_chown(monkeypatch, module)
    monkeypatch.setattr(
        module.os,
        "fchmod",
        lambda descriptor, mode: modes.append((descriptor, mode)),
    )

    module.repair_path(target, os.getuid(), os.getgid(), mode=0o640)

    assert len(modes) == 1
    assert modes[0][0] >= 0
    assert modes[0][1] == 0o640


@pytest.mark.parametrize(
    "record",
    [
        "1 2 0:1 /bad\\999 /fake rw - ext4 /dev/root rw\n",
        "1 2 0:1 / /fake rw bad_optional - ext4 /dev/root rw\n",
        "1 2 0:1 / /fake rw - ext4 /dev/\\999 rw\n",
        "1 2 0:1 / /fake rw,,nodev - ext4 /dev/root rw\n",
    ],
    ids=["bad_root_escape", "bad_optional", "bad_source_escape", "bad_options"],
)
def test_mountinfo_field_grammar_fails_before_any_chown(
    record: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_helper()
    target = tmp_path / "home"
    target.mkdir()
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(record)
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
    assert "os.fwalk" not in source
    assert "_open_anchored_root" in source
    assert "O_NOFOLLOW" in source
    assert "os.listdir(directory_fd)" in source
    assert "os.fchown" in source
    assert "dir_fd=directory_fd" in source
    assert "follow_symlinks=False" in source
