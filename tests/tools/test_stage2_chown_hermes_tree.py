"""Behavioral tests for descriptor-relative persistent-tree ownership repair."""
from __future__ import annotations

import importlib.util
import errno
import os
import subprocess
import sys
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


@pytest.mark.parametrize(
    ("value", "safe_roots"),
    [
        ("/", "/"),
        ("/opt/data/..", "/opt"),
        ("/opt/hermes", "/opt/hermes"),
        ("/opt/hermes/runtime", "/opt/hermes/runtime"),
        ("/etc/hermes", "/etc/hermes"),
        ("/tmp/hermes", "/opt/data"),
        ("relative/hermes", "relative/hermes"),
        ("/opt/data", "/opt/data/.."),
        ("//opt/data", "//opt/data"),
        ("/tmp/hermes\nstate", "/tmp/hermes\nstate"),
    ],
)
def test_root_policy_rejects_noncanonical_or_untrusted_authority(
    value: str, safe_roots: str
) -> None:
    module = _load_helper()

    with pytest.raises(ValueError, match="HERMES_HOME"):
        module.validate_root_policy(value, safe_roots)


@pytest.mark.parametrize(
    ("value", "safe_roots"),
    [
        ("/opt/data", "/opt/data"),
        ("/home/hermes/.hermes", "/home/hermes/.hermes"),
        ("/config/hermes", "/workspace:/config/hermes"),
        ("/tmp/hermes-test", "/tmp/hermes-test:/workspace"),
    ],
)
def test_root_policy_accepts_explicit_canonical_safe_root(
    value: str, safe_roots: str
) -> None:
    module = _load_helper()

    assert module.validate_root_policy(value, safe_roots) == Path(value)


def test_prepare_root_cli_fails_before_ownership_repair() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "--prepare-root",
            "/opt/hermes",
            "--safe-roots",
            "/opt/hermes",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "HERMES_HOME" in result.stderr


def test_prepare_root_rejects_symlink_without_writing_through_it(tmp_path: Path) -> None:
    module = _load_helper()
    external = tmp_path / "external"
    external.mkdir()
    configured = tmp_path / "configured"
    configured.symlink_to(external, target_is_directory=True)

    with pytest.raises(OSError):
        module.prepare_root(str(configured), str(configured))

    assert list(external.iterdir()) == []


def test_prepare_root_creates_missing_components_by_descriptor(tmp_path: Path) -> None:
    module = _load_helper()
    configured = tmp_path / "missing" / "hermes"

    prepared = module.prepare_root(str(configured), str(configured))

    assert prepared == configured
    assert configured.is_dir()


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
        resolved = Path(os.readlink(f"/proc/self/fd/{descriptor}")).name
        calls.append((resolved, descriptor, False))

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


def test_repair_tree_skips_final_symlink_and_target(
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
    assert "state-link" not in changed
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


def test_repair_tree_uses_in_process_descriptor_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_helper()
    target = tmp_path / "home"
    target.mkdir()
    (target / "state").write_text("repair")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("ownership traversal must not spawn or shell out per entry")

    monkeypatch.setattr(os, "system", forbidden)
    monkeypatch.setattr(os, "fwalk", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)

    module.repair_tree(
        target,
        os.getuid(),
        os.getgid(),
        mountinfo_path=_mountinfo(tmp_path / "mountinfo", target),
    )


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
        "١ 2 0:1 / /fake rw - ext4 /dev/root rw\n",
        "01 2 0:1 / /fake rw - ext4 /dev/root rw\n",
        "1 2 00:01 / /fake rw - ext4 /dev/root rw\n",
        "\n",
    ],
    ids=[
        "missing_separator",
        "missing_post_fields",
        "double_space",
        "unicode_mount_id",
        "leading_zero_mount_id",
        "leading_zero_device",
        "blank",
    ],
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

    module.repair_path(directory, os.getuid() + 1, os.getgid() + 1, root=tmp_path)
    module.repair_path(state_file, os.getuid() + 1, os.getgid() + 1, root=tmp_path)

    assert len(calls) == 2
    assert {path for path, _fd, _follow in calls} == {"gateways", "state.db"}


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
        module.repair_path(
            linked_parent / "state.db", os.getuid() + 1, os.getgid(), root=tmp_path
        )
    with pytest.raises(OSError):
        module.repair_path(linked_file, os.getuid() + 1, os.getgid(), root=tmp_path)

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

    module.repair_path(target, os.getuid(), os.getgid(), root=tmp_path, mode=0o640)

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


def test_open_beneath_rejects_symlink_redirection(
    tmp_path: Path,
) -> None:
    module = _load_helper()
    managed = tmp_path / "managed"
    managed.mkdir()
    foreign = tmp_path / "foreign"
    foreign.write_text("foreign")
    (managed / "redirect").symlink_to(foreign)
    parent_fd = os.open(managed, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))

    try:
        with pytest.raises(OSError):
            module._open_beneath(parent_fd, "redirect", module._FILE_FLAGS)
    finally:
        os.close(parent_fd)


def test_repair_tree_contains_regular_file_swap_to_external_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_helper()
    target = tmp_path / "home"
    target.mkdir()
    leaf = target / "state"
    leaf.write_text("original")
    foreign = tmp_path / "foreign"
    foreign.write_text("foreign")
    original_open = module._open_beneath
    swapped = False

    def swapping_open(dir_fd: int, name: str, flags: int) -> int:
        nonlocal swapped
        if name == "state" and not swapped:
            swapped = True
            leaf.unlink()
            leaf.symlink_to(foreign)
        return original_open(dir_fd, name, flags)

    monkeypatch.setattr(module, "_open_beneath", swapping_open)
    calls = _record_chown(monkeypatch, module)

    module.repair_tree(
        target,
        os.getuid(),
        os.getgid(),
        mountinfo_path=_mountinfo(tmp_path / "mountinfo", target),
    )

    assert swapped is True
    assert calls == []
    assert foreign.read_text() == "foreign"


def test_repair_tree_contains_directory_swap_to_external_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_helper()
    target = tmp_path / "home"
    child = target / "child"
    child.mkdir(parents=True)
    (child / "original").write_text("original")
    stale = target / "stale-child"
    foreign = tmp_path / "foreign-directory"
    foreign.mkdir()
    (foreign / "foreign").write_text("foreign")
    original_open = module._open_beneath
    swapped = False

    def swapping_open(dir_fd: int, name: str, flags: int) -> int:
        nonlocal swapped
        if name == "child" and not swapped:
            swapped = True
            child.rename(stale)
            child.symlink_to(foreign, target_is_directory=True)
        return original_open(dir_fd, name, flags)

    monkeypatch.setattr(module, "_open_beneath", swapping_open)
    calls = _record_chown(monkeypatch, module)

    module.repair_tree(
        target,
        os.getuid(),
        os.getgid(),
        mountinfo_path=_mountinfo(tmp_path / "mountinfo", target),
    )

    assert swapped is True
    assert calls == []
    assert (foreign / "foreign").read_text() == "foreign"


def test_repair_path_allows_configured_root_mount_without_crossing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The configured home mount is an authority boundary, not a nested mount."""
    module = _load_helper()
    home = tmp_path / "home"
    home.mkdir()
    calls = _record_chown(monkeypatch, module)

    module.repair_path(home, os.getuid() + 1, os.getgid() + 1, root=home)

    assert [path for path, _fd, _follow in calls] == ["home"]


def test_repair_path_refuses_nested_mount_target_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_helper()
    home = tmp_path / "home"
    nested = home / "foreign-file"
    nested.parent.mkdir()
    nested.write_text("do not touch")
    calls = _record_chown(monkeypatch, module)
    monkeypatch.setattr(
        module,
        "_open_beneath",
        lambda _dir_fd, name, _flags: (_ for _ in ()).throw(
            OSError(errno.EXDEV, "nested mount", name)
        ),
    )

    with pytest.raises(OSError) as error:
        module.repair_path(
            nested,
            os.getuid() + 1,
            os.getgid() + 1,
            root=home,
        )

    assert error.value.errno == errno.EXDEV
    assert calls == []


def test_repair_tree_refuses_a_nested_mount_as_its_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_helper()
    home = tmp_path / "home"
    nested = home / "foreign"
    nested.mkdir(parents=True)
    (nested / "foreign-state").write_text("do not touch")
    calls = _record_chown(monkeypatch, module)

    with pytest.raises(OSError):
        module.repair_tree(
            nested,
            os.getuid() + 1,
            os.getgid() + 1,
            root=home,
            mountinfo_path=_mountinfo(tmp_path / "mountinfo", home, nested),
        )

    assert calls == []


def test_repair_rejects_target_outside_configured_root_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_helper()
    home = tmp_path / "home"
    outsider = tmp_path / "home-other"
    home.mkdir()
    outsider.mkdir()
    calls = _record_chown(monkeypatch, module)

    with pytest.raises(ValueError, match="configured root"):
        module.repair_path(outsider, os.getuid() + 1, os.getgid() + 1, root=home)

    assert calls == []


def test_repair_path_closes_root_and_target_descriptors_after_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_helper()
    home = tmp_path / "home"
    target = home / "state"
    home.mkdir()
    target.write_text("state")
    original_open = module._open_beneath
    original_close = module.os.close
    opened: list[int] = []
    closed: list[int] = []

    def tracking_open(dir_fd: int, name: str, flags: int) -> int:
        descriptor = original_open(dir_fd, name, flags)
        opened.append(descriptor)
        return descriptor

    def tracking_close(descriptor: int) -> None:
        closed.append(descriptor)
        original_close(descriptor)

    monkeypatch.setattr(module, "_open_beneath", tracking_open)
    monkeypatch.setattr(module.os, "close", tracking_close)

    module.repair_path(target, os.getuid(), os.getgid(), root=home)

    assert set(opened) <= set(closed)
