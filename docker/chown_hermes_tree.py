"""Selective, mount-safe ownership repair for Hermes persistent trees."""
from __future__ import annotations

import argparse
import ctypes
import errno
import os
import posixpath
import re
import stat
from collections.abc import Sequence
from pathlib import Path

_MOUNT_ESCAPE = re.compile(r"\\(040|011|012|134)")
_MOUNT_ID = re.compile(r"[1-9][0-9]*")
_MOUNT_DEVICE = re.compile(r"(?:0|[1-9][0-9]*):(?:0|[1-9][0-9]*)")
_MOUNT_OPTIONAL = re.compile(
    r"(?:shared|master|propagate_from):[1-9][0-9]*|unbindable"
)
_MOUNT_DECODE = {"040": " ", "011": "\t", "012": "\n", "134": "\\"}
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)
_SYS_OPENAT2 = 437
_RESOLVE_NO_XDEV = 0x01
_RESOLVE_NO_SYMLINKS = 0x04
_RESOLVE_BENEATH = 0x08
_PROTECTED_ROOTS = tuple(
    Path(value)
    for value in ("/bin", "/boot", "/dev", "/etc", "/lib", "/lib64", "/proc", "/root", "/run", "/sbin", "/sys", "/usr", "/var")
)
_INSTALL_ROOT = Path("/opt/hermes")


class _OpenHow(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint64),
        ("mode", ctypes.c_uint64),
        ("resolve", ctypes.c_uint64),
    ]


def _open_beneath(dir_fd: int, name: str, flags: int) -> int:
    """Open the current child without following links or crossing mounts."""
    if not name or name in {".", ".."} or "/" in name:
        raise ValueError("ownership entry must be one path component")
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = getattr(libc, "syscall", None)
    if syscall is None:
        raise OSError(errno.ENOSYS, "openat2 is unavailable", name)
    syscall.restype = ctypes.c_long
    how = _OpenHow(
        flags=flags,
        mode=0,
        resolve=_RESOLVE_BENEATH | _RESOLVE_NO_SYMLINKS | _RESOLVE_NO_XDEV,
    )
    result = syscall(
        ctypes.c_long(_SYS_OPENAT2),
        ctypes.c_int(dir_fd),
        ctypes.c_char_p(os.fsencode(name)),
        ctypes.byref(how),
        ctypes.c_size_t(ctypes.sizeof(how)),
    )
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), name)
    return int(result)


def _decode_mount_path(value: str) -> str:
    if re.search(r"\\(?!040|011|012|134)", value):
        raise ValueError("malformed mountinfo path escape")
    return _MOUNT_ESCAPE.sub(lambda match: _MOUNT_DECODE[match.group(1)], value)


def _validate_mount_options(value: str) -> None:
    _decode_mount_path(value)
    if not value or any(not item for item in value.split(",")):
        raise ValueError("malformed mountinfo options")
    if any(not item.isascii() or not item.isprintable() for item in value):
        raise ValueError("malformed mountinfo options")


def _mountpoints(path: Path) -> frozenset[str]:
    points: set[str] = set()
    records = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = line[:-1] if line.endswith("\n") else line
            if not record or record.strip() != record or "  " in record or "\t" in record:
                raise ValueError("malformed mountinfo record")
            fields = record.split(" ")
            if (
                len(fields) < 10
                or fields.count("-") != 1
                or _MOUNT_ID.fullmatch(fields[0]) is None
                or _MOUNT_ID.fullmatch(fields[1]) is None
                or _MOUNT_DEVICE.fullmatch(fields[2]) is None
                or not fields[3].startswith("/")
                or not fields[4].startswith("/")
            ):
                raise ValueError("malformed mountinfo record")
            separator = fields.index("-")
            if separator < 6 or len(fields) != separator + 4:
                raise ValueError("malformed mountinfo record")
            _decode_mount_path(fields[3])
            point = _decode_mount_path(fields[4])
            _validate_mount_options(fields[5])
            if any(
                _MOUNT_OPTIONAL.fullmatch(field) is None
                for field in fields[6:separator]
            ):
                raise ValueError("malformed mountinfo optional field")
            if (
                not fields[separator + 1].isascii()
                or not fields[separator + 1].isprintable()
            ):
                raise ValueError("malformed mountinfo filesystem type")
            _decode_mount_path(fields[separator + 2])
            _validate_mount_options(fields[separator + 3])
            if not os.path.isabs(point):
                raise ValueError("malformed mountinfo record")
            points.add(point)
            records += 1
    if records == 0:
        raise ValueError("malformed mountinfo record")
    return frozenset(points)


def _open_anchored_root(target: str) -> int:
    descriptor = os.open("/", _DIRECTORY_FLAGS)
    try:
        for component in Path(target).parts[1:]:
            next_descriptor = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _relative_to_root(target: Path, root: Path) -> tuple[str, str]:
    """Return absolute lexical paths after rejecting targets outside ``root``."""
    root_text = os.path.abspath(os.fspath(root))
    target_text = os.path.abspath(os.fspath(target))
    relative = os.path.relpath(target_text, root_text)
    if relative == os.pardir or relative.startswith(f"{os.pardir}{os.sep}"):
        raise ValueError("ownership target is outside configured root")
    return root_text, relative


def validate_root_policy(value: str, safe_roots: str) -> Path:
    """Validate stage2's root-owned mutation authority before any writes."""
    if (
        not value.startswith("/")
        or value.startswith("//")
        or posixpath.normpath(value) != value
        or any(not character.isprintable() for character in value)
    ):
        raise ValueError("HERMES_HOME must be an absolute canonical data-root path")
    root = Path(value)
    if root == Path("/") or root.is_relative_to(_INSTALL_ROOT) or _INSTALL_ROOT.is_relative_to(root):
        raise ValueError("HERMES_HOME overlaps a protected container path")
    if any(root == protected or root.is_relative_to(protected) for protected in _PROTECTED_ROOTS):
        raise ValueError("HERMES_HOME overlaps a protected container path")
    accepted: set[Path] = set()
    for entry in safe_roots.split(os.pathsep):
        if not entry:
            continue
        if (
            not entry.startswith("/")
            or entry.startswith("//")
            or posixpath.normpath(entry) != entry
            or any(not character.isprintable() for character in entry)
        ):
            raise ValueError("HERMES_HOME requires canonical HERMES_WRITE_SAFE_ROOT entries")
        accepted.add(Path(entry))
    if root not in accepted:
        raise ValueError("HERMES_HOME must exactly match a HERMES_WRITE_SAFE_ROOT entry")
    return root


def _runtime_can_replace_child(metadata: os.stat_result, runtime_uid: int) -> bool:
    mode = metadata.st_mode
    return bool(
        (metadata.st_uid == runtime_uid and mode & stat.S_IWUSR)
        or mode & stat.S_IWGRP
        or mode & stat.S_IWOTH
    )


def _has_access_acl(descriptor: int) -> bool:
    """Return whether an opened directory carries a POSIX access ACL."""
    try:
        os.getxattr(descriptor, "system.posix_acl_access")
    except OSError as exc:
        no_acl_errnos = {
            errno.ENODATA,
            errno.ENOTSUP,
            errno.EOPNOTSUPP,
        }
        if exc.errno in no_acl_errnos:
            return False
        raise
    return True


def _assert_stable_parent(descriptor: int, runtime_uid: int, child: Path) -> None:
    if _has_access_acl(descriptor):
        raise OSError(
            errno.EPERM,
            "HERMES_HOME has a parent with a POSIX access ACL",
            str(child),
        )
    if _runtime_can_replace_child(os.fstat(descriptor), runtime_uid):
        raise OSError(
            errno.EPERM,
            "HERMES_HOME has a runtime-writable parent",
            str(child),
        )


def prepare_root(
    value: str,
    safe_roots: str,
    *,
    runtime_uid: int,
    runtime_gid: int,
    mountinfo_path: Path = Path("/proc/self/mountinfo"),
) -> Path:
    """Create/open a root whose pathname the runtime identity cannot replace."""
    if runtime_uid < 1 or runtime_gid < 1:
        raise ValueError("runtime UID and GID must be positive")
    root = validate_root_policy(value, safe_roots)
    mountpoints = _mountpoints(mountinfo_path)
    descriptor = os.open("/", _DIRECTORY_FLAGS)
    current = Path("/")
    try:
        for component in root.parts[1:]:
            child = current / component
            if str(child) not in mountpoints:
                _assert_stable_parent(descriptor, runtime_uid, child)
            try:
                next_descriptor = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except FileNotFoundError:
                try:
                    os.mkdir(component, mode=0o755, dir_fd=descriptor)
                except FileExistsError:
                    pass
                next_descriptor = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
            current = child
    finally:
        os.close(descriptor)
    return root


def _open_target(root_fd: int, relative: str) -> int:
    """Resolve a target only beneath the already-authorized root descriptor."""
    if relative == os.curdir:
        return os.dup(root_fd)
    descriptor = root_fd
    opened: int | None = None
    try:
        components = Path(relative).parts
        for component in components[:-1]:
            opened = _open_beneath(descriptor, component, _DIRECTORY_FLAGS)
            os.close(descriptor) if descriptor != root_fd else None
            descriptor = opened
            opened = None
        opened = _open_beneath(descriptor, components[-1], _FILE_FLAGS)
        return opened
    except BaseException:
        if opened is not None:
            os.close(opened)
        raise
    finally:
        if descriptor != root_fd:
            os.close(descriptor)


def repair_path(
    target: Path, uid: int, gid: int, *, root: Path | None = None, mode: int | None = None
) -> None:
    """Repair one directory or regular file beneath the trusted root descriptor."""
    root_text, relative = _relative_to_root(target, target if root is None else root)
    root_fd = _open_anchored_root(root_text)
    descriptor: int | None = None
    try:
        descriptor = _open_target(root_fd, relative)
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode) and not stat.S_ISREG(opened.st_mode):
            raise OSError(f"ownership target changed type during resolution: {target}")
        if opened.st_uid != uid or opened.st_gid != gid:
            os.fchown(descriptor, uid, gid)
        if mode is not None:
            os.fchmod(descriptor, mode)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(root_fd)


def repair_tree(
    target: Path,
    uid: int,
    gid: int,
    *,
    root: Path | None = None,
    mountinfo_path: Path = Path("/proc/self/mountinfo"),
) -> None:
    """Repair mismatched entries beneath the trusted root descriptor."""
    root_text, relative = _relative_to_root(target, target if root is None else root)
    nested_mounts = _mountpoints(mountinfo_path)
    target_text = os.path.abspath(os.fspath(target))
    if target_text != root_text and target_text in nested_mounts:
        raise OSError(errno.EXDEV, "ownership target is a nested mount", target_text)
    root_fd = _open_anchored_root(root_text)
    root_descriptor: int | None = None
    open_descriptors: set[int] = set()
    try:
        root_descriptor = _open_target(root_fd, relative)
        open_descriptors.add(root_descriptor)
        root_stat = os.fstat(root_descriptor)
        root_device = root_stat.st_dev
        if root_stat.st_uid != uid or root_stat.st_gid != gid:
            os.fchown(root_descriptor, uid, gid)

        pending = [(root_descriptor, target_text)]
        while pending:
            directory_fd, directory = pending.pop()
            try:
                for name in os.listdir(directory_fd):
                    full_path = os.path.join(directory, name)
                    if full_path in nested_mounts:
                        continue
                    entry_fd: int | None = None
                    try:
                        try:
                            entry_fd = _open_beneath(directory_fd, name, _FILE_FLAGS)
                        except OSError as error:
                            if error.errno in {errno.ENOENT, errno.ELOOP, errno.EXDEV}:
                                continue
                            raise
                        entry_stat = os.fstat(entry_fd)
                        if entry_stat.st_dev != root_device:
                            continue
                        if stat.S_ISDIR(entry_stat.st_mode):
                            open_descriptors.add(entry_fd)
                            if entry_stat.st_uid != uid or entry_stat.st_gid != gid:
                                os.fchown(entry_fd, uid, gid)
                            pending.append((entry_fd, full_path))
                            entry_fd = None
                            continue
                        if stat.S_ISREG(entry_stat.st_mode) and (
                            entry_stat.st_uid != uid or entry_stat.st_gid != gid
                        ):
                            os.fchown(entry_fd, uid, gid)
                    finally:
                        if entry_fd is not None:
                            os.close(entry_fd)
            finally:
                os.close(directory_fd)
                open_descriptors.remove(directory_fd)
    finally:
        for descriptor in open_descriptors:
            os.close(descriptor)
        os.close(root_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--mode", type=lambda value: int(value, 8))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--prepare-root", metavar="HERMES_HOME")
    parser.add_argument("--safe-roots", metavar="HERMES_WRITE_SAFE_ROOT")
    parser.add_argument("--runtime-uid", type=int)
    parser.add_argument("--runtime-gid", type=int)
    parser.add_argument("target", nargs="?", type=Path)
    parser.add_argument("uid", nargs="?", type=int)
    parser.add_argument("gid", nargs="?", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.prepare_root is not None:
        if args.safe_roots is None or args.runtime_uid is None or args.runtime_gid is None:
            parser.error(
                "--prepare-root requires --safe-roots, --runtime-uid, and --runtime-gid"
            )
        if args.single or args.mode is not None or args.root is not None or any(
            value is not None for value in (args.target, args.uid, args.gid)
        ):
            parser.error("--prepare-root cannot be combined with ownership repair")
        prepare_root(
            args.prepare_root,
            args.safe_roots,
            runtime_uid=args.runtime_uid,
            runtime_gid=args.runtime_gid,
        )
        return 0
    if any(value is not None for value in (args.safe_roots, args.runtime_uid, args.runtime_gid)):
        parser.error("safe-root and runtime identity arguments require --prepare-root")
    if args.root is None or args.target is None or args.uid is None or args.gid is None:
        parser.error("ownership repair requires --root, target, uid, and gid")
    if args.uid < 0 or args.gid < 0:
        raise ValueError("UID and GID must be non-negative")
    if args.mode is not None and not args.single:
        raise ValueError("--mode requires --single")
    if args.mode is not None and not 0 <= args.mode <= 0o7777:
        raise ValueError("mode must be between 0000 and 7777")
    if args.single:
        repair_path(args.target, args.uid, args.gid, root=args.root, mode=args.mode)
    else:
        repair_tree(args.target, args.uid, args.gid, root=args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
