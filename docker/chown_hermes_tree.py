"""Selective, mount-safe ownership repair for Hermes persistent trees."""
from __future__ import annotations

import argparse
import os
import re
import stat
from collections.abc import Iterable
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


def _open_matching_entry(
    name: str,
    entry_stat: os.stat_result,
    *,
    dir_fd: int,
    flags: int,
) -> tuple[int, os.stat_result]:
    descriptor = os.open(name, flags, dir_fd=dir_fd)
    try:
        opened_stat = os.fstat(descriptor)
        if (entry_stat.st_dev, entry_stat.st_ino) != (
            opened_stat.st_dev,
            opened_stat.st_ino,
        ):
            raise OSError(f"ownership target changed during traversal: {name}")
        return descriptor, opened_stat
    except BaseException:
        os.close(descriptor)
        raise


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


def _open_anchored_parent(target: str) -> tuple[int, str]:
    parts = Path(target).parts
    if len(parts) < 2:
        raise ValueError("ownership path must name an entry below root")
    descriptor = os.open("/", _DIRECTORY_FLAGS)
    try:
        for component in parts[1:-1]:
            next_descriptor = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, parts[-1]


def repair_path(target: Path, uid: int, gid: int, *, mode: int | None = None) -> None:
    """Repair one directory or regular file through an anchored descriptor."""
    target_text = os.path.abspath(os.fspath(target))
    parent_fd, name = _open_anchored_parent(target_text)
    descriptor: int | None = None
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(before.st_mode) and not stat.S_ISREG(before.st_mode):
            raise OSError(
                f"ownership target is not a directory or regular file: {target_text}"
            )
        descriptor = os.open(name, _FILE_FLAGS, dir_fd=parent_fd)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise OSError(f"ownership target changed during resolution: {target_text}")
        if not stat.S_ISDIR(after.st_mode) and not stat.S_ISREG(after.st_mode):
            raise OSError(f"ownership target changed type during resolution: {target_text}")
        if after.st_uid != uid or after.st_gid != gid:
            os.fchown(descriptor, uid, gid)
        if mode is not None:
            os.fchmod(descriptor, mode)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def repair_tree(
    target: Path,
    uid: int,
    gid: int,
    *,
    mountinfo_path: Path = Path("/proc/self/mountinfo"),
) -> None:
    """Repair mismatched entries through anchored directory descriptors."""
    target_text = os.path.abspath(os.fspath(target))
    nested_mounts = _mountpoints(mountinfo_path)
    root_descriptor = _open_anchored_root(target_text)
    open_descriptors = {root_descriptor}
    try:
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
                    try:
                        entry_stat = os.stat(
                            name, dir_fd=directory_fd, follow_symlinks=False
                        )
                    except FileNotFoundError:
                        continue
                    if full_path in nested_mounts or entry_stat.st_dev != root_device:
                        continue
                    if not stat.S_ISDIR(entry_stat.st_mode):
                        if not stat.S_ISREG(entry_stat.st_mode):
                            continue
                        try:
                            leaf_fd, leaf_stat = _open_matching_entry(
                                name,
                                entry_stat,
                                dir_fd=directory_fd,
                                flags=_FILE_FLAGS,
                            )
                        except FileNotFoundError:
                            continue
                        try:
                            if leaf_stat.st_dev != root_device:
                                continue
                            if leaf_stat.st_uid != uid or leaf_stat.st_gid != gid:
                                os.fchown(leaf_fd, uid, gid)
                        finally:
                            os.close(leaf_fd)
                        continue

                    try:
                        child_fd, child_stat = _open_matching_entry(
                            name,
                            entry_stat,
                            dir_fd=directory_fd,
                            flags=_DIRECTORY_FLAGS,
                        )
                    except FileNotFoundError:
                        continue
                    open_descriptors.add(child_fd)
                    if child_stat.st_dev != root_device:
                        os.close(child_fd)
                        open_descriptors.remove(child_fd)
                        continue
                    if child_stat.st_uid != uid or child_stat.st_gid != gid:
                        os.fchown(child_fd, uid, gid)
                    pending.append((child_fd, full_path))
            finally:
                os.close(directory_fd)
                open_descriptors.remove(directory_fd)
    finally:
        for descriptor in open_descriptors:
            os.close(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--mode", type=lambda value: int(value, 8))
    parser.add_argument("target", type=Path)
    parser.add_argument("uid", type=int)
    parser.add_argument("gid", type=int)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.uid < 0 or args.gid < 0:
        raise ValueError("UID and GID must be non-negative")
    if args.mode is not None and not args.single:
        raise ValueError("--mode requires --single")
    if args.mode is not None and not 0 <= args.mode <= 0o7777:
        raise ValueError("mode must be between 0000 and 7777")
    if args.single:
        repair_path(args.target, args.uid, args.gid, mode=args.mode)
    else:
        repair_tree(args.target, args.uid, args.gid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
