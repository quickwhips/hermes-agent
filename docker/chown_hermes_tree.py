"""Selective, mount-safe ownership repair for Hermes persistent trees."""
from __future__ import annotations

import argparse
import os
import re
import stat
from collections.abc import Iterable
from pathlib import Path

_MOUNT_ESCAPE = re.compile(r"\\(040|011|012|134)")
_MOUNT_DEVICE = re.compile(r"[0-9]+:[0-9]+")
_MOUNT_DECODE = {"040": " ", "011": "\t", "012": "\n", "134": "\\"}
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


def _decode_mount_path(value: str) -> str:
    if re.search(r"\\(?!040|011|012|134)", value):
        raise ValueError("malformed mountinfo path escape")
    return _MOUNT_ESCAPE.sub(lambda match: _MOUNT_DECODE[match.group(1)], value)


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
                or not fields[0].isdigit()
                or not fields[1].isdigit()
                or _MOUNT_DEVICE.fullmatch(fields[2]) is None
                or not fields[3].startswith("/")
                or not fields[4].startswith("/")
            ):
                raise ValueError("malformed mountinfo record")
            separator = fields.index("-")
            if separator < 6 or len(fields) != separator + 4:
                raise ValueError("malformed mountinfo record")
            point = _decode_mount_path(fields[4])
            if not os.path.isabs(point):
                raise ValueError("malformed mountinfo record")
            points.add(point)
            records += 1
    if records == 0:
        raise ValueError("malformed mountinfo record")
    return frozenset(points)


def _repair_entry(
    name: str,
    entry_stat: os.stat_result,
    uid: int,
    gid: int,
    *,
    dir_fd: int,
) -> None:
    if entry_stat.st_uid == uid and entry_stat.st_gid == gid:
        return
    os.chown(name, uid, gid, dir_fd=dir_fd, follow_symlinks=False)


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
                        _repair_entry(name, entry_stat, uid, gid, dir_fd=directory_fd)
                        continue

                    try:
                        child_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=directory_fd)
                    except FileNotFoundError:
                        continue
                    open_descriptors.add(child_fd)
                    child_stat = os.fstat(child_fd)
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
    parser.add_argument("target", type=Path)
    parser.add_argument("uid", type=int)
    parser.add_argument("gid", type=int)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.uid < 0 or args.gid < 0:
        raise ValueError("UID and GID must be non-negative")
    repair_tree(args.target, args.uid, args.gid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
