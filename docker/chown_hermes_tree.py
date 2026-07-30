"""Selective, mount-safe ownership repair for Hermes persistent trees."""
from __future__ import annotations

import argparse
import os
import re
import stat
from collections.abc import Iterable
from pathlib import Path

_MOUNT_ESCAPE = re.compile(r"\\(040|011|012|134)")
_MOUNT_CHARS = {"040": " ", "011": "\t", "012": "\n", "134": "\\"}


def _decode_mount_path(value: str) -> str:
    return _MOUNT_ESCAPE.sub(lambda match: _MOUNT_CHARS[match.group(1)], value)


def _mountpoints(path: Path) -> frozenset[str]:
    points: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            fields = line.rstrip("\n").split(" ")
            if len(fields) < 6:
                raise ValueError(f"malformed mountinfo line {number}")
            points.add(_decode_mount_path(fields[4]))
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


def repair_tree(
    target: Path,
    uid: int,
    gid: int,
    *,
    mountinfo_path: Path = Path("/proc/self/mountinfo"),
) -> None:
    """Repair mismatched entries without following symlinks or nested mounts."""
    target_text = os.path.abspath(os.fspath(target))
    root_stat = os.lstat(target_text)
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("ownership repair target must be a directory")
    root_device = root_stat.st_dev
    nested_mounts = _mountpoints(mountinfo_path).difference({target_text})

    if root_stat.st_uid != uid or root_stat.st_gid != gid:
        os.chown(target_text, uid, gid, follow_symlinks=False)

    for directory, names, files, directory_fd in os.fwalk(
        target_text, topdown=True, follow_symlinks=False
    ):
        for name in tuple(names):
            full_path = os.path.join(directory, name)
            try:
                entry_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                names.remove(name)
                continue
            if full_path in nested_mounts or entry_stat.st_dev != root_device:
                names.remove(name)
                continue
            _repair_entry(name, entry_stat, uid, gid, dir_fd=directory_fd)

        for name in files:
            try:
                entry_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if entry_stat.st_dev != root_device:
                continue
            _repair_entry(name, entry_stat, uid, gid, dir_fd=directory_fd)


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
