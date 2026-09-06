"""Two readings out of /proc, about ONE process, for the cast lane.

They answer "is that viewer actually receiving something from that host",
which ``Popen.poll() is None`` cannot: a RustDesk viewer parked on an
accept-or-password prompt is a live process, and round 2 called that a
landing.

NOTHING HERE OPENS A SOCKET, A WINDOW OR A CAPTURE DEVICE. It reads text
files the kernel already wrote, about a pid this app started itself, and
nothing it reads leaves the box. It cannot see a picture and there is no
path from it to one.
"""
from __future__ import annotations

import os
import socket
import struct
from typing import Optional

from jarvis.logs import get_logger

log = get_logger("procnet")

TCP_ESTABLISHED = "01"


def _hex_v4(dotted: str) -> str:
    """An IPv4 address as /proc/net/tcp spells it: little-endian hex."""
    packed = socket.inet_aton(str(dotted))
    return "%08X" % struct.unpack("<I", packed)[0]


def _socket_inodes(pid: int) -> set:
    """The socket inodes this pid holds open. A fd that vanishes between
    the listing and the readlink is simply not counted."""
    out = set()
    base = "/proc/%d/fd" % int(pid)
    try:
        names = os.listdir(base)
    except OSError:
        return out
    for name in names:
        try:
            target = os.readlink(os.path.join(base, name))
        except OSError:
            continue
        if target.startswith("socket:["):
            out.add(target[8:-1])
    return out


def _established_inodes(host_hex: str) -> set:
    """The inodes of ESTABLISHED IPv4 sockets whose PEER is ``host_hex``.

    /proc/net/tcp is the whole namespace, so it is intersected with the
    pid's own fds by the caller -- this half only narrows by peer.
    """
    out = set()
    try:
        with open("/proc/net/tcp", "r", encoding="ascii") as handle:
            rows = handle.read().splitlines()[1:]
    except OSError:
        return out
    for row in rows:
        cols = row.split()
        if len(cols) < 10 or cols[3] != TCP_ESTABLISHED:
            continue
        peer = cols[2].split(":")[0]
        if peer.upper() != host_hex:
            continue
        out.add(cols[9])
    return out


def has_socket_to(pid: int, host: str) -> bool:
    """Does ``pid`` hold an ESTABLISHED IPv4 socket to ``host``?

    Necessary evidence of a connection, and not sufficient on its own --
    a viewer waiting to be accepted holds one too, which is why the caller
    also asks whether bytes are arriving.
    """
    try:
        host_hex = _hex_v4(host)
    except OSError:
        return False
    mine = _socket_inodes(pid)
    if not mine:
        return False
    return bool(mine & _established_inodes(host_hex))


def rchar(pid: int) -> Optional[int]:
    """Bytes this pid has read, from /proc/<pid>/io, or None.

    None is NO OPINION -- the file is unreadable on some kernels and for
    some ownerships -- and the cast lane turns that into "I cannot tell
    whether it connected" rather than into a landing.
    """
    try:
        with open("/proc/%d/io" % int(pid), "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("rchar:"):
                    return int(line.split(":", 1)[1].strip())
    except (OSError, ValueError):
        return None
    return None


__all__ = ["has_socket_to", "rchar"]
