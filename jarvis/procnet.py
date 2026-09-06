"""Readings out of /proc for the cast lane. Both directions of it.

They answer "is that cast actually carrying a desktop", which
``Popen.poll() is None`` cannot: a RustDesk viewer parked on an
accept-or-password prompt is a live process, and round 2 called that a
landing.

THE TWO DIRECTIONS NEED DIFFERENT EVIDENCE, because only one of them has a
process on this box.

* HPCOMPUTER -> SPARK. The viewer is OURS: this app spawned it. Ask about
  that pid -- ``has_socket_to`` and ``rchar``.
* SPARK -> HPCOMPUTER. The viewer runs on HIS Windows machine and there is
  no pid here to ask about. The Spark is the end being VIEWED, so the
  local evidence is an INBOUND connection from that host to the
  screen-sharing port, and bytes leaving over it --
  ``established_to`` -> ``pid_for_inode`` -> ``wchar``.

AND THE SECOND ONE HAS A TRAP THE FIRST DOES NOT. The Windows helper's own
long poll is an ESTABLISHED socket to HPCOMPUTER, held open about 2.4
times a minute for 25 seconds at a time. "Is there a connection to that
host" is therefore true almost always and means nothing at all, which is
why ``established_to`` takes the LOCAL PORTS and the caller passes the
screen-sharing ones. A probe that is true whatever happens is worse than
no probe: it is a rubber stamp on a sentence.

NOTHING HERE OPENS A SOCKET, A WINDOW OR A CAPTURE DEVICE. It reads text
files the kernel already wrote and nothing it reads leaves the box. It
cannot see a picture and there is no path from it to one.
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


def _tcp_rows() -> list:
    """Every row of /proc/net/tcp bar the header, or an empty list."""
    try:
        with open("/proc/net/tcp", "r", encoding="ascii") as handle:
            return handle.read().splitlines()[1:]
    except OSError:
        return []


def _established_inodes(host_hex: str) -> set:
    """The inodes of ESTABLISHED IPv4 sockets whose PEER is ``host_hex``.

    /proc/net/tcp is the whole namespace, so it is intersected with the
    pid's own fds by the caller -- this half only narrows by peer.
    """
    out = set()
    for row in _tcp_rows():
        cols = row.split()
        if len(cols) < 10 or cols[3] != TCP_ESTABLISHED:
            continue
        peer = cols[2].split(":")[0]
        if peer.upper() != host_hex:
            continue
        out.add(cols[9])
    return out


def established_to(host: str, ports=None) -> set:
    """The inodes of ESTABLISHED IPv4 sockets whose PEER is ``host``,
    narrowed to a set of LOCAL ports.

    ``ports`` is not optional in spirit even though it is in signature:
    with it left out this answers "any connection at all to that machine",
    and the Windows helper's own long poll makes that true almost always.
    The callers in the cast lane always pass the screen-sharing ports.

    No pid is involved, because the connection this is about was made from
    the other machine and nothing on this box owns it in a way we can name
    up front. ``pid_for_inode`` is the second step.
    """
    try:
        host_hex = _hex_v4(host)
    except OSError:
        return set()
    wanted = None
    if ports is not None:
        wanted = set()
        for port in ports:
            try:
                wanted.add(int(port))
            except (TypeError, ValueError):
                continue
        if not wanted:
            return set()
    out = set()
    for row in _tcp_rows():
        cols = row.split()
        if len(cols) < 10 or cols[3] != TCP_ESTABLISHED:
            continue
        if cols[2].split(":")[0].upper() != host_hex:
            continue
        if wanted is not None:
            try:
                local = int(cols[1].split(":")[1], 16)
            except (IndexError, ValueError):
                continue
            if local not in wanted:
                continue
        out.add(cols[9])
    return out


def pid_for_inode(inode) -> Optional[int]:
    """Which process holds that socket inode, or None for NO OPINION.

    None covers three different things and the cast lane treats them
    alike: the socket went away, it belongs to another user whose /proc
    this app may not look into, or the scan simply lost the race. All
    three mean "I cannot tell", and the lane turns that into "I can't say
    it landed" rather than into a landing.

    It scans /proc's numeric directories. That is the only way back from
    an inode to a pid without opening a netlink socket, and this module
    opens nothing.
    """
    want = "socket:[%s]" % str(inode)
    try:
        names = os.listdir("/proc")
    except OSError:
        return None
    for name in names:
        if not name.isdigit():
            continue
        base = "/proc/%s/fd" % name
        try:
            fds = os.listdir(base)
        except OSError:
            continue                        # not ours to look into
        for fd in fds:
            try:
                if os.readlink(os.path.join(base, fd)) == want:
                    return int(name)
            except OSError:
                continue
    return None


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


def _io_field(pid, field: str) -> Optional[int]:
    """One counter out of /proc/<pid>/io, or None for NO OPINION.

    None is not zero and it is not a no: the file is unreadable on some
    kernels and for some ownerships, and the cast lane turns that into "I
    cannot tell whether it connected" rather than into a landing.
    """
    head = "%s:" % field
    try:
        with open("/proc/%d/io" % int(pid), "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith(head):
                    return int(line.split(":", 1)[1].strip())
    except (OSError, ValueError):
        return None
    return None


def rchar(pid: int) -> Optional[int]:
    """Bytes this pid has taken in. The VIEWING end's evidence."""
    return _io_field(pid, "rchar")


def wchar(pid: int) -> Optional[int]:
    """Bytes this pid has put out. The SERVING end's evidence, and the
    mirror of ``rchar``: when HPCOMPUTER is the one watching, the stream
    leaves this box rather than arriving at it."""
    return _io_field(pid, "wchar")


__all__ = ["established_to", "has_socket_to", "pid_for_inode", "rchar",
           "wchar"]
