"""ROUND 4: the /proc evidence for the OTHER direction, Spark -> HPCOMPUTER.

Round 3's probe asks about a process THIS app started. The Spark ->
HPCOMPUTER cast has no such process on this box: the viewer runs on
HPCOMPUTER and the Spark is the end being VIEWED, so the only local
evidence is an INBOUND connection from that host to the screen-sharing
port, and bytes leaving over it.

THE TRAP THIS EXISTS TO AVOID, and it is not hypothetical: the Windows
helper's own long poll is an ESTABLISHED socket to HPCOMPUTER, held open
about 2.4 times a minute for 25 s at a time. "Is there a connection to
that host" is therefore TRUE almost always and means nothing. The probe
must be scoped to the screen-sharing port or it is a rubber stamp.

Everything here is read out of /proc about THIS pytest process, plus one
LOOPBACK pair opened and closed inside the test. Nothing reaches
HPCOMPUTER, nothing opens a camera or a capture device, and nothing here
starts a RustDesk session.
"""
from __future__ import annotations

import os
import pathlib
import re
import socket
import threading

from jarvis import procnet


def loopback_pair():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    accepted = []
    thread = threading.Thread(target=lambda: accepted.append(server.accept()))
    thread.daemon = True
    thread.start()
    client = socket.create_connection(("127.0.0.1", port), timeout=2.0)
    thread.join(timeout=2.0)
    return server, client, accepted, port


def test_an_inbound_connection_is_found_by_host_AND_local_port():
    """The port scoping is the whole point: without it the helper's own
    poll to HPCOMPUTER answers yes for ever."""
    server, client, accepted, port = loopback_pair()
    try:
        on_port = procnet.established_to("127.0.0.1", (port,))
        print("\n  loopback pair on port %d: %d socket(s) at that port"
              % (port, len(on_port)))
        assert on_port, "the accepted socket must be found at its own port"
        other = procnet.established_to("127.0.0.1", (port + 1,))
        print("  the same host at port %d: %d socket(s)" % (port + 1, len(other)))
        assert not other, "a different local port must not count"
        nowhere = procnet.established_to("192.0.2.1", (port,))
        assert not nowhere
        assert not procnet.established_to("rm -rf /", (port,))
        print("  a host this box holds nothing to: %d" % len(nowhere))
    finally:
        client.close()
        for sock, _addr in accepted:
            sock.close()
        server.close()


def test_a_socket_can_be_traced_back_to_the_process_holding_it():
    """The byte counter needs a pid, and an inbound socket does not come
    with one. A pid that cannot be found is NO OPINION, never a no."""
    server, client, accepted, port = loopback_pair()
    try:
        inodes = procnet.established_to("127.0.0.1", (port,))
        assert inodes
        pids = {procnet.pid_for_inode(i) for i in inodes}
        print("\n  inodes %s -> pids %s (this process is %d)"
              % (sorted(inodes), sorted(p for p in pids if p), os.getpid()))
        assert os.getpid() in pids
        assert procnet.pid_for_inode("not an inode") is None
        assert procnet.pid_for_inode("999999999999") is None
    finally:
        client.close()
        for sock, _addr in accepted:
            sock.close()
        server.close()


def test_bytes_written_is_a_number_for_a_live_pid_and_no_opinion_otherwise():
    """The serving end STREAMS OUT, so the counter that matters here is
    the write side -- the mirror of round 3's ``rchar`` on the viewer."""
    mine = procnet.wchar(os.getpid())
    assert isinstance(mine, int) and mine > 0
    print("\n  wchar for this pid: %d" % mine)
    assert procnet.wchar(999999) is None
    assert procnet.wchar("not a pid") is None


def test_the_module_still_only_ever_reads():
    """Source-level, unchanged in spirit: this file may look at /proc and
    nothing else. It may not connect, spawn or write."""
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "jarvis" / "procnet.py").read_text()
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))
    code = re.sub(r'"""[\s\S]*?"""', "", code)
    for banned in ("subprocess", "Popen", "connect", "urlopen", "requests",
                   "cv2", "VideoCapture", 'open("/dev', "remove",
                   "unlink", "system(", "__import__"):
        assert banned not in code, banned
    print("\n  jarvis/procnet.py opens %d file(s), all of them under /proc"
          % code.count("open("))
    assert "/proc/net/tcp" in code
    assert "socket.socket" not in code
