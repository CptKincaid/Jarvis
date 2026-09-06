"""The two /proc readings the cast lane's connection evidence stands on.

They are read against THIS pytest process and against a pid that does not
exist. Nothing here opens a socket, a viewer, a window or a capture
device, and nothing reaches HPCOMPUTER: the point of the module is that
it asks the kernel about a process this box already owns.
"""
from __future__ import annotations

import os
import pathlib
import re
import socket
import threading

from jarvis import procnet


def test_bytes_read_is_a_number_for_a_live_pid_and_no_opinion_for_a_dead_one():
    """``rchar`` climbs, and a pid that is not there is None -- NO
    OPINION, which jarvis/castview.py turns into "I can't tell whether it
    connected" rather than into a landing."""
    mine = procnet.rchar(os.getpid())
    assert isinstance(mine, int) and mine > 0
    pathlib.Path(__file__).read_bytes()          # do some reading
    again = procnet.rchar(os.getpid())
    print("\n  rchar %d -> %d" % (mine, again))
    assert again >= mine
    assert procnet.rchar(999999) is None
    assert procnet.rchar("not a pid") is None


def test_an_established_socket_is_found_only_for_the_pid_that_owns_it():
    """A loopback pair, opened and closed inside this test. It never
    leaves the machine and it is torn down before the test returns."""
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
    try:
        assert procnet.has_socket_to(os.getpid(), "127.0.0.1") is True
        # ...and a host this process holds nothing to
        assert procnet.has_socket_to(os.getpid(), "192.0.2.1") is False
        # ...and a pid that is not there
        assert procnet.has_socket_to(999999, "127.0.0.1") is False
        # ...and rubbish where an address should be
        assert procnet.has_socket_to(os.getpid(), "rm -rf /") is False
        print("\n  loopback pair on port %d: found for this pid, not for "
              "999999, not for 192.0.2.1" % port)
    finally:
        client.close()
        for sock, _addr in accepted:
            sock.close()
        server.close()


def test_the_module_only_ever_reads():
    """Source-level, the way jarvis/castview.py and jarvis/gesture.py are
    pinned: this file may look at /proc and nothing else. In particular it
    may not connect, spawn or write."""
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "jarvis" / "procnet.py").read_text()
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))
    code = re.sub(r'"""[\s\S]*?"""', "", code)
    for banned in ("subprocess", "Popen", "connect", "urlopen", "requests",
                   "cv2", "VideoCapture", "open(\"/dev", "write", "remove",
                   "unlink", "system(", "__import__"):
        assert banned not in code, banned
    # the only things it opens are the two /proc files it documents
    assert code.count("open(") == 2
    assert "/proc/net/tcp" in code and "/proc/%d/io" in code
