"""'Run diagnostics': a status line in character, from real data."""
from types import SimpleNamespace

from jarvis.commander import _DIAG_RX, _h_diagnostics


def test_the_phrasings():
    for said in ("run diagnostics", "diagnostics", "run a diagnostic", "system status",
                 "status report", "self test", "how are your systems"):
        assert _DIAG_RX.match(said), said
    assert not _DIAG_RX.match("what's the status of the build")


def test_the_handler_speaks_what_the_app_reports():
    c = SimpleNamespace(_svc=lambda n: (lambda: "All systems nominal, sir.") if n == "diagnostics" else None)
    res = _h_diagnostics(c, "run diagnostics", None)
    assert res.handled and res.speak and res.reply == "All systems nominal, sir."
    assert _h_diagnostics(SimpleNamespace(_svc=lambda n: None), "diagnostics", None) is None
