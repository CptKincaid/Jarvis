"""The USERS page ON A DISPLAY, at the window he actually runs.

Everything here needs a real Tk tree, so it runs only when
``JARVIS_UI_TEST_DISPLAY`` names a PRIVATE Xvfb and refuses his desktop
displays outright -- the rule tests/test_ui_layout_rules.py and
tests/test_ui_classic_frozen.py already write down.

RENDERED AT 1040x1760, HIS WINDOW, and at 920x1440. This morning a drawer
row was reviewed twice at 920x1440 and overflowed its slot at his real
size, and he found it in ten minutes; a page whose height grows with the
number of people is exactly the shape that does that again.

No registry of his is opened: every snapshot here is a dict of invented
people, handed to the page through the services seam.
"""
import os

import pytest

from jarvis.ui import theme
from jarvis.ui import users_page as up
from jarvis.ui import widgets as wg

FORBIDDEN_DISPLAYS = (":0", ":1")
FD_SETSIZE = 1024
HIS_W, HIS_H = 1040, 1760
OLD_W, OLD_H = 920, 1440
SCALE = 2.0

FAKE_HASH = "zzz$1$not-a-real-hash$zzz"


def _display() -> str:
    d = (os.environ.get("JARVIS_UI_TEST_DISPLAY") or "").strip()
    if not d:
        return ""
    base = d.split(".")[0]
    if base in FORBIDDEN_DISPLAYS or base.split(":")[-1] in ("0", "1"):
        pytest.fail("JARVIS_UI_TEST_DISPLAY=%r is a desktop display; this "
                    "file builds windows and will not open one on his "
                    "screen" % d)
    return d


@pytest.fixture
def root():
    display = _display()
    if not display:
        pytest.skip("set JARVIS_UI_TEST_DISPLAY=:9N (a private Xvfb) to run "
                    "the measured users-page tests")
    import tkinter as tk
    try:
        fds = len(os.listdir("/proc/self/fd"))
    except OSError:
        fds = 0
    if fds >= FD_SETSIZE - 32:
        pytest.skip("this process already holds %d descriptors" % fds)
    try:
        r = tk.Tk(screenName=display)
    except tk.TclError as exc:
        pytest.skip("no X server at %s: %s" % (display, exc))
    r.geometry("%dx%d+0+0" % (HIS_W, HIS_H))
    theme.resolve_fonts(r)
    theme.apply_scale(SCALE)
    wg.set_scale(SCALE)
    yield r
    try:
        r.destroy()
    except Exception:                              # noqa: BLE001 - teardown
        pass
    theme.apply_scale(1.0)
    wg.set_scale(1.0)
    theme.select_look(theme.DEFAULT_LOOK)


class Svc:
    """A stand-in app half. Records every call; holds no secret."""

    def __init__(self, snapshot=None, unlock=(True, "Unlocked, sir.")):
        self.snapshot = snapshot
        self.calls = []
        self._unlock = unlock

    def people_snapshot(self):
        self.calls.append("snapshot")
        return self.snapshot

    def people_unlock(self, code):
        self.calls.append("unlock")
        return self._unlock

    def people_add(self, **kw):
        self.calls.append(("add", kw.get("label")))
        return True, "enrolled"

    def people_set_role(self, label, role, **kw):
        self.calls.append(("set_role", label, role))
        return True, "changed"

    def people_forget(self, label):
        self.calls.append(("forget", label))
        return True, "forgotten"


def _snapshot(n=2, *, fault="", admin=None, code=True):
    from jarvis import gate as gt
    people = [{"label": "alderman", "name": "Alderman", "role": "owner",
               "voice": True, "face": "alderman", "face_dim": 128,
               "has_phrase": False, "has_code": code, "consent": "owner",
               "enrolled_at": "2026-01-01T09:00:00"}]
    for i in range(max(0, n - 1)):
        people.append({"label": "guest%d" % i, "name": "Guest %d" % i,
                       "role": "known", "voice": False, "face": "guest%d" % i,
                       "face_dim": 128, "has_phrase": False,
                       "has_code": False, "consent": "console",
                       "enrolled_at": "2026-02-02T09:00:00"})
    if fault:
        people = []
    return {"people": people, "gate_line": "owner-gate: SHADOW -- 1 owner "
            "(alderman), voice leg live, face leg unavailable. Nothing is "
            "being refused.", "fault_kind": fault,
            "path": "/tmp/nowhere/people.json", "gallery": ["alderman"],
            "admin": admin or (gt.ADMIN_CODE if code else gt.ADMIN_NOCODE),
            "admin_line": "an owner has set an override code"}


def _page(root, look="holo", snapshot=None, geometry=(HIS_W, HIS_H)):
    """The page, placed over a host of exactly his stage size."""
    import tkinter as tk
    theme.select_look(look)
    w, h = geometry
    root.geometry("%dx%d+0+0" % (w, h))
    host = tk.Frame(root, bg=theme.BG, width=w, height=h)
    host.pack(fill="both", expand=True)
    host.pack_propagate(False)
    stage = tk.Frame(host, bg=theme.BG)
    stage.pack(fill="both", expand=True)
    svc = Svc(_snapshot() if snapshot is None else snapshot)
    page = up.UsersPage(host, services=svc, cover=(stage,))
    root.update_idletasks()
    page.show()
    root.update_idletasks()
    root.update()
    return page, svc, host


def _walk(widget):
    yield widget
    for child in widget.winfo_children():
        yield from _walk(child)


def _texts(widget):
    out = []
    for w in _walk(widget):
        for key in ("text",):
            try:
                out.append(str(w.cget(key)))
            except Exception:                      # noqa: BLE001
                pass
        if w.__class__.__name__ == "Canvas":
            try:
                for item in w.find_all():
                    if w.type(item) == "text":
                        out.append(str(w.itemcget(item, "text")))
            except Exception:                      # noqa: BLE001
                pass
    return out


# =============================================== it builds, in both looks
@pytest.mark.parametrize("look", ["holo", "classic"])
def test_the_page_builds_and_places_over_the_stage(root, look):
    page, _svc, host = _page(root, look)
    assert page.is_open is True
    assert page.winfo_ismapped()
    assert page.winfo_width() == host.winfo_width()
    page.hide()
    root.update_idletasks()
    assert page.is_open is False
    assert not page.winfo_ismapped()


def test_showing_the_page_reads_the_file_and_hiding_it_relocks(root):
    page, svc, _host = _page(root)
    assert "snapshot" in svc.calls
    page._lock.unlock()
    assert page._lock.locked() is False
    page.hide()
    assert page._lock.locked() is True, "leaving the tab must relock at once"


def test_the_page_never_polls_the_file(root):
    """Unlike SENSORS there is nothing to poll: the registry is a file, and
    a timer on a file is cost with no reading behind it. The page reads it
    on show() and the app seam re-reads before every write."""
    page, svc, _host = _page(root)
    before = [c for c in svc.calls if c == "snapshot"]
    for _ in range(20):
        root.update()
    assert [c for c in svc.calls if c == "snapshot"] == before


# ================================================= the code, on the screen
def test_the_unlock_entry_is_masked_and_starts_empty(root):
    page, _svc, _host = _page(root)
    entry = page._code_entry
    assert entry is not None
    assert entry.cget("show") not in ("", None), "the code must not be echoed"
    assert entry.get() == ""


def test_a_press_empties_the_entry_before_anything_is_called(root, monkeypatch):
    """The clear happens on the press, not when the call comes back. Run
    inline here (the real control spawns a thread, and Tk cannot be read
    from one) so the service can look at the entry while it is "running"."""
    monkeypatch.setattr(up.UsersUnlockControl, "_thread",
                        staticmethod(lambda fn, *a: fn(*a)))
    page, svc, _host = _page(root)
    page._code_entry.insert(0, "zzz-typed-here-zzz")
    seen = {}

    def watcher(code):
        seen["entry_was"] = page._code_entry.get()
        seen["len"] = len(code)
        return True, "Unlocked, sir."

    svc.people_unlock = watcher
    assert page._unlock_pressed() == "started"
    assert page._code_entry.get() == ""
    assert seen["entry_was"] == "", \
        "the entry still held the code while the check ran"
    assert seen["len"] == len("zzz-typed-here-zzz")
    root.update()
    assert page._lock.locked() is False, "a good code should unlock the tab"
    # ...and nothing typed is anywhere on the page afterwards
    assert "zzz-typed-here-zzz" not in "\n".join(_texts(page))


def test_no_widget_anywhere_carries_a_hash_or_a_code(root):
    snap = _snapshot()
    # a snapshot that somehow carried one -- the page must still not draw it
    snap["people"][0]["code_hash"] = FAKE_HASH
    snap["people"][0]["phrase_hash"] = FAKE_HASH
    page, _svc, _host = _page(root, snapshot=snap)
    blob = "\n".join(_texts(page))
    assert FAKE_HASH not in blob
    assert "zzz" not in blob
    # ...and it says set / not set instead
    assert "code: set" in blob


# ============================================ what it says about each person
def test_every_person_and_the_gate_s_own_line_are_on_the_page(root):
    page, _svc, _host = _page(root, snapshot=_snapshot(3))
    blob = "\n".join(_texts(page))
    assert "Alderman" in blob and "Guest 0" in blob and "Guest 1" in blob
    assert "owner-gate:" in blob, "the tab and the log must agree on the mode"


def test_the_sole_owner_s_destructive_controls_are_disabled(root):
    page, _svc, _host = _page(root, snapshot=_snapshot(1))
    row = page._row_widgets["alderman"]
    # RoundButton keeps its own disabled state (a Canvas has no -state a
    # button honours), and invoke() is what a press actually calls.
    assert row["forget"]._state == "disabled"
    assert row["role"]._state == "disabled"
    row["forget"].invoke()
    row["role"].invoke()
    assert page._forget.armed_for == "", "a disabled Forget still armed"
    assert "only owner" in "\n".join(_texts(page))


def test_a_guest_may_be_forgotten(root):
    page, _svc, _host = _page(root, snapshot=_snapshot(2))
    row = page._row_widgets["guest0"]
    assert row["forget"]._state == "normal"


# ================================================ the code guards the writes
def test_a_locked_page_refuses_a_write_and_calls_nothing(root):
    page, svc, _host = _page(root)
    assert page._lock.locked() is True
    svc.calls.clear()
    page._forget_pressed("guest0")
    assert not any(c for c in svc.calls if isinstance(c, tuple))
    assert page._toasts and "code" in page._toasts[-1][0].lower()


def test_an_unlocked_page_arms_the_forget_and_still_asks_twice(root):
    page, svc, _host = _page(root)
    page._lock.unlock()
    svc.calls.clear()
    page._forget_pressed("guest0")
    assert page._forget.armed_for == "guest0"
    assert not any(isinstance(c, tuple) and c[0] == "forget"
                   for c in svc.calls), "arming must not forget anybody"
    blob = "\n".join(_texts(page))
    assert "cannot be undone" in blob.lower()
    assert "gallery" in blob.lower()
    # a wrong confirmation forgets nobody
    page._forget_confirm("guest0", "yes")
    assert not any(isinstance(c, tuple) and c[0] == "forget"
                   for c in svc.calls)
    # the label, typed, does
    page._forget_confirm("guest0", "guest0")
    assert ("forget", "guest0") in svc.calls


def test_an_owner_with_no_code_is_never_asked_for_one(root):
    page, svc, _host = _page(root, snapshot=_snapshot(2, code=False))
    assert page._lock.locked() is True
    svc.calls.clear()
    page._forget_pressed("guest0")
    assert page._forget.armed_for == "guest0", \
        "demanding a code he never set would lock him out of setting one"


# ==================================================== the bootstrap surface
def test_a_missing_registry_offers_the_first_owner(root):
    page, _svc, _host = _page(root, snapshot=_snapshot(fault="missing"))
    blob = "\n".join(_texts(page))
    assert "first owner" in blob.lower()
    assert page._add_btn is not None
    assert page._add_btn._state == "normal"


def test_a_corrupt_registry_offers_no_way_to_write_at_all(root):
    page, _svc, _host = _page(root, snapshot=_snapshot(fault="malformed"))
    blob = "\n".join(_texts(page))
    assert "/tmp/nowhere/people.json" in blob
    assert page._add_btn._state == "disabled"
    page._add_btn.invoke()
    assert page._adding is False, "a disabled Add still opened the form"


# =========================================================== the geometry
@pytest.mark.parametrize("geom", [(HIS_W, HIS_H), (OLD_W, OLD_H)])
@pytest.mark.parametrize("look", ["holo", "classic"])
def test_the_add_button_is_on_screen_at_both_his_sizes(root, geom, look):
    """The bug he photographed on 09-05 was SAVE falling off the bottom of
    a scrolling column. This page grows with the number of people, so the
    foot is pinned from the first build rather than retrofitted."""
    page, _svc, _host = _page(root, look, snapshot=_snapshot(12),
                              geometry=geom)
    root.update_idletasks()
    btn = page._add_btn
    assert btn.winfo_ismapped(), "the Add button was unmapped by the packer"
    bottom = page.winfo_rooty() + page.winfo_height()
    assert btn.winfo_rooty() + btn.winfo_height() <= bottom, \
        (geom, look, btn.winfo_rooty(), bottom)


def test_a_crowd_scrolls_the_body_and_leaves_the_foot_alone(root):
    page, _svc, _host = _page(root, snapshot=_snapshot(30))
    root.update_idletasks()
    assert page.overflow_px() > 0, "30 people should not fit"
    foot_before = page._foot.winfo_rooty()
    page._scroll(4)
    root.update_idletasks()
    assert page._foot.winfo_rooty() == foot_before


def test_his_own_registry_shape_fits_without_scrolling(root):
    """MEASURED for the state he is actually in: one owner, plus room."""
    page, _svc, _host = _page(root, snapshot=_snapshot(1))
    root.update_idletasks()
    assert page.overflow_px() == 0


# ================================================== the look is read late
def test_the_module_captures_no_look_at_import_time():
    import ast
    tree = ast.parse(open(up.__file__).read())
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        text = ast.dump(node)
        assert "theme.LOOK" not in text and "'LOOK'" not in text, \
            ast.unparse(node)
        for token in ("theme.BG", "theme.INK", "theme.CYAN", "theme.MUTED",
                      "theme.WARN", "theme.RAISED", "theme.SURFACE"):
            assert token not in ast.unparse(node), ast.unparse(node)
