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
        admin = admin or (gt.ADMIN_REFUSE if fault in ("malformed",
                                                       "unreadable")
                          else gt.ADMIN_FIRST)
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


def test_the_page_never_polls_the_heavy_snapshot(root):
    """The full snapshot rebuilds every row, the startup line and the
    gallery listing. That is far too much for a one-second tick, so the
    tick asks people_admin_state -- one small file read and three values --
    and only does a full refresh when the ANSWER changed."""
    page, svc, _host = _page(root)
    before = [c for c in svc.calls if c == "snapshot"]
    for _ in range(20):
        page._tick_lock()
        root.update()
    assert [c for c in svc.calls if c == "snapshot"] == before


# ==================================== the lock is a decision, not a memory
class LiveSvc(Svc):
    """An app half that answers the two seams that keep the page honest.

    ``admin`` and ``unlocked_s`` are set by the test, the way the real app
    sets them from the file on disk and its own dwell. Every write refuses
    with the gate's own sentence when a code is owed, which is what the
    real seam does -- the PAGE is not what decides it.
    """

    def __init__(self, snapshot=None, admin=None, unlocked_s=0.0):
        super().__init__(snapshot)
        from jarvis import gate as gt
        self.admin = admin or gt.ADMIN_NOCODE
        self.unlocked_s = float(unlocked_s)
        self.relocked = 0

    def people_admin_state(self):
        self.calls.append("admin_state")
        return {"admin": self.admin, "admin_line": "the file says %s"
                % self.admin, "unlocked_s": self.unlocked_s}

    def people_snapshot(self):
        """OFF THE FILE, like the real one: app.people_snapshot re-reads
        rather than handing back the gate's boot-time copy, so a snapshot
        taken after the code was set says so."""
        self.calls.append("snapshot")
        snap = dict(self.snapshot or {})
        snap["admin"] = self.admin
        return snap

    def people_relock(self):
        self.calls.append("relock")
        self.relocked += 1
        self.unlocked_s = 0.0

    def _refuse_if_owed(self):
        from jarvis import gate as gt
        if self.admin == gt.ADMIN_CODE and self.unlocked_s <= 0:
            return True, gt.ADMIN_CODE_OWED
        return False, ""

    def people_forget(self, label):
        owed, why = self._refuse_if_owed()
        self.calls.append(("forget", label, "refused" if owed else "done"))
        return (False, why) if owed else (True, "forgotten")

    def people_add(self, **kw):
        owed, why = self._refuse_if_owed()
        self.calls.append(("add", kw.get("label"),
                           "refused" if owed else "done"))
        return (False, why) if owed else (True, "enrolled")


def _live(root, admin=None, unlocked_s=0.0, snapshot=None):
    """The page opened on a snapshot that says NO code is owed -- his real
    starting point, an owner he made in this tab with no code set yet."""
    import tkinter as tk
    from jarvis import gate as gt
    theme.select_look("holo")
    root.geometry("%dx%d+0+0" % (HIS_W, HIS_H))
    host = tk.Frame(root, bg=theme.BG, width=HIS_W, height=HIS_H)
    host.pack(fill="both", expand=True)
    host.pack_propagate(False)
    snap = snapshot if snapshot is not None else _snapshot(code=False)
    svc = LiveSvc(snap, admin=admin or gt.ADMIN_NOCODE,
                  unlocked_s=unlocked_s)
    page = up.UsersPage(host, services=svc)
    root.update_idletasks()
    page.show()
    root.update_idletasks()
    return page, svc, host


def test_a_code_set_while_the_tab_is_open_raises_the_unlock_row(root):
    """HIS SEQUENCE, ON A REAL PAGE. He makes the first owner here, the
    foot note tells him a new code still needs a terminal, he goes and sets
    one, and he comes back to the tab he never closed.

    ``show()`` was the only thing that re-read the file, so switching tabs
    and back was the only way to see it -- and until then ``_paint_lock``
    drew no unlock row at all, because the snapshot still said open_nocode.
    """
    from jarvis import gate as gt
    page, svc, _host = _live(root)
    assert page.admin_state == gt.ADMIN_NOCODE
    assert not page._lock_row.winfo_ismapped(), "a code was not owed yet"
    assert page._guard("forget") is True

    svc.admin = gt.ADMIN_CODE          # he sets one at a terminal
    page._tick_lock()
    root.update_idletasks()

    assert page.admin_state == gt.ADMIN_CODE
    assert page._lock_row.winfo_ismapped(), "no unlock row was drawn"
    assert page._code_entry.winfo_ismapped()
    assert page._guard("forget") is False


def test_a_write_refused_by_the_seam_raises_the_row_and_takes_the_cursor(
        root):
    """THE PAGE IS NOT THE GUARD. Even from a stale snapshot that lets the
    press through, the seam refuses and the page must then show him where
    to type -- the row lives in the pinned head, and he is looking at a
    person block halfway down."""
    from jarvis import gate as gt
    page, svc, _host = _live(root)
    svc.admin = gt.ADMIN_CODE          # the file changed; the page has not
    assert page._guard("forget") is True, "the stale snapshot still allows"

    page._write("people_forget", "guest0")
    root.update_idletasks()

    assert ("forget", "guest0", "refused") in svc.calls
    assert page._toasts[-1][0] == gt.ADMIN_CODE_OWED
    assert page._lock_row.winfo_ismapped()
    # focus_lastfor, not focus_get: Xvfb runs with no window manager, so
    # the toplevel never holds the keyboard and focus_get() is None however
    # the page behaved. focus_lastfor is where the focus WILL land.
    assert page.focus_lastfor() is page._code_entry


def test_leaving_the_tab_relocks_the_app_and_not_only_the_page(root):
    """A page that dropped its own dwell and left the app's standing looked
    locked and was not."""
    page, svc, _host = _live(root, unlocked_s=90.0)
    page._lock.unlock()
    page.hide()
    assert svc.relocked == 1
    assert page._lock.locked() is True


def test_the_lock_button_relocks_the_app_too(root):
    page, svc, _host = _live(root, unlocked_s=90.0)
    page._lock.unlock()
    page._lock_pressed()
    root.update_idletasks()
    assert svc.relocked == 1
    assert page._lock.locked() is True


def test_the_countdown_on_screen_is_the_apps_dwell_not_a_second_clock(root):
    """Two 120-second clocks started a moment apart is two answers to the
    same question. The tick takes the app's remaining seconds as the
    truth."""
    from jarvis import gate as gt
    page, svc, _host = _live(root, admin=gt.ADMIN_CODE,
                             snapshot=_snapshot(code=True))
    page._lock.unlock()                # the page thinks it has 120 s
    svc.unlocked_s = 4.0               # the app knows it has four
    page._tick_lock()
    root.update_idletasks()
    assert 3.0 <= page._lock.remaining() <= 4.0
    svc.unlocked_s = 0.0               # and when the app shuts, so does it
    page._tick_lock()
    assert page._lock.locked() is True


def test_an_app_half_without_the_two_seams_still_draws(root):
    """build_ui_services drops what an older app half does not offer, so
    the page has to work with neither."""
    page, _svc, _host = _page(root)          # plain Svc: no admin_state
    page._tick_lock()
    root.update_idletasks()
    assert page.is_open is True
    page.hide()                              # must not raise on no relock
    assert page._lock.locked() is True


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


def test_only_the_face_fault_is_amber_and_not_the_whole_detail_line(root):
    """The same defect, on the widget tree at HIS window. The chips label
    must be MUTED even for the person whose pointer is stale; the amber
    belongs to a line of its own that says what is wrong."""
    snap = _snapshot(n=2)
    snap["people"][1]["face"] = "gone-from-the-gallery"
    page, _svc, _host = _page(root, snapshot=snap)
    labels = [w for w in _walk(page)
              if w.__class__.__name__ == "Label"]
    detail = [w for w in labels
              if "phrase: not set" in str(w.cget("text"))]
    assert detail, "no chips line was drawn"
    for w in detail:
        assert str(w.cget("fg")) == str(theme.MUTED), (
            "the chips line is wearing the fault colour: %r"
            % str(w.cget("text")))
    amber = [w for w in labels
             if str(w.cget("fg")) == str(theme.WARN)
             and "gone-from-the-gallery" in str(w.cget("text"))]
    assert amber, "the dangling face pointer is never explained in amber"


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


# ======================================================= the add form itself
def _open_form(root, page, **fields):
    page._lock.unlock()
    page._add_pressed()
    root.update_idletasks()
    for key, value in fields.items():
        page._add_fields[key].delete(0, "end")
        page._add_fields[key].insert(0, value)
    return page._add_fields


def test_the_consent_paragraph_names_whoever_is_being_added(root):
    page, _svc, _host = _page(root)
    _open_form(root, page, label="pemberton")
    page._retitle_consent()
    text = page._consent_lbl.cget("text")
    assert "pemberton" in text
    assert "128" not in text, "adding a row stores no face measurement"
    assert "cannot command Jarvis" in text


def test_a_wrong_consent_adds_nobody(root):
    page, svc, _host = _page(root)
    _open_form(root, page, label="pemberton", name="Pemberton", consent="yes")
    svc.calls.clear()
    page._create_pressed()
    assert not any(isinstance(c, tuple) and c[0] == "add" for c in svc.calls)
    assert "nothing was written" in page._toasts[-1][0]


def test_their_own_label_typed_adds_them_as_a_console_consent(root):
    page, svc, _host = _page(root)
    _open_form(root, page, label="pemberton", name="Pemberton",
               consent="pemberton")
    svc.calls.clear()
    captured = {}
    svc.people_add = lambda **kw: (captured.update(kw), (True, "enrolled"))[1]
    page._create_pressed()
    assert captured["label"] == "pemberton"
    assert captured["role"] == "known"
    assert captured["consent"] == "console", \
        "a console consent must never be recorded as a terminal one"


def test_a_label_the_gallery_could_not_store_never_reaches_the_app(root):
    page, svc, _host = _page(root)
    _open_form(root, page, label="Pemberton Smythe", consent="x")
    svc.calls.clear()
    page._create_pressed()
    assert not any(isinstance(c, tuple) and c[0] == "add" for c in svc.calls)
    assert "label" in page._toasts[-1][0]


def test_a_face_label_the_gallery_could_not_store_is_refused_too(root):
    """The pointer is the whole reason the face leg can name anyone. A
    label one store would refuse must not reach the other."""
    page, svc, _host = _page(root)
    _open_form(root, page, label="pemberton", consent="pemberton",
               face="NOT A LABEL")
    svc.calls.clear()
    page._create_pressed()
    assert not any(isinstance(c, tuple) and c[0] == "add" for c in svc.calls)
    assert "face" in page._toasts[-1][0].lower()


def test_a_face_pointer_at_an_empty_gallery_is_allowed_but_said_out_loud(root):
    """He may add somebody before their face is enrolled. That is fine and
    it is not silent: an unresolved pointer is exactly how the face leg
    stops naming anyone, so it is named at the moment it is created."""
    page, svc, _host = _page(root)
    _open_form(root, page, label="pemberton", consent="pemberton",
               face="pemberton")
    svc.calls.clear()
    page._create_pressed()
    assert any(isinstance(c, tuple) and c[0] == "add" for c in svc.calls)
    joined = " ".join(line for line, _k in page._toasts)
    assert "gallery" in joined.lower()


def test_the_first_owner_form_asks_nobody_for_consent(root):
    page, _svc, _host = _page(root, snapshot=_snapshot(fault="missing"))
    page._add_pressed()
    root.update_idletasks()
    assert page._add_owner is True, "the first row must be an owner"
    assert "consent" not in page._add_fields
    assert "confirm_owner" not in page._add_fields


# ============================ what the first render at 1040x1760 showed
def test_the_unlock_row_is_actually_on_screen_when_a_code_is_set(root):
    """FOUND BY LOOKING AT THE RENDER, 2026-09-05, at 1040x1760. The page
    paints once at BUILD time, before the snapshot has been read, so the
    admin state is still "refuse" and the lock row is packed away. Nothing
    packed it back when the snapshot said a code IS required: the tab said
    "locked" and offered no box to type the code into, which makes every
    administrative action unreachable rather than guarded."""
    page, _svc, _host = _page(root)
    root.update_idletasks()
    assert page._code_entry.winfo_ismapped(), \
        "the tab demands a code and shows nowhere to type one"
    assert page._unlock_btn.winfo_ismapped()
    assert page._lock_btn.winfo_ismapped()


def test_no_person_line_runs_off_the_right_edge(root):
    """ALSO FOUND BY LOOKING. The chips line was one long unwrapped Label,
    so "face: pemberton (128-D) · phrase: not set · code: not set" was cut
    mid-word at the window edge -- the same class of defect he photographed
    on the drawer this morning, and at his size, not a convenient one."""
    for geom in ((HIS_W, HIS_H), (OLD_W, OLD_H)):
        page, _svc, _host = _page(root, snapshot=_snapshot(3), geometry=geom)
        root.update_idletasks()
        width = page.winfo_width()
        assert width > 1
        for w in _walk(page._body):
            if w.__class__.__name__ != "Label":
                continue
            assert w.winfo_reqwidth() <= width, \
                (geom, w.cget("text")[:60], w.winfo_reqwidth(), width)
        page.hide()
        _host.destroy()


def test_the_foot_note_and_the_gate_line_wrap_too(root):
    page, _svc, _host = _page(root)
    root.update_idletasks()
    width = page.winfo_width()
    for lbl in (page._gate_lbl, page._admin_lbl, page._note_lbl):
        assert lbl.winfo_reqwidth() <= width, lbl.cget("text")[:60]


def test_the_lock_status_is_not_cut_off_by_its_own_row(root):
    """FOUND BY LOOKING at the second 1040x1760 render. The status sat
    BESIDE the entry and two buttons, which left it ~340 px of a 1040-px
    window, and "locked — the override code opens this tab and nothing
    else" was cut at "ope". The sensors page learned the same lesson this
    morning: a caption belongs BELOW the control, not next to it."""
    for geom in ((HIS_W, HIS_H), (OLD_W, OLD_H)):
        page, _svc, _host = _page(root, geometry=geom)
        root.update_idletasks()
        assert page._lock_lbl.winfo_reqwidth() <= page.winfo_width(), \
            (geom, page._lock_lbl.cget("text"))
        # and it is on its own line: below the entry, not beside it
        assert page._lock_lbl.winfo_rooty() > page._code_entry.winfo_rooty()
        page.hide()
        _host.destroy()


def test_arming_a_forget_brings_its_confirmation_into_view(root):
    """FOUND BY LOOKING at the 30-users-forget render at 1040x1760. The
    warning is ten lines long, so arming a forget on anybody but the first
    person pushed the entry and the "Forget <label>" button below the
    fold: he would press Forget, read a wall of amber, and find no way to
    confirm without discovering that the body scrolls."""
    page, _svc, _host = _page(root, snapshot=_snapshot(4))
    page._lock.unlock()
    page._forget_pressed("guest2")
    root.update_idletasks()
    entry = page._row_widgets["guest2"]["confirm"]
    assert entry.winfo_ismapped()
    top = page._canvas.winfo_rooty()
    bottom = top + page._canvas.winfo_height()
    assert entry.winfo_rooty() >= top, "the confirmation is above the fold"
    assert entry.winfo_rooty() + entry.winfo_height() <= bottom, \
        "the confirmation is below the fold"


# ===================== what the 31-users-add render showed at 1040x1760
@pytest.mark.parametrize("geom", [(HIS_W, HIS_H), (OLD_W, OLD_H)])
def test_the_delete_command_wraps_inside_its_own_slot_not_the_page(root, geom):
    """FOUND BY LOOKING AT 1040x1760, 2026-09-05.

    The delete command SHARES ITS ROW with the Copy button, but it was
    wrapped to the whole page width like every other label -- so it was
    told it had ~90 px more room than pack() had actually given it, and a
    line landing in that gap is drawn past the label's own window and
    clipped mid-word. It is the identical defect the chips line already
    carries a comment about; this row was simply missed because the
    button beside it is what makes the slot narrower.

    The invariant, checked at BOTH his sizes: a label never wraps to more
    room than it was given.
    """
    page, _svc, _host = _page(root, snapshot=_snapshot(n=2), geometry=geom)
    page._lock.unlock()
    page._forget_pressed("guest0")
    root.update_idletasks()
    root.update()
    cmd = [w for w in _walk(page)
           if w.__class__.__name__ == "Label"
           and "--delete" in str(w.cget("text"))]
    assert cmd, "the delete command is not on the panel"
    for w in cmd:
        allotted = int(w.winfo_width())
        assert allotted > 1, "the row never got laid out"
        assert int(w.cget("wraplength")) <= allotted, (
            "wrapped to %d px inside a %d px slot, so ~%d px of every full "
            "line is clipped" % (int(w.cget("wraplength")), allotted,
                                 int(w.cget("wraplength")) - allotted))


@pytest.mark.parametrize("geom", [(HIS_W, HIS_H), (OLD_W, OLD_H)])
def test_nothing_you_have_to_press_runs_off_the_right_edge(root, geom):
    """FOUND ON THE 920x1440 RENDER, 2026-09-05.

    The forget panel's row is an entry, "Forget <label>" and "Cancel", and
    the delete command shares its row with "Copy". At his 1040 window they
    fit. At 920 they did not: the body scrolls VERTICALLY only, so the
    overflow is not scrolled to, it is simply cut -- the render showed
    "Co" and "Car" against the window edge.

    A control you cannot fully see is a control you cannot trust you have
    pressed, and this is the destructive panel. Measured on the widget
    tree at both his sizes.
    """
    page, _svc, _host = _page(root, snapshot=_snapshot(n=2), geometry=geom)
    page._lock.unlock()
    page._forget_pressed("guest0")
    root.update_idletasks()
    root.update()
    # The body is a canvas that scrolls VERTICALLY. Its content frame is
    # forced to the canvas width, so anything the frame REQUESTS beyond
    # that is not scrolled to -- it is squeezed off the right edge. So the
    # measurement is the content frame's requested width against the
    # canvas it has to live in.
    cv = page._canvas
    inner = [root.nametowidget(cv.itemcget(i, "window"))
             for i in cv.find_all() if cv.type(i) == "window"]
    assert inner, "the scrolling body has no content frame"
    for w in inner:
        want, room = int(w.winfo_reqwidth()), int(cv.winfo_width())
        assert want <= room, (
            "the forget panel asks for %d px of a %d px body, so %d px of "
            "it is cut off the right edge" % (want, room, want - room))


@pytest.mark.parametrize("geom", [(HIS_W, HIS_H), (OLD_W, OLD_H)])
def test_no_label_on_this_page_wraps_wider_than_its_own_slot(root, geom):
    """THE GENERAL FORM of the defect above, so the next row that puts a
    button beside a paragraph cannot reintroduce it quietly.

    Swept across the three states the page actually reaches -- the list,
    an armed forget, and the add form -- at both his window sizes.
    """
    page, _svc, _host = _page(root, snapshot=_snapshot(n=3), geometry=geom)
    page._lock.unlock()
    bad = []
    for state in ("list", "forget", "add"):
        if state == "forget":
            page._forget_pressed("guest0")
        elif state == "add":
            page._forget.disarm()
            page._add_pressed()
        root.update_idletasks()
        root.update()
        for w in _walk(page):
            if w.__class__.__name__ != "Label":
                continue
            try:
                wrap = int(w.cget("wraplength"))
                got = int(w.winfo_width())
            except Exception:                      # noqa: BLE001
                continue
            if wrap and got > 1 and wrap > got:
                bad.append((state, str(w.cget("text"))[:44], wrap, got))
    assert bad == [], bad


@pytest.mark.parametrize("geom", [(HIS_W, HIS_H), (OLD_W, OLD_H)])
def test_create_is_pinned_where_add_is_and_never_scrolls_away(root, geom):
    """The consent paragraph is twenty lines. With Create at the bottom of
    the scrolling body it sat below the fold on a page that already has
    people on it -- the same defect as a SAVE button that scrolls away,
    which is the one he photographed this morning. The primary action
    lives in the pinned foot, beside where "Add a person" was."""
    page, _svc, _host = _page(root, snapshot=_snapshot(12), geometry=geom)
    page._lock.unlock()
    page._add_pressed()
    root.update_idletasks()
    btn = page._create_btn
    assert btn.winfo_ismapped(), "Create is not on screen"
    bottom = page.winfo_rooty() + page.winfo_height()
    assert btn.winfo_rooty() + btn.winfo_height() <= bottom, geom
    assert not page._add_btn.winfo_ismapped(), \
        "Add and Create must not both be offered"


def test_opening_the_form_shows_its_first_field(root):
    """The body kept the scroll position it had from the LAST thing that
    was open, so the form's name, label and face rows were above the fold
    the moment it appeared -- visible in the 31-users-add render, which
    opens on 'role'."""
    page, _svc, _host = _page(root, snapshot=_snapshot(12))
    page._lock.unlock()
    page._forget_pressed("guest9")          # scrolls right down
    root.update_idletasks()
    assert page._canvas.canvasy(0) > 0
    page._add_pressed()
    root.update_idletasks()
    assert page._canvas.canvasy(0) == 0, "the form opened part-way down"
    first = page._add_fields["name"]
    assert first.winfo_ismapped()
    assert first.winfo_rooty() >= page._canvas.winfo_rooty()


@pytest.mark.parametrize("geom", [(HIS_W, HIS_H), (OLD_W, OLD_H)])
def test_every_form_row_fits_the_window(root, geom):
    """The caption column was 26 CHARACTERS wide, which at S=2 is ~410 px
    of a 1040-px window, so "existing owner's label (owners only)" ran
    into its own entry."""
    page, _svc, _host = _page(root, geometry=geom)
    page._lock.unlock()
    page._add_pressed()
    root.update_idletasks()
    width = page.winfo_width()
    for entry in page._add_fields.values():
        right = (entry.winfo_rootx() - page.winfo_rootx()
                 + entry.winfo_width())
        assert right <= width, (geom, right, width)
    for w in _walk(page._body):
        if w.__class__.__name__ == "Label":
            assert w.winfo_reqwidth() <= width, (geom, w.cget("text")[:50])


# ------------------------------------------- the foot row: buttons take first
def _foot_widget(page, kind):
    """The live widget for 'create'/'cancel'/'path' out of the pinned foot."""
    return {"create": page._create_btn, "cancel": page._cancel_btn,
            "path": page._path_lbl}[kind]


# 920x1440 is the geometry the console was PHOTOGRAPHED at on 2026-09-05
# (scripts/ui_shots.py --geometry 920x1440 --scale 2.0), and it is the
# narrow end of what he uses; 1040x1760 is the stage size the rest of this
# file measures at. The defect below was invisible at the wider one, which
# is exactly why the pin has to carry both.
SHOT_W, SHOT_H = 920, 1440


@pytest.mark.parametrize("geometry", [(SHOT_W, SHOT_H), (HIS_W, HIS_H)],
                         ids=["photographed-920x1440", "his-stage-1040x1760"])
def test_the_cancel_button_is_drawn_whole_beside_a_long_people_path(
        root, geometry):
    """MEASURED on the shipped merge, 2026-09-05: the foot read
    "Create  Cance|/home/example/.local/state/jarvis/people.json" -- the
    path label had been packed at BUILD time, before Create and Cancel were
    packed at paint time, so the packer gave it its full width and cut the
    right-hand end off Cancel.

    This is the SECOND time this exact defect has been fixed in this file;
    the destructive panel's row carries the first. The rule both share: the
    BUTTONS take their width first and the path takes what is left."""
    snap = dict(_snapshot())
    snap["path"] = "/home/hunterp/.local/state/jarvis/people.json"
    page, _, host = _page(root, snapshot=snap, geometry=geometry)
    page._adding = True
    page._paint()
    root.update_idletasks()
    root.update()
    cancel = _foot_widget(page, "cancel")
    create = _foot_widget(page, "create")
    assert cancel.winfo_ismapped() and create.winfo_ismapped()
    # THE SQUEEZE IS THE SIGNAL. Tk does not overflow the parent; it hands
    # the LEFT-packed buttons whatever the right-packed label left and
    # narrows them, so a cut button keeps its winfo_reqwidth() and loses
    # winfo_width(). MEASURED pre-fix at 2.0 scale, with the path label
    # asking for 524 px at every window width:
    #     1040 px window -> Cancel 159 of 159   (fits; the defect is hidden)
    #      920 px window -> Cancel 138 of 159   (the photograph)
    #      880 px window -> Cancel  98 of 159
    #      800 px window -> Cancel  18 of 159
    #      760 px window -> Cancel   1 of 159   (gone entirely)
    # 1040 is why this pin carries both geometries: at his wider stage the
    # bug is invisible, and a pin that only measured there would have
    # passed on the broken build.
    row = cancel.master
    for name, btn in (("Create", create), ("Cancel", cancel)):
        assert btn.winfo_width() >= btn.winfo_reqwidth(), (
            "%s is drawn %d px wide in a %d px row but needs %d -- the "
            "path label took the room" % (name, btn.winfo_width(),
                                          row.winfo_width(),
                                          btn.winfo_reqwidth()))


def test_the_path_gives_up_its_width_and_keeps_the_tail(root):
    """The tail names the file, so the HEAD is what goes. An elided path
    must still end in the file name, and must never be wider than the room
    the buttons left it."""
    snap = dict(_snapshot())
    snap["path"] = ("/home/hunterp/some/deliberately/very/long/path/that/"
                    "cannot/possibly/fit/beside/two/buttons/people.json")
    page, _, host = _page(root, snapshot=snap, geometry=(SHOT_W, SHOT_H))
    page._adding = True
    page._paint()
    root.update_idletasks()
    root.update()
    shown = page._path_lbl.cget("text")
    assert shown != snap["path"], "the path was not trimmed at all"
    if shown:
        assert shown.endswith("people.json") or shown.startswith("…")
        row_w = page._path_lbl.master.winfo_width()
        assert page._path_lbl.winfo_reqwidth() < row_w, (
            "the trimmed path still asks for %d px of a %d px row"
            % (page._path_lbl.winfo_reqwidth(), row_w))
    # ...and Cancel is still whole beside it.
    cancel = _foot_widget(page, "cancel")
    assert cancel.winfo_width() >= cancel.winfo_reqwidth()


@pytest.mark.parametrize("geometry", [(SHOT_W, SHOT_H), (HIS_W, HIS_H)],
                         ids=["photographed-920x1440", "his-stage-1040x1760"])
def test_the_path_is_never_cut_without_an_ellipsis_to_say_so(root, geometry):
    """A label with anchor="e" that is narrower than its text shows the
    RIGHT end and silently drops the left. PHOTOGRAPHED on the first
    version of this fix: "ome/example/.local/state/jarvis/people.json" --
    the head was gone and nothing said so, because the fit ran before the
    geometry manager had given the row a width.

    THE RULE: whatever the label ends up showing, it must ASK FOR no more
    room than it has. If it was shortened, the ellipsis is what says so."""
    snap = dict(_snapshot())
    snap["path"] = "/home/hunterp/.local/state/jarvis/people.json"
    page, _, host = _page(root, snapshot=snap, geometry=geometry)
    page._adding = True
    page._paint()
    root.update_idletasks()
    root.update()
    lbl = _foot_widget(page, "path")
    if not lbl.winfo_ismapped():
        return
    assert lbl.winfo_reqwidth() <= lbl.winfo_width(), (
        "the path asks for %d px of the %d it was given, so %r is being "
        "drawn with its head cut off"
        % (lbl.winfo_reqwidth(), lbl.winfo_width(), lbl.cget("text")))
    shown = lbl.cget("text")
    if shown and shown != snap["path"]:
        assert shown.startswith("\u2026"), (
            "%r was shortened but does not say so" % shown)


def test_a_short_path_is_shown_in_full(root):
    """The fix must not trim what already fits."""
    snap = dict(_snapshot())
    snap["path"] = "/tmp/p.json"
    page, _, host = _page(root, snapshot=snap)
    page._adding = True
    page._paint()
    root.update_idletasks()
    root.update()
    assert page._path_lbl.cget("text") == "/tmp/p.json"
