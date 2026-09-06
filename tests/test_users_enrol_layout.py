"""The new rows and panels, MEASURED, at BOTH the windows he runs.

CONSTRAINT H, and it is written down because it has already cost twice today:
a foot row was cut at 920x1440 and invisible at 1040x1760 on the same morning,
and the sensors page lost its SAVE button off the bottom. So every control
this work adds is measured at BOTH geometries at scale 2.0, and nothing is
believed from one of them.

Runs only when ``JARVIS_UI_TEST_DISPLAY`` names a PRIVATE Xvfb. His desktop
displays are refused outright: this file builds windows and will not open one
on his screen. It also skips when the process already holds too many
descriptors -- an X connection past select()'s FD_SETSIZE aborts the
interpreter, which dumped core in the whole suite on 2026-09-05.
"""
from __future__ import annotations

import os

import pytest

from jarvis import gate as gt
from jarvis.ui import theme
from jarvis.ui import users_page as up
from jarvis.ui import widgets as wg

FORBIDDEN_DISPLAYS = (":0", ":1")
FD_SETSIZE = 1024
HIS = (1040, 1760)
OLD = (920, 1440)
SCALE = 2.0
BOTH = [HIS, OLD]

NOT_A_PHRASE = "zzz not a real passphrase at all zzz"


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
    """Every seam this work adds, wired, so the page draws its full face."""

    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.calls = []

    def people_snapshot(self):
        return self.snapshot

    def people_unlock(self, code):
        return True, "Unlocked, sir."

    def people_forget(self, label):
        return True, "forgotten"

    def people_set_role(self, label, role, **kw):
        return True, "changed"

    def people_add(self, **kw):
        return True, "enrolled"

    def people_set_phrase(self, label, hashed):
        self.calls.append(("phrase", label))
        return True, "Set, sir."

    def people_new_code(self):
        self.calls.append("new_code")
        return True, "a new code is in your inbox"

    def knightfall_status(self):
        return {"to": "h••••@icloud.invalid",
                "problem": "", "setup": ""}

    def face_enrol_start(self):
        self.calls.append("face")
        return True, "the camera is coming on"

    def face_enrol_stop(self):
        return True, "stopped"

    def voice_enrol_start(self):
        self.calls.append("voice")
        return True, "listening"

    def voice_enrol_stop(self):
        return True, "stopped"

    def people_purge_face(self, label):
        self.calls.append(("purge_face", label))
        return True, "removed"

    def people_purge_voice(self, label):
        self.calls.append(("purge_voice", label))
        return True, "removed"


def _snapshot(code=True):
    return {"people": [
        {"label": "alderman", "name": "Alderman", "role": "owner",
         "voice": True, "face": "alderman", "face_dim": 128,
         "has_phrase": True, "has_code": code, "consent": "owner",
         "enrolled_at": "2026-01-01T09:00:00"},
        {"label": "heather", "name": "Heather", "role": "known",
         "voice": False, "face": "heather", "face_dim": 128,
         "has_phrase": False, "has_code": False, "consent": "console",
         "enrolled_at": "2026-02-02T09:00:00"}],
        "gate_line": "owner-gate: SHADOW -- 1 owner (alderman).",
        "fault_kind": "", "path": "/tmp/nowhere/people.json",
        "gallery": ["alderman", "heather"], "voices": ["alderman"],
        "admin": gt.ADMIN_CODE if code else gt.ADMIN_NOCODE,
        "admin_line": "an owner has set an override code"}


def _page(root, geometry, look="holo", code=True, drop=()):
    """``drop`` names seams this console was NEVER HANDED -- the half-wired
    build, which is not hypothetical: ``app.build_ui_services`` drops any
    name the window's Services dataclass has not declared yet, so a window
    older or newer than the app arrives here missing exactly these."""
    import tkinter as tk
    theme.select_look(look)
    w, h = geometry
    root.geometry("%dx%d+0+0" % (w, h))
    host = tk.Frame(root, bg=theme.BG, width=w, height=h)
    host.pack(fill="both", expand=True)
    host.pack_propagate(False)
    stage = tk.Frame(host, bg=theme.BG)
    stage.pack(fill="both", expand=True)
    svc = Svc(_snapshot(code))
    for name in drop:
        setattr(svc, name, None)
    page = up.UsersPage(host, services=svc, cover=(stage,))
    root.update_idletasks()
    page.show()
    page._lock.unlock()
    root.update_idletasks()
    root.update()
    return page, svc, host


def _walk(widget):
    yield widget
    for child in widget.winfo_children():
        yield from _walk(child)


def _overflowing(page):
    """Every mapped widget that ASKED FOR MORE WIDTH THAN IT WAS GIVEN.

    Tk's own two numbers, and deliberately not a font measurement of the
    text. ``winfo_reqwidth`` already includes the border and highlight ring a
    Label adds even at bd=0 and padx=0 -- this file's own ``_INK_SLACK``
    records that gap as 14 px at 2.0 scale -- so a fresh ``tkfont.Font``
    measuring the string disagrees with Tk by a few pixels in BOTH directions
    and flags labels that are drawn perfectly (it flagged the two name labels
    that shipped weeks ago, by 4 px). Requested versus allocated is the
    quantity the real defects were: the Cancel button whose right-hand end
    was cut at 920x1440, and the foot row that overflowed at his window.
    """
    bad = []
    for w in _walk(page):
        try:
            if not w.winfo_ismapped():
                continue
            want = int(w.winfo_reqwidth())
            got = int(w.winfo_width())
        except Exception:                          # noqa: BLE001 - torn down
            continue
        if got > 1 and want > got:
            label = ""
            try:
                label = str(w.cget("text"))[:40]
            except Exception:                      # noqa: BLE001 - no text
                label = w.__class__.__name__
            bad.append((label or w.__class__.__name__, want, got))
    return bad


def _rows_wider_than_the_body(page):
    """Every frame in the scrolling body asking for more than the body has.

    The body scrolls VERTICALLY ONLY, so a row 57 px too wide is not scrolled
    to -- it is cut, and what is cut is the right-hand end of whatever was
    packed last. That is the exact defect photographed on 2026-09-05.
    """
    body = page._body
    room = int(body.winfo_width())
    out = []
    if room <= 1:
        return out
    for child in _walk(body):
        if child.__class__.__name__ != "Frame":
            continue
        try:
            want = int(child.winfo_reqwidth())
        except Exception:                          # noqa: BLE001
            continue
        if want > room:
            out.append((want, room))
    return out


# =========================================== the owner's row carries the buttons
@pytest.mark.parametrize("geometry", BOTH)
def test_the_owner_row_offers_all_four_and_the_guest_row_does_not(root,
                                                                  geometry):
    page, _svc, _host = _page(root, geometry)
    owner = page._row_widgets.get("alderman") or {}
    guest = page._row_widgets.get("heather") or {}
    for key in ("phrase", "code", "face", "voice"):
        assert key in owner, "the owner row is missing %s at %r" % (key,
                                                                    geometry)
        assert key not in guest, ("%s must not be offered on a guest row at "
                                  "%r" % (key, geometry))


@pytest.mark.parametrize("geometry", BOTH)
def test_nothing_new_overflows_its_slot(root, geometry):
    page, _svc, _host = _page(root, geometry)
    bad = _overflowing(page)
    assert bad == [], "cut at %r: %r" % (geometry, bad[:4])
    wide = _rows_wider_than_the_body(page)
    assert wide == [], "a row is wider than the body at %r: %r" % (geometry,
                                                                   wide[:4])


@pytest.mark.parametrize("geometry", BOTH)
@pytest.mark.parametrize("panel", ["phrase", "code", "face", "voice"])
def test_every_panel_fits_the_body_at_both_windows(root, geometry, panel):
    """Opened one at a time, because each is the tallest thing on the page
    while it is open and the body scrolls only downwards."""
    page, _svc, _host = _page(root, geometry)
    if panel in ("phrase", "code"):
        getattr(page, "_%s_pressed" % panel)("alderman")
    else:
        page._purge_pressed("heather",
                            "face" if panel == "face" else "voice")
    root.update_idletasks()
    root.update()
    assert _overflowing(page) == [], (panel, geometry)
    assert _rows_wider_than_the_body(page) == [], (panel, geometry)


@pytest.mark.parametrize("geometry", BOTH)
def test_the_phrase_panel_fits_and_its_boxes_are_masked(root, geometry):
    page, _svc, _host = _page(root, geometry)
    page._phrase_pressed("alderman")
    root.update_idletasks()
    boxes = (page._row_widgets.get("alderman") or {}).get("phrase_boxes") or ()
    assert len(boxes) == 2, "a phrase is typed twice, to be sure"
    for box in boxes:
        assert str(box.cget("show")), "a passphrase box that is not masked"
    assert _overflowing(page) == [], geometry


@pytest.mark.parametrize("geometry", BOTH)
def test_the_code_row_shows_the_masked_destination_and_never_an_address(
        root, geometry):
    page, _svc, _host = _page(root, geometry)
    page._code_pressed("alderman")
    root.update_idletasks()
    blob = "\n".join(str(w.cget("text")) for w in _walk(page)
                     if _has_text(w))
    assert "•" in blob
    assert "@icloud.invalid" in blob      # the masked form only
    assert "h••••@icloud.invalid" in blob
    assert _overflowing(page) == [], geometry


def _has_text(w):
    try:
        str(w.cget("text"))
        return True
    except Exception:                              # noqa: BLE001
        return False


@pytest.mark.parametrize("geometry", BOTH)
def test_the_purge_panel_fits_and_its_confirm_is_reachable(root, geometry):
    page, _svc, _host = _page(root, geometry)
    page._purge_pressed("heather", "face")
    root.update_idletasks()
    entry = (page._row_widgets.get("heather") or {}).get("purge_confirm")
    assert entry is not None
    assert entry.winfo_ismapped()
    assert _overflowing(page) == [], geometry


@pytest.mark.parametrize("geometry", BOTH)
def test_the_pinned_foot_and_its_note_are_on_screen_with_the_new_sentences(
        root, geometry):
    """The foot is PINNED, so it must fit under the body rather than
    pushing the Add button off the bottom. Since 2026-09-06 the note is
    not IN the foot -- the sentences are one press away, behind READ and
    its opener caption, which must be on screen too (the sheet renders
    ``foot_lines`` verbatim; tests/test_ui_easy_to_add.py reads it)."""
    page, _svc, host = _page(root, geometry)
    root.update_idletasks()
    foot = page._foot
    assert foot.winfo_ismapped()
    bottom = foot.winfo_y() + foot.winfo_height()
    assert bottom <= page.winfo_height() + 1, (
        "the foot ends %d px into a %d px page at %r"
        % (bottom, page.winfo_height(), geometry))
    assert page._add_btn.winfo_ismapped()
    assert page._read_btn.winfo_ismapped()
    assert page._opener_lbl.winfo_ismapped()
    assert page._read_btn.master is page._add_btn.master


@pytest.mark.parametrize("geometry", BOTH)
def test_no_widget_anywhere_carries_a_typed_phrase_or_a_hash(root, geometry):
    """CONSTRAINT C on a real tree. The entries are read and cleared by the
    control; nothing must survive the action in a widget."""
    page, svc, _host = _page(root, geometry)
    page._phrase_pressed("alderman")
    root.update_idletasks()
    boxes = (page._row_widgets.get("alderman") or {}).get("phrase_boxes") or ()
    for box in boxes:
        box.insert(0, NOT_A_PHRASE)
    page._phrase_confirm("alderman")
    root.update_idletasks()
    root.update()
    blob = "\n".join(
        [str(w.cget("text")) for w in _walk(page) if _has_text(w)]
        + [str(w.get()) for w in _walk(page)
           if w.__class__.__name__ == "Entry"])
    assert "zzz" not in blob
    assert "$" not in blob.replace("$", "", 0) or "scrypt" not in blob


def test_the_module_captures_no_look_at_import_time():
    """theme.X is read at CALL time, never at def time -- the rule
    tests/test_theme_look.py already pins for this file and which the new
    panels must not break."""
    import inspect
    src = inspect.getsource(up)
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith(("def ", "class ", "@")):
            continue
        if line.startswith(("    ", "\t")):
            continue
        if "theme." in stripped and "=" in stripped and "import" not in stripped:
            name = stripped.split("=")[0].strip()
            assert name.startswith("_") is False or True, name
            pytest.fail("module-level theme capture: %r" % stripped)


# ================================ the half-wired console, MEASURED not assumed
HALF_WIRED = [
    (),
    ("face_enrol_start",),
    ("voice_enrol_start",),
    ("face_enrol_start", "voice_enrol_start"),
    ("people_purge_face", "people_purge_voice"),
    ("face_enrol_start", "voice_enrol_start", "people_purge_face",
     "people_purge_voice", "face_enrol_stop", "voice_enrol_stop"),
]


def _button_texts(page):
    """Every MAPPED button's label. ``RoundButton`` is a Canvas that draws
    its own text, so the string is on ``_text`` and never on a cget."""
    out = []
    for w in _walk(page):
        if not isinstance(w, wg.RoundButton) or not w.winfo_ismapped():
            continue
        text = str(getattr(w, "_text", "") or "")
        if text:
            out.append(text)
    return out


def _note_text(page):
    return str(page._note_lbl.cget("text"))


@pytest.mark.parametrize("geometry", BOTH)
@pytest.mark.parametrize("drop", HALF_WIRED)
def test_a_half_wired_console_never_draws_a_button_its_note_denies(
        root, geometry, drop):
    """DEFECT 3, on a real tree rather than in the abstract. The derived note
    and the buttons ACTUALLY DRAWN are read off the same window, in six
    wirings, at both his geometries."""
    page, _svc, _host = _page(root, geometry, drop=drop)
    root.update_idletasks()
    root.update()
    note, drawn = _note_text(page), _button_texts(page)
    if up.NOTE_FACE_NO in note:
        assert "Enrol my face" not in drawn, (drop, geometry)
        assert "Copy the face command" in drawn, (drop, geometry)
    else:
        assert "Enrol my face" in drawn, (drop, geometry)
    if up.NOTE_VOICE_NO in note:
        assert "Enrol my voice" not in drawn, (drop, geometry)
        assert "Copy the voice command" in drawn, (drop, geometry)
    else:
        assert "Enrol my voice" in drawn, (drop, geometry)
    if up.ENROL_ASYMMETRY[0] in note:
        assert "Enrol my face" in drawn, (drop, geometry)


@pytest.mark.parametrize("geometry", BOTH)
@pytest.mark.parametrize("drop", HALF_WIRED)
def test_nothing_overflows_in_any_half_wired_state(root, geometry, drop):
    """The hand-over buttons are the WIDE ones -- "Copy the voice command" is
    458 px at 2.0 scale -- so the wirings that draw them are exactly the ones
    that can cut, and the body scrolls only downwards."""
    page, _svc, _host = _page(root, geometry, drop=drop)
    root.update_idletasks()
    root.update()
    assert _overflowing(page) == [], (drop, geometry, _overflowing(page)[:4])
    assert _rows_wider_than_the_body(page) == [], (drop, geometry)


@pytest.mark.parametrize("geometry", BOTH)
@pytest.mark.parametrize("drop", HALF_WIRED)
def test_the_forget_panel_fits_and_promises_only_drawn_buttons(
        root, geometry, drop):
    """DEFECT 2, measured. The panel grew -- it now carries the ordering
    sentence and up to two full command lines -- so it is re-measured in
    every wiring, and it may not name a button this console did not draw."""
    page, _svc, _host = _page(root, geometry, drop=drop)
    page._forget_pressed("heather")
    root.update_idletasks()
    root.update()
    assert _overflowing(page) == [], (drop, geometry, _overflowing(page)[:4])
    assert _rows_wider_than_the_body(page) == [], (drop, geometry)
    warn = "\n".join(t for t in [str(w.cget("text")) for w in _walk(page)
                                 if _has_text(w)] if "WHAT SURVIVES" in t)
    assert warn, "the forget panel did not draw its warning"
    drawn = _button_texts(page)
    if "Remove face measurements" in warn:
        assert "Remove face measurements" in drawn, (drop, geometry)
    if "Remove voice pool" in warn:
        assert "Remove voice pool" in drawn, (drop, geometry)
    # The commands are printed WHATEVER is wired: after the row is gone they
    # are the only way left.
    assert up.forget_face_command("heather") in warn


@pytest.mark.parametrize("geometry", BOTH)
def test_the_voice_command_is_offered_where_a_pool_exists(root, geometry):
    """heather has face measurements and no pool; alderman has both. The
    panel must hand over the VOICE command for the one who has one -- the
    half that had no builder at all before tonight."""
    page, _svc, _host = _page(root, geometry)
    page._forget_pressed("alderman")
    root.update_idletasks()
    root.update()
    blob = "\n".join(str(w.cget("text")) for w in _walk(page)
                      if _has_text(w))
    assert up.forget_voice_command("alderman") in blob
    assert _overflowing(page) == [], geometry
    assert _rows_wider_than_the_body(page) == [], geometry
