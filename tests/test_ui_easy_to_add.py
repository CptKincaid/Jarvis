"""EASY TO ADD A PERSON OR A SENSOR -- measured, at both his windows.

Hunter, 2026-09-06, verbatim: "Double check the uis for users settings and
sensors and make them easy to use for me too add things".

WHAT WAS WRONG, MEASURED on the rig (scripts/ui_shots.py) at jarvis-v3
a4ecf8b, private Xvfb, S=2.0:

* USERS at 920x1440: the pinned foot carried an 18-line note (576 px) and
  the pinned head two 2-line strings, leaving 88 px -- 8% of the page -- for
  the list. With the add form open, 0 of its 5 inputs were on screen. Every
  confirm panel showed its box and its button with its WARNING scrolled
  off. At 1040x1760 the consent box (mandatory) was 712 px below the fold.
* SENSORS -> USERS lit the CHAT tab (both looks, both directions): the
  page's close hook selected CHAT while the strip was mid-switch.
* SENSORS: nothing said "add a sensor"; SETUP and + NEW were bare words.
  On the sheet, what a new sensor NEEDS (address, wi-fi, password, ota code)
  was under a mount paragraph and a gates line, below the fold at 920, and
  the wi-fi caption asked for 531 px of a 449 px slot.
* SETTINGS: 3718 px of drawer in a 1329 px viewport with no mark and no
  keyboard -- Privacy, Knightfall and Restart all below the fold.

Every test here is a NUMBER off a real Tk tree: nothing is looked at.
Runs only when ``JARVIS_UI_TEST_DISPLAY`` names a PRIVATE Xvfb; his desktop
displays are refused outright. No registry, gallery or config of his is
opened: every snapshot is invented and handed through the services seam.
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
SCALE = 2.0
HIS, OLD = (1040, 1760), (920, 1440)
BOTH = [HIS, OLD]
# The stage the USERS page actually covers in the console at each window,
# MEASURED off the rig (the tab strip and header above, the command bar and
# status strip below): NOT the whole window, which is what the older tests
# used and which is 400-700 px more generous than the truth.
USERS_STAGE_H = {920: 1044, 1040: 1364}
# The SENSORS page with the camera pane packed (tests/test_ui_layout_rules.py
# measured 852 at 920; 1172 is the same page at his own default geometry).
SENSORS_STAGE_H = {920: 852, 1040: 1172}
FONT_GLOBALS = ("_FAMILY", "_FAMILY_MONO", "_HAS_DISPLAY", "_DISPLAY")


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
                    "the measured easy-to-add tests")
    import tkinter as tk
    try:
        fds = len(os.listdir("/proc/self/fd"))
    except OSError:
        fds = 0
    if fds >= FD_SETSIZE - 32:
        pytest.skip("this process already holds %d descriptors" % fds)
    fonts = {k: getattr(theme, k) for k in FONT_GLOBALS}
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
    for k, v in fonts.items():
        setattr(theme, k, v)


# ================================================================ helpers
def _walk(widget):
    yield widget
    for child in widget.winfo_children():
        yield from _walk(child)


def _inside(widget, box) -> bool:
    """Is ``widget`` wholly inside ``box`` (a widget), by root geometry?"""
    top, bottom = box.winfo_rooty(), box.winfo_rooty() + box.winfo_height()
    y0 = widget.winfo_rooty()
    y1 = y0 + widget.winfo_height()
    return bool(widget.winfo_ismapped()) and y0 >= top and y1 <= bottom


def _overflowing(top):
    """Every mapped widget asking for more width than it was given, and
    every label wrapped wider than its own slot -- the two shapes of "cut
    at the right edge" this tree keeps producing."""
    bad = []
    for w in _walk(top):
        try:
            if not w.winfo_ismapped():
                continue
            want, got = int(w.winfo_reqwidth()), int(w.winfo_width())
        except Exception:                          # noqa: BLE001 - torn down
            continue
        if got <= 1:
            continue
        label = w.__class__.__name__
        try:
            label = str(w.cget("text"))[:40] or label
        except Exception:                          # noqa: BLE001 - no text
            label = str(getattr(w, "_text", "") or label)
        if want > got:
            bad.append(("req", label, want, got))
        try:
            wrap = int(w.cget("wraplength"))
        except Exception:                          # noqa: BLE001 - not a Label
            wrap = 0
        if wrap and wrap > got:
            bad.append(("wrap", label, wrap, got))
    return bad


def _buttons(top):
    return [w for w in _walk(top)
            if isinstance(w, wg.RoundButton) and w.winfo_ismapped()]


def _bare(top):
    """Mapped RoundButtons drawn with NO outline -- a bare word."""
    return [b._text for b in _buttons(top) if not b._spec.get("outline")]


# ============================================================ the USERS page
class Svc:
    """Every seam the app wires today, so the page draws its full face."""

    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.calls = []

    def people_snapshot(self):
        return self.snapshot

    def people_unlock(self, code):
        return True, "Unlocked, sir."

    def people_forget(self, label):
        self.calls.append(("forget", label))
        return True, "forgotten"

    def people_set_role(self, label, role, **kw):
        return True, "changed"

    def people_add(self, **kw):
        self.calls.append(("add", kw))
        return True, "%s is enrolled" % kw.get("label")

    def people_set_phrase(self, label, hashed):
        return True, "Set, sir."

    def people_new_code(self):
        return True, "a new code is in your inbox"

    def knightfall_status(self):
        return {"to": "h••••@icloud.invalid", "problem": "", "setup": ""}

    def face_enrol_start(self):
        return True, "the camera is coming on"

    def face_enrol_stop(self):
        return True, "stopped"

    def voice_enrol_start(self):
        return True, "listening"

    def voice_enrol_stop(self):
        return True, "stopped"

    def people_purge_face(self, label):
        return True, "removed"

    def people_purge_voice(self, label):
        return True, "removed"


def _snapshot(n=3, *, mode="SHADOW", names=None, code=True):
    """``n`` invented people: one owner and ``n-1`` known guests."""
    names = list(names or ())
    people = [{"label": "alderman", "name": "Alderman", "role": "owner",
               "voice": True, "face": "alderman", "face_dim": 128,
               "has_phrase": True, "has_code": code, "consent": "owner",
               "enrolled_at": "2026-01-01T09:00:00"}]
    for i in range(max(0, n - 1)):
        name = names[i] if i < len(names) else "Guest %d" % i
        people.append({"label": name.lower().replace(" ", ""),
                       "name": name, "role": "known", "voice": False,
                       "face": name.lower().replace(" ", ""),
                       "face_dim": 128, "has_phrase": False,
                       "has_code": False, "consent": "console",
                       "enrolled_at": "2026-02-02T09:00:00"})
    return {"people": people,
            "gate_line": ("owner-gate: %s -- 1 owner (alderman), voice leg "
                          "live, face leg unavailable (the gallery is "
                          "empty). Nothing is being refused." % mode),
            "fault_kind": "", "path": "/tmp/nowhere/people.json",
            "gallery": [p["label"] for p in people],
            "voices": ["alderman"],
            "admin": gt.ADMIN_CODE if code else gt.ADMIN_NOCODE,
            "admin_line": "an owner has set an override code"}


def _page(root, geometry, look="holo", snapshot=None, drop=()):
    """The page over a host of exactly the STAGE it covers in the console."""
    import tkinter as tk
    theme.select_look(look)
    w, h = geometry
    root.geometry("%dx%d+0+0" % (w, h))
    stage_h = USERS_STAGE_H[w]
    host = tk.Frame(root, bg=theme.BG, width=w, height=stage_h)
    host.pack(fill="x")
    host.pack_propagate(False)
    stage = tk.Frame(host, bg=theme.BG)
    stage.pack(fill="both", expand=True)
    svc = Svc(_snapshot() if snapshot is None else snapshot)
    for name in drop:
        setattr(svc, name, None)
    page = up.UsersPage(host, services=svc, cover=(stage,))
    root.update_idletasks()
    page.show()
    page._lock.unlock()
    root.update_idletasks()
    root.update()
    assert page.winfo_height() == stage_h, (page.winfo_height(), stage_h)
    return page, svc, host


def _settle(root):
    root.update_idletasks()
    root.update()


# ---------------------------------------------- 1: the list gets the page
@pytest.mark.parametrize("look", ["holo", "classic"])
@pytest.mark.parametrize("geometry", BOTH)
def test_the_list_viewport_is_most_of_the_page(root, geometry, look):
    """MEASURED at a4ecf8b: 88 px of a 1044-px page at 920x1440 (8%), 504
    of 1364 at his window (37%). The single biggest lever on the page is the
    18-line note pinned in the foot; with it moved to a sheet the list gets
    >= 700 px at 920 and >= 1000 at 1040, with three people on the page."""
    page, _svc, _host = _page(root, geometry, look)
    view = page._canvas.winfo_height()
    want = 700 if geometry == OLD else 1000
    assert view >= want, ("the list viewport is %d px of a %d px page at "
                          "%r (%s)" % (view, page.winfo_height(), geometry,
                                       look))


# ------------------------------------- 2: the add form shows its own boxes
@pytest.mark.parametrize("look", ["holo", "classic"])
@pytest.mark.parametrize("geometry", BOTH)
def test_every_add_form_input_is_inside_the_viewport_on_its_step(
        root, geometry, look):
    """THE THING HE ASKED FOR. At 920x1440 on a4ecf8b, 0 of the 5 inputs
    were on screen with the form open; at 1040x1760 three were, and the
    consent box -- which Create refuses without -- sat 712 px below the
    fold. Now: step 1 (who) shows every box; step 2 (their agreement) shows
    the whole paragraph AND the box he types into, on one screen."""
    page, _svc, _host = _page(root, geometry, look)
    page._add_pressed()
    _settle(root)
    view = page._canvas
    fields = page._add_fields
    shown = {k: e for k, e in fields.items() if e.winfo_ismapped()}
    assert {"name", "label", "face"} <= set(shown), sorted(shown)
    assert "consent" not in shown, "step 1 asks WHO; the agreement is step 2"
    for key, entry in shown.items():
        assert _inside(entry, view), ("the %s box is not inside the %d-px "
                                      "viewport at %r" % (key, view.winfo_height(),
                                                          geometry))
    assert _inside(page._next_btn, page) and not page._create_btn.winfo_ismapped()
    fields["label"].insert(0, "pemberton")
    fields["name"].insert(0, "Pemberton")
    page._next_pressed()
    _settle(root)
    assert page._add_step == 2, page._toasts[-1:]
    assert fields["consent"].winfo_ismapped()
    assert _inside(fields["consent"], view), "the consent box is off screen"
    assert _inside(page._consent_lbl, view), (
        "the consent paragraph (%d px) is not whole inside the %d-px "
        "viewport at %r" % (page._consent_lbl.winfo_height(),
                            view.winfo_height(), geometry))
    assert "pemberton" in page._consent_lbl.cget("text")
    assert page._create_btn.winfo_ismapped() and _inside(page._create_btn, page)
    assert page.focus_lastfor() is fields["consent"]


@pytest.mark.parametrize("geometry", BOTH)
def test_the_first_owner_form_is_one_step_and_fits(root, geometry):
    snap = _snapshot(0)
    snap["people"], snap["fault_kind"] = [], "missing"
    snap["admin"] = gt.ADMIN_FIRST
    page, _svc, _host = _page(root, geometry, snapshot=snap)
    page._add_pressed()
    _settle(root)
    assert page._add_owner is True
    assert "consent" not in page._add_fields
    assert not page._next_btn.winfo_ismapped()
    assert page._create_btn.winfo_ismapped()
    for key, entry in page._add_fields.items():
        assert _inside(entry, page._canvas), (key, geometry)


@pytest.mark.parametrize("geometry", BOTH)
def test_a_refused_step_names_the_box_and_puts_the_cursor_in_it(root,
                                                                geometry):
    """A toast alone vanishes before he has read it: the offending box gets
    the focus and an amber caption under it."""
    page, svc, _host = _page(root, geometry)
    page._add_pressed()
    _settle(root)
    page._add_fields["label"].insert(0, "Pemberton Smythe")
    page._next_pressed()
    _settle(root)
    assert page._add_step == 1
    assert page.focus_lastfor() is page._add_fields["label"]
    amber = [w for w in _walk(page._body)
             if w.__class__.__name__ == "Label" and w.winfo_ismapped()
             and str(w.cget("fg")) == str(theme.WARN)]
    assert amber and "label" in amber[0].cget("text").lower()
    # ...and a wrong agreement on step 2 does the same for the consent box
    page._add_fields["label"].delete(0, "end")
    page._add_fields["label"].insert(0, "pemberton")
    page._next_pressed()
    _settle(root)
    assert page._add_step == 2
    page._add_fields["consent"].insert(0, "yes")
    page._create_pressed()
    _settle(root)
    assert not any(c[0] == "add" for c in svc.calls)
    assert page.focus_lastfor() is page._add_fields["consent"]


@pytest.mark.parametrize("geometry", BOTH)
def test_their_own_label_on_step_two_creates_them(root, geometry):
    page, svc, _host = _page(root, geometry)
    page._add_pressed()
    _settle(root)
    page._add_fields["label"].insert(0, "pemberton")
    page._add_fields["name"].insert(0, "Pemberton")
    page._next_pressed()
    _settle(root)
    page._add_fields["consent"].insert(0, "pemberton")
    page._create_pressed()
    _settle(root)
    added = [c for c in svc.calls if c[0] == "add"]
    assert added and added[0][1]["consent"] == "console"
    assert page._adding is False


def test_the_agreement_cannot_be_given_before_it_has_been_shown(root):
    """The paragraph is READ before it is typed to: Create refuses while the
    words are not on screen, even with the right label in the box."""
    page, svc, _host = _page(root, OLD)
    page._add_pressed()
    _settle(root)
    page._add_fields["label"].insert(0, "pemberton")
    page._add_fields["consent"].insert(0, "pemberton")
    page._create_pressed()
    assert not any(c[0] == "add" for c in svc.calls)


# ------------------------------------------ 3: a thing that acts is a button
@pytest.mark.parametrize("geometry", BOTH)
def test_every_acting_button_wears_a_border_in_holo(root, geometry):
    """MEASURED at a4ecf8b: 1 of 17 buttons on the page was outlined. To
    him a bare word is not a button. Every state the page reaches."""
    page, _svc, _host = _page(root, geometry, "holo")
    seen = set()
    for state in ("list", "forget", "purge", "phrase", "code", "add", "agree",
                  "role"):
        if state == "forget":
            page._forget_pressed("guest0")
        elif state == "purge":
            page._cancel()
            page._purge_pressed("guest0", "face")
        elif state == "phrase":
            page._cancel()
            page._phrase_pressed("alderman")
        elif state == "code":
            page._cancel()
            page._code_pressed("alderman")
        elif state == "add":
            page._cancel()
            page._add_pressed()
        elif state == "agree":
            page._add_fields["label"].insert(0, "pemberton")
            page._next_pressed()
        elif state == "role":
            page._cancel()
            page._role_pressed("guest0", "known")
        _settle(root)
        assert _bare(page) == [], (state, geometry, _bare(page))
        seen.update(b._text for b in _buttons(page))
    for word in ("Unlock", "Lock", "Forget", "Make owner", "Make known",
                 "Cancel", "Set", "Copy the face command", "READ", "DETAILS",
                 "Create", "Remove face measurements", "Enrol my face"):
        assert word in seen, (word, sorted(seen))


# ------------------------------- 4: the owner's row is whole at the narrow one
@pytest.mark.parametrize("geometry", BOTH)
def test_the_owner_rows_chips_and_buttons_are_inside_the_viewport(root,
                                                                  geometry):
    """MEASURED at a4ecf8b, 920x1440: the chips line was 32% visible and all
    nine owner buttons sat at y 675-951 against a viewport of 472-560."""
    page, _svc, _host = _page(root, geometry)
    owner = page._row_widgets["alderman"]
    view = page._canvas
    for key, w in owner.items():
        if isinstance(w, wg.RoundButton):
            assert _inside(w, view), (key, geometry)
    chips = [w for w in _walk(owner["block"])
             if w.__class__.__name__ == "Label"
             and "phrase: set" in str(w.cget("text"))]
    assert chips and _inside(chips[0], view), geometry
    assert len([w for w in owner.values()
                if isinstance(w, wg.RoundButton)]) >= 9


# --------------------------------------------- 5: nothing runs off the edge
@pytest.mark.parametrize("geometry", BOTH)
def test_no_label_or_button_asks_for_more_width_than_it_was_given(
        root, geometry):
    """With a 12-letter name: "Make owner" was cut 17 px at 920 on a
    10-letter one (a4ecf8b), and longer names cut more."""
    snap = _snapshot(3, names=["Featherstone", "Marchbanks"])
    page, _svc, _host = _page(root, geometry, snapshot=snap)
    for state in ("list", "forget", "add", "agree", "phrase", "purge"):
        if state == "forget":
            page._forget_pressed("featherstone")
        elif state == "add":
            page._cancel()
            page._add_pressed()
        elif state == "agree":
            page._add_fields["label"].insert(0, "pemberton")
            page._next_pressed()
        elif state == "phrase":
            page._cancel()
            page._phrase_pressed("alderman")
        elif state == "purge":
            page._cancel()
            page._purge_pressed("marchbanks", "face")
        _settle(root)
        bad = _overflowing(page)
        assert bad == [], (state, geometry, bad[:4])
    # the name is what gives way, and it says so
    names = [str(w.cget("text")) for w in
             _walk(page._row_widgets["featherstone"]["block"])
             if w.__class__.__name__ == "Label"
             and str(w.cget("text")).startswith("Feather")]
    assert names, "the person's name is gone from the row"
    assert names[0] == "Featherstone" or names[0].endswith("\u2026"), names


# ---------------------- 6: a confirm panel is read whole before it is typed to
PANELS = ["forget", "purge_face", "purge_voice", "phrase"]


@pytest.mark.parametrize("panel", PANELS)
@pytest.mark.parametrize("geometry", BOTH)
def test_each_confirm_panel_shows_its_question_its_box_and_its_buttons(
        root, geometry, panel):
    """MEASURED at a4ecf8b, 920x1440: the forget warning (672 px) was 0%
    visible, the purge warning (512 px) 0%, both passphrase boxes hidden --
    he could confirm a destructive action without the warning ever on
    screen. At his own 1040 the forget panel opened with its QUESTION above
    the viewport (48% visible)."""
    page, _svc, _host = _page(root, geometry)
    if panel == "forget":
        page._forget_pressed("guest1")
        who = "guest1"
    elif panel == "purge_face":
        page._purge_pressed("guest1", "face")
        who = "guest1"
    elif panel == "purge_voice":
        page._purge_pressed("alderman", "voice")
        who = "alderman"
    else:
        page._phrase_pressed("alderman")
        who = "alderman"
    _settle(root)
    top, bottom = page._row_widgets[who]["panel"]
    view = page._canvas
    assert _inside(top, view), (
        "%s: the question (%d px tall) is not inside the %d-px viewport at "
        "%r" % (panel, top.winfo_height(), view.winfo_height(), geometry))
    assert _inside(bottom, view), (panel, geometry)
    boxes = [w for w in _walk(bottom) if w.__class__.__name__ == "Entry"]
    buttons = [w for w in _walk(bottom) if isinstance(w, wg.RoundButton)]
    if panel == "phrase":
        boxes = list(page._row_widgets[who]["phrase_boxes"])
    assert boxes and buttons
    for w in boxes + buttons:
        assert _inside(w, view), (panel, geometry)


# ------------------------------------------------ 7: the gate line is HIS
@pytest.mark.parametrize("n", [1, 3, 12])
@pytest.mark.parametrize("mode", ["OFF", "SHADOW", "ENFORCE"])
def test_the_plain_gate_line_is_one_line_at_the_narrow_window(root, mode, n):
    """"owner-gate: SHADOW -- 1 owner (alderman), voice leg live, face leg
    unavailable (the gallery is empty). Nothing is being refused." is for
    the log. He gets one plain line; the raw string stays behind DETAILS."""
    import tkinter.font as tkfont
    page, _svc, _host = _page(root, OLD, snapshot=_snapshot(n, mode=mode))
    lbl = page._gate_lbl
    line_h = tkfont.Font(font=lbl.cget("font")).metrics("linespace")
    assert lbl.winfo_reqheight() <= line_h + wg.px(2), (
        mode, n, lbl.cget("text"), lbl.winfo_reqheight(), line_h)
    assert lbl.winfo_reqwidth() <= lbl.winfo_width(), (mode, n,
                                                       lbl.cget("text"))
    text = lbl.cget("text").lower()
    assert "owner-gate" not in text and "leg" not in text
    assert "recognition" in text
    assert ("known" in text) or ("nobody" in text)
    # the diagnostic is one press away and is the app's own string
    assert "owner-gate: %s" % mode in page.gate_detail_text()
    assert page._details_btn.winfo_ismapped()
    assert not page._gate_raw_lbl.winfo_ismapped()
    page._details_pressed()
    _settle(root)
    assert page._gate_raw_lbl.winfo_ismapped()
    assert "owner-gate: %s" % mode in page._gate_raw_lbl.cget("text")
    assert page._gate_raw_lbl.winfo_reqwidth() <= page.winfo_width()
    # the people-file path lives here, not in the foot
    assert "/tmp/nowhere/people.json" in page._path_lbl.cget("text")
    assert page._path_lbl.master is not page._act_row


# --------------------------------------- 8: the nine sentences, reachable
WIRINGS = {
    "all": (),
    "none": ("people_set_phrase", "people_new_code", "face_enrol_start",
             "voice_enrol_start", "people_purge_face", "people_purge_voice",
             "face_enrol_stop", "voice_enrol_stop"),
    "face-only": ("people_set_phrase", "people_new_code", "voice_enrol_start",
                  "people_purge_face", "people_purge_voice",
                  "voice_enrol_stop"),
}


@pytest.mark.parametrize("wiring", sorted(WIRINGS))
@pytest.mark.parametrize("geometry", BOTH)
def test_the_nine_sentences_are_one_press_away_and_are_the_derived_tuple(
        root, geometry, wiring):
    """Moved, not deleted, and still DERIVED from the seams: the sheet
    renders foot_lines(services) and nothing else -- a hand-copied sheet
    would be the 09-05 stale-note defect reborn."""
    page, svc, _host = _page(root, geometry, drop=WIRINGS[wiring])
    want = "\n".join("· " + line for line in up.foot_lines(svc))
    assert not page._note_sheet.winfo_ismapped()
    assert page._read_btn.winfo_ismapped() and _inside(page._read_btn, page)
    assert page._opener_lbl.winfo_ismapped()
    assert "cannot do" in page._opener_lbl.cget("text").lower()
    page._read_btn.invoke()
    _settle(root)
    assert page._note_sheet.winfo_ismapped()
    assert page._note_lbl.cget("text") == want
    assert page._note_lbl.winfo_ismapped()
    close = page._note_close_btn
    assert _inside(close, page), (geometry, wiring)
    over = page._note_sheet.overflow_px()
    assert over >= 0
    if over > 0:
        assert page._note_thumb.winfo_ismapped()
        assert page._note_canvas.winfo_children()[-1] is page._note_thumb
    assert _overflowing(page._note_sheet) == []
    # the first and last sentence are the two that never derive away
    assert page._note_lbl.cget("text").startswith("· " + up.NOTE_NOT_A_LOCK)
    assert page._note_lbl.cget("text").endswith(up.NOTE_NO_UNDO) or \
        up.NOTE_NO_UNDO in page._note_lbl.cget("text")
    close.invoke()
    _settle(root)
    assert not page._note_sheet.winfo_ismapped()


def test_a_capability_landing_repaints_the_sheet_while_it_is_shut(root):
    page, svc, _host = _page(root, OLD, drop=("face_enrol_start",))
    assert up.NOTE_FACE_NO in page._note_lbl.cget("text")
    svc.face_enrol_start = lambda: (True, "on")
    page.refresh()
    _settle(root)
    assert up.NOTE_FACE_NO not in page._note_lbl.cget("text")
    assert up.ENROL_ASYMMETRY[0] in page._note_lbl.cget("text")


# ------------------------------------ 9: leaving the tab forgets everything
def test_leaving_the_tab_shuts_the_sheet_the_details_and_the_form(root):
    page, _svc, _host = _page(root, OLD)
    page._details_pressed()
    page._read_btn.invoke()
    page._note_close_btn.invoke()
    page._add_pressed()
    page._add_fields["label"].insert(0, "pemberton")
    page._next_pressed()
    page._read_btn.invoke()
    _settle(root)
    assert page._add_step == 2 and page._note_sheet.winfo_ismapped()
    page.hide()
    page.show()
    _settle(root)
    assert page._adding is False and page._add_step == 1
    assert not page._note_sheet.winfo_ismapped()
    assert not page._gate_raw_lbl.winfo_ismapped()


# ======================================================== the TAB STRIP
class _Page:
    """A stand-in surface with the shipping close hook wired the way
    main_window wires the real pages."""

    def __init__(self, closed):
        self.is_open = False
        self._closed = closed

    def show(self):
        self.is_open = True

    def hide(self):
        if not self.is_open:
            return
        self.is_open = False
        self._closed()


@pytest.mark.parametrize("look", ["holo", "classic"])
def test_sensors_then_users_lights_users_and_back_again(root, look):
    """CONFIRMED AS A REAL DEFECT at a4ecf8b, both looks, both directions:
    SENSORS -> USERS left the strip on CHAT with the Users page open. The
    page's close hook called strip.select("chat") while the strip was
    mid-switch. Driven through the SHIPPING hooks on a stand-in window."""
    from jarvis.ui import main_window as mw
    from jarvis.ui.tab_strip import TabStrip
    theme.select_look(look)
    strip = TabStrip(root, bg=theme.BG)
    strip.pack(fill="x")

    class _Win:
        pass

    win = _Win()
    win.tabs = strip
    win.users = _Page(lambda: mw.MainWindow._users_closed(win))
    win.sensors = _Page(lambda: mw.MainWindow._sensors_closed(win))
    strip.add("chat", "CHAT")
    strip.add("sensors", "SENSORS", select=win.sensors.show,
              leave=win.sensors.hide)
    strip.add("users", "USERS", select=win.users.show, leave=win.users.hide)
    root.update_idletasks()

    def lit():
        return [k for k in strip.keys if strip.widget(k).selected]

    strip.select("sensors")
    assert strip.selected == "sensors" and win.sensors.is_open
    strip.select("users")
    assert strip.selected == "users", (look, strip.selected)
    assert lit() == ["users"] and win.users.is_open and not win.sensors.is_open
    strip.select("sensors")
    assert strip.selected == "sensors" and lit() == ["sensors"]
    assert win.sensors.is_open and not win.users.is_open
    # a page hiding ITSELF still puts the strip back on CHAT
    win.sensors.hide()
    assert strip.selected == "chat" and lit() == ["chat"]
    # ...and standby's own CHAT press is still the idempotent no-op
    strip.select("users")
    strip.select("chat")
    assert strip.selected == "chat" and not win.users.is_open


# ====================================================== the SENSORS page
class _Cfg:
    """get_option / set_option over a dict. No file, no app."""

    def __init__(self, rooms=True):
        self.data = {
            "presence": {"room_sensor_enabled": True, "rooms": (
                [{"name": "office", "url": "http://192.0.2.10/binary_sensor/"
                                           "Presence", "primary": True}]
                if rooms else [])},
            "zones": {"enabled": True, "rooms": [
                {"name": "office", "enabled": True, "camera_zone": "",
                 "bands": [{"name": "at the desk", "near_m": 0.75,
                            "far_m": 3.5}]}] if rooms else []}}
        self.sensing = None

    def get_option(self, key, default=None):
        node = self.data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_option(self, key, value):
        node = self.data
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
        return True


def _sensors(root, geometry, tmp_path, look="holo", rooms=True):
    import tkinter as tk
    from jarvis.ui import sensors_page as sp
    theme.select_look(look)
    w, h = geometry
    root.geometry("%dx%d+0+0" % (w, h))
    stage = SENSORS_STAGE_H[w]
    host = tk.Frame(root, bg=theme.BG, width=w, height=stage)
    host.pack_propagate(False)
    host.pack(fill="x")
    page = sp.SensorsPage(host, services=_Cfg(rooms), profile_dir=tmp_path)
    page.place(in_=host, x=0, y=0, relwidth=1.0, height=stage)
    root.update_idletasks()
    return page, host


@pytest.mark.parametrize("geometry", BOTH)
def test_the_sensors_page_has_a_bordered_control_that_says_add_a_sensor(
        root, geometry, tmp_path):
    """MEASURED at a4ecf8b: SETUP was kind ghost (no outline) and nothing on
    the page said "add". The button's label says what it does, it is
    outlined, and pressing it lands on the NEW-ROOM form with the room box
    focused. The row still fits at 920."""
    page, host = _sensors(root, geometry, tmp_path)
    btn = page.setup_btn
    assert btn is not None and btn.winfo_ismapped()
    words = str(btn._text).upper()
    assert "ADD" in words and "SENSOR" in words, words
    assert btn._spec.get("outline"), "a bare word is not a button to him"
    assert page._save_btn._spec.get("outline")
    row = btn.master
    used = sum(c.winfo_reqwidth() for c in row.winfo_children())
    budget = geometry[0] - 2 * theme.PAD
    assert used <= budget, (used, budget, geometry)
    # the foot row and every button on the page, whole
    foot = [b for b in _overflowing(page._foot)]
    assert foot == [], foot
    for b in _buttons(page):
        assert b.winfo_width() >= b.winfo_reqwidth(), b._text
    btn.invoke()
    _settle(root)
    sheet = page.setup
    assert sheet is not None and sheet.is_open
    assert sheet._room == "", "ADD A SENSOR opens the new-room form"
    assert sheet.focus_lastfor() is sheet._field["room"]
    assert str(sheet._field["room"].cget("state")) == "normal"
    sheet.hide()


def test_the_empty_sensors_page_says_to_press_the_add_control(root, tmp_path):
    from jarvis.ui import sensors_page as sp
    page, host = _sensors(root, OLD, tmp_path, rooms=False)
    line = sp.empty_state_line(page._get_option)
    assert "ADD A SENSOR" in line and "SETUP" not in line
    labels = [w for w in _walk(page) if w.__class__.__name__ == "Label"
              and "ADD A SENSOR" in str(w.cget("text"))]
    assert labels


# ========================================================= the SETUP sheet
def _write_profile(directory, room):
    """An INVENTED profile, through the package's own writer, into the
    test's own directory. His own profiles are never opened."""
    from jarvis import sensorprofile
    sensorprofile.write(room, {"ssid": "PRETEND-NET-5G", "ip": "192.0.2.10",
                               "gateway": "192.0.2.1",
                               "subnet": "255.255.255.0", "preset": "desk",
                               "nearest_m": 2.0, "range_m": 3.5,
                               "still": True, "timeout_s": 10,
                               "flashed": True},
                        password="invented-psk-0000",
                        ota_password="invented-ota-0000",
                        directory=directory)


def _sheet(root, geometry, tmp_path):
    import tkinter as tk
    from jarvis.ui import sensor_setup as ss
    theme.select_look("holo")
    _write_profile(tmp_path, "office")
    w, h = geometry
    root.geometry("%dx%d+0+0" % (w, h))
    stage = SENSORS_STAGE_H[w]
    host = tk.Frame(root, bg=theme.BG, width=w, height=stage)
    host.pack_propagate(False)
    host.pack(fill="x")
    sheet = ss.SetupSheet(host, services=_Cfg(), directory=tmp_path,
                          get=lambda *a, **k: "", post=lambda *a, **k: "",
                          run=lambda *a, **k: None, sleep=lambda s: None)
    sheet.show()
    root.update_idletasks()
    return sheet, host


@pytest.mark.parametrize("geometry", BOTH)
def test_what_a_new_sensor_needs_is_above_the_fold(root, geometry, tmp_path):
    """MEASURED at a4ecf8b, 920 (+ NEW form, 572-px viewport): address,
    gateway/mask, mac, wi-fi, password and ota code were ALL below the fold
    under a mount paragraph and the gates line; at 1040 the ota box was
    hidden. The rows a new sensor NEEDS come first."""
    sheet, host = _sheet(root, geometry, tmp_path)
    sheet.select("")
    _settle(root)
    view = sheet._canvas
    for key in ("room", "ip", "ssid"):
        assert _inside(sheet._field[key], view), (key, geometry,
                                                  view.winfo_height())
    for key in ("password", "ota_password"):
        assert _inside(sheet._secret[key], view), (key, geometry,
                                                   view.winfo_height())
    # and the rows are in NEED order: address before mount, wi-fi before
    # the gates line
    y = {k: sheet._field[k].winfo_rooty() for k in ("ip", "ssid",
                                                     "nearest_m")}
    assert y["ip"] < y["nearest_m"] and y["ssid"] < y["nearest_m"]
    assert sheet._gates.winfo_rooty() > sheet._secret["ota_password"].winfo_rooty()
    sheet.hide()


@pytest.mark.parametrize("geometry", BOTH)
def test_the_gates_line_wraps_and_no_sheet_label_is_cut(root, geometry,
                                                        tmp_path):
    """The gates line wraps to its slot (2 lines at 920, 1 at 1040); the
    wi-fi caption, which asked for 531 px of a 449-px slot at 920, is on a
    line of its own."""
    sheet, host = _sheet(root, geometry, tmp_path)
    for room in ("office", ""):
        sheet.select(room)
        if not room:
            # a blank form has no gates line until it validates
            for key, value in (("room", "den"), ("ip", "192.0.2.30"),
                               ("ssid", "PRETEND-NET-5G")):
                sheet._field[key].delete(0, "end")
                sheet._field[key].insert(0, value)
            sheet._recompute()
        _settle(root)
        gates = sheet._gates
        assert gates.winfo_ismapped()
        assert "move gate" in gates.cget("text"), (room,
                                                    sheet._addr.cget("text"))
        assert int(gates.cget("wraplength")) <= gates.winfo_width()
        assert gates.winfo_reqwidth() <= gates.winfo_width()
        bad = _overflowing(sheet)
        assert bad == [], (room, geometry, bad[:4])
    sheet.hide()


@pytest.mark.parametrize("geometry", BOTH)
def test_every_sheet_button_wears_a_border(root, geometry, tmp_path):
    """8 of 11 were bare at a4ecf8b: CLOSE, CHECK, TUNE, COPY FLASH COMMAND,
    BOARD & WIRING and every unselected room chip."""
    sheet, host = _sheet(root, geometry, tmp_path)
    sheet.select("office")
    _settle(root)
    assert _bare(sheet) == [], _bare(sheet)
    words = {b._text for b in _buttons(sheet)}
    assert {"+ NEW", "OFFICE", "CLOSE", "CHECK", "TUNE", "COPY FLASH COMMAND",
            "SAVE PROFILE", "BOARD & WIRING"} <= words, words
    # the chosen room chip is the accent one, the rest default
    chips = {b._text: b._kind for b in _buttons(sheet)
             if b._text in ("+ NEW", "OFFICE")}
    assert chips == {"OFFICE": "accent", "+ NEW": "default"}
    sheet.hide()


# ===================================================== the SETTINGS drawer
def _drawer(root, geometry, look="holo", on_open_tab=None):
    import tkinter as tk
    from jarvis.ui import views
    theme.select_look(look)
    w, h = geometry
    root.geometry("%dx%d+0+0" % (w, h))
    host = tk.Frame(root, width=w, height=h)
    host.pack_propagate(False)
    host.pack()
    drawer = views.SettingsDrawer(host, services=None,
                                  on_open_tab=on_open_tab)
    drawer.open()
    for _ in range(12):
        root.update()
    root.update_idletasks()
    return drawer, host


def _last_control(drawer):
    return [w for w in _walk(drawer._inner)
            if isinstance(w, (wg.RoundButton, wg.Toggle))][-1]


@pytest.mark.parametrize("geometry", BOTH)
def test_the_drawer_is_reachable_to_its_last_control_by_keyboard(root,
                                                                 geometry):
    """MEASURED at a4ecf8b: 3718 px of drawer in a 1329-px viewport at 920
    (36% visible), wheel-only, no mark. Now Home/End/Page/arrow keys scroll
    it, and a thumb says there is more."""
    drawer, host = _drawer(root, geometry)
    canvas = drawer._canvas
    assert drawer._inner.winfo_reqheight() > canvas.winfo_height()
    assert drawer._thumb.winfo_ismapped()
    assert canvas.winfo_children()[-1] is drawer._thumb
    last = _last_control(drawer)
    assert not _inside(last, canvas), "the drawer no longer overflows"
    drawer._key_scroll("end")
    _settle(root)
    assert _inside(last, canvas), (geometry, last.winfo_rooty(),
                                   canvas.winfo_rooty() + canvas.winfo_height())
    thumb_y = drawer._thumb.winfo_y()
    drawer._key_scroll("home")
    _settle(root)
    assert canvas.canvasy(0) == 0
    assert drawer._thumb.winfo_y() < thumb_y
    drawer._key_scroll("down")
    _settle(root)
    assert canvas.canvasy(0) > 0
    drawer._key_scroll("page_down")
    _settle(root)
    assert canvas.canvasy(0) > canvas.winfo_height() // 2
    # the keys are bound on the drawer, which takes focus when it opens
    for seq in ("<Down>", "<Up>", "<Next>", "<Prior>", "<Home>", "<End>"):
        assert drawer.bind(seq), seq


def test_the_drawer_points_at_the_users_and_sensors_tabs_with_buttons(root):
    """Where he looks first for "add a user" is Settings; the only pointer
    was a caption. In holo it is a button that closes the drawer and lights
    the tab, through a seam the console wires (on_open_tab)."""
    opened = []
    drawer, host = _drawer(root, OLD, on_open_tab=opened.append)
    users = [b for b in _buttons(drawer) if "Users tab" in b._text]
    sensors = [b for b in _buttons(drawer) if "Sensors tab" in b._text]
    assert len(users) == 1 and len(sensors) == 1
    assert users[0]._spec.get("outline") and sensors[0]._spec.get("outline")
    users[0].invoke()
    assert opened == ["users"]
    assert drawer._open is False
    drawer.open()
    sensors[0].invoke()
    assert opened == ["users", "sensors"]


def test_classic_keeps_its_caption_and_grows_no_thumb(root):
    """CLASSIC IS FROZEN (tests/test_ui_classic_frozen.py): the caption
    stays a caption and no 2-px strip is drawn in its drawer."""
    drawer, host = _drawer(root, OLD, look="classic")
    assert not any("Users tab" in b._text for b in _buttons(drawer))
    captions = [w for w in _walk(drawer._inner)
                if w.__class__.__name__ == "Label"
                and "Users tab" in str(w.cget("text"))]
    assert captions
    assert not drawer._thumb.winfo_ismapped()
    # ...but the keyboard works in both looks: behaviour, not pixels
    drawer._key_scroll("end")
    _settle(root)
    assert drawer._canvas.canvasy(0) > 0
