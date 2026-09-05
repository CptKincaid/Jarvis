"""Knightfall in the settings drawer (Privacy section): a masked entry, an
Open button, and "Email me a new Knightfall code".

Tk-free the way tests/test_restart_button.py is: the logic lives in
``KnightfallControl`` (read, clear, call off the Tk thread, toast the
line), and the drawer is built with ``__new__`` over fakes for the entry,
the toast and after(). The load-bearing assertion is that the ENTRY IS
CLEARED BEFORE THE SERVICE RETURNS -- a code must not sit on screen while
the mail goes out -- and that the code itself never reaches a toast, a
log or a thread name.
"""
import inspect
import threading
from dataclasses import fields
from types import SimpleNamespace

from jarvis.ui import views as views_mod
from jarvis.ui.main_window import Services
from jarvis.ui.views import KnightfallControl, SettingsDrawer

TYPED = "xx7typed"


# ------------------------------------------------------------ Services
def test_services_declare_both_knightfall_hooks_unwired_by_default():
    names = {f.name for f in fields(Services)}
    assert {"knightfall_code", "knightfall_new_code"} <= names
    svc = Services()
    assert svc.knightfall_code is None and svc.knightfall_new_code is None


def test_the_app_supplies_both_hooks():
    import jarvis.app as app_mod
    src = inspect.getsource(app_mod.JarvisApp.ui_service_kwargs)
    assert "knightfall_code=self.knightfall_code" in src
    assert "knightfall_new_code=self.knightfall_new_code" in src


# ------------------------------------------------- the Tk-free control
class _Box:
    """A fake entry: what was read, and when it was cleared."""

    def __init__(self, text):
        self.text = text
        self.cleared = 0

    def read(self):
        return self.text

    def clear(self):
        self.text = ""
        self.cleared += 1


def _control(services, box, *, spawn=None):
    toasts, later = [], []
    ctl = KnightfallControl(services, read=box.read, clear=box.clear,
                            toast=lambda text, kind="info": toasts.append((text, kind)),
                            later=later.append, spawn=spawn)
    return ctl, toasts, later


def _drain(later):
    while later:
        later.pop(0)()


def test_the_entry_is_cleared_before_the_service_returns():
    release, inside = threading.Event(), threading.Event()
    seen = []

    def service(code):
        seen.append(code)
        inside.set()
        assert release.wait(2.0)
        return "Knightfall accepted, sir; a new code is in your inbox."
    box = _Box(TYPED)
    ctl, toasts, later = _control(SimpleNamespace(knightfall_code=service), box)
    assert ctl.open_pressed() == "started"
    assert inside.wait(2.0), "the service never ran"
    assert box.text == "" and box.cleared == 1, "cleared while the call is in flight"
    assert toasts == [] and later == []
    release.set()
    for _ in range(50):
        if later:
            break
        threading.Event().wait(0.02)
    _drain(later)
    assert seen == [TYPED]
    assert toasts == [("Knightfall accepted, sir; a new code is in your inbox.", "ok")]


def test_a_refusal_line_is_a_warn_toast_and_the_code_is_nowhere_in_it():
    box = _Box(TYPED)
    ctl, toasts, later = _control(
        SimpleNamespace(knightfall_code=lambda code: "Knightfall: that is not a code I know"),
        box, spawn=lambda fn, *a: fn(*a))
    ctl.open_pressed()
    _drain(later)
    assert toasts == [("Knightfall: that is not a code I know", "warn")]
    assert TYPED not in repr(toasts)


def test_the_service_runs_off_the_calling_thread_by_default():
    names, done = [], threading.Event()

    def service(code):
        names.append(threading.current_thread().name)
        done.set()
        return "Knightfall: no code was given"
    ctl, toasts, later = _control(SimpleNamespace(knightfall_code=service),
                                  _Box(""))
    ctl.open_pressed()
    assert done.wait(2.0)
    assert names and names[0] != threading.main_thread().name
    assert TYPED not in names[0]


def test_not_wired_is_a_toast_and_the_entry_is_still_cleared():
    box = _Box(TYPED)
    ctl, toasts, later = _control(SimpleNamespace(), box)
    assert ctl.open_pressed() == "not wired"
    assert toasts == [("Knightfall not wired", "warn")] and box.text == ""
    ctl, toasts, later = _control(None, _Box(TYPED))
    assert ctl.new_pressed() == "not wired"
    assert toasts == [("Knightfall not wired", "warn")]


def test_a_failing_service_is_logged_not_raised_and_still_toasts():
    def boom(code):
        raise RuntimeError("smtp exploded")
    ctl, toasts, later = _control(SimpleNamespace(knightfall_code=boom),
                                  _Box(TYPED), spawn=lambda fn, *a: fn(*a))
    ctl.open_pressed()
    _drain(later)
    assert len(toasts) == 1 and toasts[0][1] == "warn"
    assert "Knightfall" in toasts[0][0] and TYPED not in toasts[0][0]


def test_the_new_code_button_calls_its_own_service_and_toasts_the_line():
    calls = []

    def new_code():
        calls.append(1)
        return "Knightfall: a new code is in your inbox."
    ctl, toasts, later = _control(SimpleNamespace(knightfall_new_code=new_code),
                                  _Box(""), spawn=lambda fn, *a: fn(*a))
    assert ctl.new_pressed() == "started"
    _drain(later)
    assert calls == [1]
    assert toasts == [("Knightfall: a new code is in your inbox.", "ok")]
    ctl, toasts, later = _control(
        SimpleNamespace(knightfall_new_code=lambda: "Knightfall: one code a minute, sir."),
        _Box(""), spawn=lambda fn, *a: fn(*a))
    ctl.new_pressed()
    _drain(later)
    assert toasts[0][1] == "warn"


# --------------------------------------------------------- drawer wiring
class _FakeEntry:
    def __init__(self, text):
        self.text = text
        self.deleted = []

    def get(self):
        return self.text

    def delete(self, first, last=None):
        self.deleted.append((first, last))
        self.text = ""


class _FakeToast:
    def __init__(self):
        self.shown = []

    def show(self, text, kind="info", ms=1800):
        self.shown.append((text, kind))


def _drawer(services, typed=TYPED):
    d = SettingsDrawer.__new__(SettingsDrawer)       # no tk.Frame.__init__
    d.services = services
    d.toast = _FakeToast()
    d.after_calls = []
    d.after = lambda ms, fn: d.after_calls.append((ms, fn)) or f"job{len(d.after_calls)}"
    d._knightfall_entry = _FakeEntry(typed)
    return d


def test_open_reads_the_entry_clears_it_at_once_and_toasts_off_the_tk_thread():
    release, inside = threading.Event(), threading.Event()

    def service(code):
        inside.set()
        assert release.wait(2.0)
        return "Knightfall accepted, sir; the code stays as it is (mail: nope)."
    d = _drawer(SimpleNamespace(knightfall_code=service))
    d._knightfall_open_pressed()
    assert inside.wait(2.0)
    assert d._knightfall_entry.text == "" and d._knightfall_entry.deleted == [(0, "end")]
    assert d.toast.shown == []
    release.set()
    for _ in range(50):
        if d.after_calls:
            break
        threading.Event().wait(0.02)
    assert d.after_calls and d.after_calls[0][0] == 0
    d.after_calls[0][1]()
    assert d.toast.shown == [
        ("Knightfall accepted, sir; the code stays as it is (mail: nope).", "ok")]


def test_without_a_service_the_drawer_toasts_not_wired():
    d = _drawer(SimpleNamespace())
    d._knightfall_open_pressed()
    assert d.toast.shown == [("Knightfall not wired", "warn")]
    assert d._knightfall_entry.text == ""
    d = _drawer(None)
    d._knightfall_new_pressed()
    assert d.toast.shown == [("Knightfall not wired", "warn")]


def test_the_new_code_button_is_wired_through_the_same_control():
    d = _drawer(SimpleNamespace(knightfall_new_code=lambda: "Knightfall: a new code is in your inbox."))
    d._knightfall_new_pressed()
    for _ in range(50):
        if d.after_calls:
            break
        threading.Event().wait(0.02)
    d.after_calls[0][1]()
    assert d.toast.shown == [("Knightfall: a new code is in your inbox.", "ok")]


def test_the_privacy_section_builds_the_row_with_a_masked_entry():
    """The wiring, not the widgets: Privacy must get the row and the
    button through the helpers, and the entry must be masked."""
    src = inspect.getsource(SettingsDrawer._build_sections)
    priv = src.index('self._section("Privacy")')
    system = src.index('self._section("System")')
    assert "self._knightfall_row(box)" in src[priv:system]
    row = inspect.getsource(SettingsDrawer._knightfall_row)
    assert 'show="•"' in row
    assert views_mod.KNIGHTFALL_NOT_WIRED == "Knightfall not wired"
    # CLASSIC keeps the 09-04 words exactly; it is frozen
    # (tests/test_ui_classic_frozen.py) and its row still overflows.
    assert '"Knightfall code"' in row
    assert '"Email me a new Knightfall code"' in row
    # HOLO -- the look he runs -- has the words that FIT the 576 px slot:
    # the full label left no arrangement that did (2026-09-05).
    assert SettingsDrawer.KNIGHTFALL_LABEL_HOLO == "Knightfall"
    assert SettingsDrawer.KNIGHTFALL_NEW_HOLO == "Email me a new code"
    assert "KNIGHTFALL_LABEL_HOLO" in row and "KNIGHTFALL_NEW_HOLO" in row
    # the look is read at CALL time, never captured beside the def
    assert 'theme.LOOK == "holo"' in row


def test_the_masked_box_is_sized_from_the_code_the_generator_makes():
    """A box narrower than a code shows him a code that scrolls. The
    number is the drawer's, the length is passphrase's, and they are
    pinned to each other rather than both being hand-picked."""
    from jarvis import passphrase as pp
    assert SettingsDrawer.KNIGHTFALL_CHARS >= pp.NEW_CODE_LEN == 8


# ------------------------------------------- what the row says, and when
# Before this, the caption was the flat sentence "Typed only, never spoken.
# Using it emails you the next one." -- which is FALSE with no mail account
# configured, the state he was actually in: the button mails nothing and
# the promise is made before he presses. The caption is state-dependent
# now, and the destination is MASKED because the drawer is on screen.
def _status(**kw):
    base = {"to": "", "problem": "", "setup": "setup line"}
    base.update(kw)
    return base


def test_with_no_mail_account_the_row_says_so_and_the_button_is_dead():
    text, can_press = views_mod.format_knightfall_status(_status())
    assert can_press is False
    assert views_mod.KNIGHTFALL_NO_ACCOUNT in text
    assert "setup line" in text, "he is told where to fix it"
    assert "emails you the next one" not in text


def test_with_an_account_the_row_names_the_masked_destination():
    text, can_press = views_mod.format_knightfall_status(
        _status(to="h…@example.com"))
    assert can_press is True
    assert "The next code goes to h…@example.com." in text
    assert "Typed only, never spoken." in text


def test_a_bad_configured_destination_is_said_and_the_button_is_dead():
    text, can_press = views_mod.format_knightfall_status(
        _status(to="h…@example.com", problem="that address is not an address"))
    assert can_press is False
    assert "that address is not an address" in text


def test_an_unwired_or_broken_status_reads_as_no_account():
    for bad in (None, {}, "nonsense"):
        text, can_press = views_mod.format_knightfall_status(bad)
        assert can_press is False and views_mod.KNIGHTFALL_NO_ACCOUNT in text


def test_the_row_is_built_from_the_status_and_refreshed_on_open():
    """Wiring, not widgets: the caption must not be a literal any more, and
    it must be re-read when the drawer opens (his config can change while
    it is shut)."""
    row = inspect.getsource(SettingsDrawer._knightfall_row)
    assert "_refresh_knightfall" in row
    assert "Using it emails you" not in row
    assert "_refresh_knightfall" in inspect.getsource(SettingsDrawer.open)


def test_refresh_sets_the_caption_and_disables_the_button():
    d = SettingsDrawer.__new__(SettingsDrawer)
    d.services = SimpleNamespace(knightfall_status=lambda: _status())
    lbl, btn = _FakeLabel(), _FakeButton()
    d._knightfall_info = lbl
    d._knightfall_new = btn
    d._refresh_knightfall()
    assert views_mod.KNIGHTFALL_NO_ACCOUNT in lbl.text
    assert btn.enabled is False
    d.services = SimpleNamespace(
        knightfall_status=lambda: _status(to="h…@example.com"))
    d._refresh_knightfall()
    assert "h…@example.com" in lbl.text and btn.enabled is True


def test_a_status_service_that_raises_does_not_take_the_drawer_down():
    def boom():
        raise RuntimeError("nope")
    d = SettingsDrawer.__new__(SettingsDrawer)
    d.services = SimpleNamespace(knightfall_status=boom)
    lbl, btn = _FakeLabel(), _FakeButton()
    d._knightfall_info, d._knightfall_new = lbl, btn
    d._refresh_knightfall()
    assert views_mod.KNIGHTFALL_NO_ACCOUNT in lbl.text and btn.enabled is False


def test_services_declare_the_status_hook_and_the_app_supplies_it():
    import jarvis.app as app_mod
    assert "knightfall_status" in {f.name for f in fields(Services)}
    assert Services().knightfall_status is None
    assert "knightfall_status=self.knightfall_status" in \
        inspect.getsource(app_mod.JarvisApp.ui_service_kwargs)


class _FakeLabel:
    def __init__(self):
        self.text = ""

    def configure(self, **kw):
        self.text = kw.get("text", self.text)


class _FakeButton:
    def __init__(self):
        self.enabled = True

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)
