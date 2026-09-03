"""Measure the REAL tab row, in a process of its own. Not a test module.

WHY A SUBPROCESS. tests/test_tab_strip.py has to drive the shipping widget
-- selection, the keyboard and the show/hide lifecycle are behaviour, and
grepping this module's source for "<Return>" passes on code whose binding
has been deleted. But a Tk root may not be created inside the pytest
process: CLAUDE.md's own convention is "No Tk in unit tests", and building
one there ABORTED the suite outright (2026-09-03, SIGABRT at 87%, the
faulthandler dump 100 threads deep with no Python frame to blame). So the
Tk lives here, behind one ``subprocess.run``, and what crosses back is
JSON: measurements and a transcript of what the row did.

THE SPLIT IS DELIBERATE. This file MEASURES and records; it asserts
almost nothing. tests/test_tab_strip.py does the asserting, against the
pure functions and the theme tokens -- so a comparison can never be "the
driver said it was fine".

THE DISPLAY IS OURS. A private Xvfb is started here and torn down here;
the roots are created with ``tk.Tk(screenName=...)`` so DISPLAY is never
read or written, and his :1 is never touched. Nothing in this process has
a lens or a microphone: it is a Frame, two Canvases and a font.

    python -m tests.live_tab_strip          # JSON on stdout
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback

XVFB = os.path.expanduser("~/.local/xvfb/usr/bin/Xvfb")
SHIPPED = ("CHAT", "SENSORS")


# ------------------------------------------------------------- the display
def start_display():
    """(display name, Popen) or (None, None) with the reason on stderr."""
    import tkinter as tk
    if not os.path.exists(XVFB):
        return None, None
    for number in range(90, 100):
        if os.path.exists("/tmp/.X%d-lock" % number):
            continue
        name = ":%d" % number
        proc = subprocess.Popen(
            [XVFB, name, "-screen", "0", "1024x1600x24", "-nolisten", "tcp"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 5.0
        while time.time() < deadline and proc.poll() is None:
            try:
                probe = tk.Tk(screenName=name)
            except tk.TclError:
                time.sleep(0.1)
                continue
            probe.destroy()
            return name, proc
        proc.terminate()
    return None, None


# --------------------------------------------------------------- the rig
class Rig:
    """One root, one strip, and a recorder on every callback."""

    def __init__(self, display, scale=2.0, look="holo", words=SHIPPED,
                 on_change_raises=False, select_raises=()):
        import tkinter as tk
        from jarvis.ui import tab_strip as ts
        from jarvis.ui import theme
        from jarvis.ui.widgets import set_scale
        self.ts, self.theme = ts, theme
        self.root = tk.Tk(screenName=display)
        self.root.geometry("920x1440")
        # What main_window.create() does once a root exists: with no root
        # the display face is unknown and every width falls back to the
        # body font.
        theme.resolve_fonts(self.root)
        theme.select_look(look)
        theme.apply_scale(scale)
        set_scale(scale)
        self.calls: list = []
        self.changes: list = []

        def on_change(key):
            self.changes.append(key)
            if on_change_raises:
                raise RuntimeError("a broken listener")

        self.strip = ts.TabStrip(self.root, bg=theme.BG, on_change=on_change)
        self.strip.pack(fill="x", side="top")
        for word in words:
            key = word.lower()
            self.strip.add(key, word,
                           select=self._cb(key, "select", key in select_raises),
                           leave=self._cb(key, "leave", False))
        self.pump()

    def _cb(self, key, which, raises):
        def run():
            self.calls.append("%s:%s" % (key, which))
            if raises:
                raise RuntimeError("no page")
        return run

    def pump(self):
        self.root.update_idletasks()
        self.root.update()

    def tab(self, key):
        return self.strip.widget(key)

    def focus_key(self):
        got = self.root.focus_get()
        for key in self.strip.keys:
            if self.tab(key) is got:
                return key
        return None

    def state(self, name):
        """One entry in the transcript: everything a reader would want to
        know after a press, and the calls SINCE the last entry."""
        calls, self.calls = list(self.calls), []
        changes, self.changes = list(self.changes), []
        return {"step": name, "selected": self.strip.selected,
                "flags": {k: bool(self.tab(k).selected)
                          for k in self.strip.keys},
                "calls": calls, "changes": changes, "focus": self.focus_key()}

    def send(self, key, event):
        self.tab(key).event_generate(event)
        self.pump()

    def close(self):
        try:
            self.root.destroy()
        except Exception:                     # noqa: BLE001 - already gone
            pass


def items(canvas):
    """[(type, fill)] for every canvas item, in draw order."""
    return [[canvas.type(i), canvas.itemcget(i, "fill")]
            for i in canvas.find_all()]


# ------------------------------------------------------------ the readings
def geometry(display) -> dict:
    """Widths and heights off real widgets at both scales, both looks."""
    import tkinter.font as tkfont
    out = {}
    for scale in (1.0, 2.0):
        for look in ("holo", "classic"):
            rig = Rig(display, scale, look)
            text_w, req_w, req_h, line_h = {}, {}, {}, None
            for word in SHIPPED:
                tab = rig.tab(word.lower())
                font = tkfont.Font(root=rig.root, font=tab._font)
                text_w[word] = font.measure(word)
                line_h = font.metrics("linespace")
                req_w[word.lower()] = tab.winfo_reqwidth()
                req_h[word.lower()] = tab.winfo_reqheight()
            out["%s-%s" % (scale, look)] = {
                "scale": scale, "look": look, "text_w": text_w,
                "linespace": line_h, "req_w": req_w, "req_h": req_h,
                "strip_req_h": rig.strip.winfo_reqheight(),
                "measured_widths": [list(p) for p in rig.strip.measured_widths()],
                "clipped_918": [list(c) for c in rig.strip.clipped(918)],
                "display_face": rig.theme._DISPLAY,
            }
            rig.close()
    return out


def four_tabs(display) -> dict:
    rig = Rig(display, 2.0, "holo",
              words=("CHAT", "SENSORS", "SETTINGS", "CAMERA"))
    out = {"keys": list(rig.strip.keys),
           "measured_widths": [list(p) for p in rig.strip.measured_widths()],
           "clipped_918": [list(c) for c in rig.strip.clipped(918)],
           "clipped_300": [list(c) for c in rig.strip.clipped(300)]}
    rig.close()
    return out


def transcript(display) -> list:
    """What the row DOES: every press, every key, in order."""
    steps = []
    rig = Rig(display)
    steps.append(rig.state("built"))
    rig.send("sensors", "<ButtonRelease-1>")
    steps.append(rig.state("click-sensors"))
    rig.send("sensors", "<ButtonRelease-1>")
    steps.append(rig.state("click-sensors-again"))
    rig.send("chat", "<ButtonRelease-1>")
    steps.append(rig.state("click-chat"))
    steps.append(dict(rig.state("select-same-returns"),
                      returned=[rig.strip.select("chat"),
                                rig.strip.select("nosuchtab")]))
    rig.close()

    for key in ("<Return>", "<space>"):
        rig = Rig(display)
        rig.tab("sensors").focus_set()
        rig.pump()
        steps.append(dict(rig.state("focus-sensors" + key),
                          takefocus=str(rig.tab("sensors").cget("takefocus"))))
        rig.send("sensors", key)
        steps.append(rig.state("press" + key))
        rig.close()

    rig = Rig(display)
    rig.tab("chat").focus_set()
    rig.pump()
    steps.append(rig.state("focus-chat"))
    rig.send("chat", "<Right>")
    steps.append(rig.state("right-from-chat"))
    rig.send("sensors", "<Left>")
    steps.append(rig.state("left-from-sensors"))
    rig.send("chat", "<Left>")
    steps.append(rig.state("left-at-the-start"))
    rig.tab("sensors").focus_set()
    rig.pump()
    rig.state("park-on-sensors")
    rig.send("sensors", "<Right>")
    steps.append(rig.state("right-at-the-end"))
    rig.strip.select("sensors")
    rig.pump()
    rig.strip.focus_selected()
    rig.pump()
    steps.append(rig.state("focus-selected"))
    rig.close()
    return steps


def paint(display) -> dict:
    """The selection channels, read off the canvas with the keyboard focus
    parked on the OTHER tab -- the state his console sits in all day."""
    rig = Rig(display)
    rig.tab("sensors").focus_set()          # focus on the UNSELECTED tab
    rig.pump()
    chat, sensors = rig.tab("chat"), rig.tab("sensors")
    out = {"selected_key": rig.strip.selected,
           "selected_items": items(chat), "unselected_items": items(sensors),
           "ring": {"chat": str(chat.cget("highlightcolor")),
                    "sensors": str(sensors.cget("highlightcolor"))},
           "bg": {"chat": str(chat.cget("bg")),
                  "sensors": str(sensors.cget("bg"))},
           # ONE tab, both states: the underline lives inside the box, so
           # a click must not resize anything under the row.
           "chat_size_selected": [chat.winfo_reqwidth(),
                                  chat.winfo_reqheight()],
           "tokens": {name: getattr(rig.theme, name)
                      for name in ("FOCAL", "MUTED", "CYAN", "RAISED", "INK")}}
    rig.strip.select("sensors")
    rig.pump()
    out["chat_size_unselected"] = [chat.winfo_reqwidth(),
                                   chat.winfo_reqheight()]
    out["chat_items_unselected"] = items(chat)
    rig.close()
    return out


def a_surface_that_raises(display) -> dict:
    """A page that cannot open, and a listener that throws: the row still
    lights the tab the press asked for."""
    rig = Rig(display, on_change_raises=True, select_raises=("sensors",))
    changed = rig.strip.select("sensors")
    rig.pump()
    out = dict(rig.state("select-a-broken-surface"), returned=bool(changed))
    rig.close()
    return out


def observe(display) -> dict:
    return {"display": display,
            "geometry": geometry(display),
            "four_tabs": four_tabs(display),
            "transcript": transcript(display),
            "paint": paint(display),
            "raises": a_surface_that_raises(display)}


def main() -> int:
    display, proc = start_display()
    if display is None:
        print(json.dumps({"skip": "no usable Xvfb at %s" % XVFB}))
        return 0
    try:
        out = observe(display)
    except Exception:                         # noqa: BLE001 - report it back
        out = {"error": traceback.format_exc()}
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:                 # noqa: BLE001 - already gone
                pass
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
