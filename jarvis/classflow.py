"""Class-start staging: the material is already open when it begins.

09:10 arrives, and instead of a line he has already heard ten minutes ago,
the desk is set: today's notes file for BIOSENSORS exists with its dated
header, the music is down, and a card says so. He walks into a room that is
ready.

What it deliberately does NOT do, and why:

* **No new windows.** The 2026-08-26 desktop freeze was window churn on :1.
  Everything of value here -- a primed, dated notes file and quiet music --
  costs zero windows and zero focus changes. Opening the file in an editor
  is behind ``class_flow.open_notes`` (default off); it shells out to
  ``xdg-open`` and nothing else.
* **No silent recording.** ``LectureNotes.add()`` appends every accepted
  utterance to disk. Arming that from a calendar tick would record a room
  he never agreed to record, so ``class_flow.auto_notes`` is off by
  default, and when he does turn it on Jarvis SAYS that notes are open --
  consent in the config, notice in the room.
* **Nothing while he is out.** ``presence.phone_ip`` is empty on this box,
  so the phone gate is inert; the real signal is X idle time
  (jarvis/desk.py). Quiet hours and DND stop it too -- but the calendar
  leg of quiet is deliberately ignored, because with jarvis/courses.py
  feeding quiet.py the class itself is now a quiet window, and a gate that
  the event trips the instant it starts would never let the stager run.

Every action is logged and recorded in a state file, and every one of them
is undone: the music comes back and auto-notes close at the end of the
event, on stop(), and on the first wake word after the class is over (the
safety net for a stage whose end tick never ran because the app was down).
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from jarvis import courses as courses_mod
from jarvis import desk as desk_mod
from jarvis import lecture as lecture_mod
from jarvis import music as music_mod
from jarvis.events import BriefingReady, HotwordDetected, bus
from jarvis.logs import get_logger

log = get_logger("classflow")

TICK_S = 45.0                  # "09:10 arrives" needs better than headsup's 300 s
GRACE_MIN = 5                  # a class already 6 minutes old is not "starting"
DEFAULT_IDLE_MIN = 15

STAGED_LINE = "{course} staged — notes ready"
NOTES_ROW = "notes: {path}"
MUSIC_ROW = "music paused for the hour"


class ClassStager:
    """One tick, one class: stage at the start, restore at the end."""

    def __init__(self, cfg, get_calendar: Callable = None, services=None,
                 state_path: Optional[Path] = None, now: Callable = None,
                 get_commander: Callable = None, bg: Optional[Callable] = None,
                 publish: Optional[Callable] = None, tick_s: float = TICK_S,
                 turns_path=None, opener: Optional[Callable] = None):
        self._cfg = cfg
        self._get_calendar = get_calendar
        self._services = services
        self._get_commander = get_commander
        self._state_path = Path(state_path) if state_path else None
        self._now = now or (lambda tz=None: datetime.now(tz))
        self._bg = bg or self._thread_bg
        self._publish = publish or bus.publish
        self.tick_s = float(tick_s)
        self._turns_path = turns_path
        self._opener = opener or self._xdg_open
        self._lock = threading.RLock()
        self._workers: list = []
        state = self._load()
        self._staged: dict = state.get("staged", {})     # key -> what was done
        self._done: dict = state.get("done", {})         # key -> event start ISO
        self._skips: set = set()                         # log-once, not state
        self._stop = threading.Event()
        self._thread = None
        bus.subscribe(HotwordDetected, self._on_wake)

    # ------------------------------------------------------------ config
    def _get(self, key: str, default=None):
        get = getattr(self._cfg, "get", None)
        if not callable(get):
            return default
        try:
            value = get(key, default)
        except Exception:                   # noqa: BLE001 - config boundary
            log.debug("classflow: config read failed for %s", key, exc_info=True)
            return default
        return default if value is None else value

    @property
    def enabled(self) -> bool:
        return bool(self._get("class_flow.enabled", True))

    # ------------------------------------------------------------- state
    def _load(self) -> dict:
        try:
            if self._state_path and self._state_path.exists():
                data = json.loads(self._state_path.read_text())
                if isinstance(data, dict):
                    staged = data.get("staged")
                    done = data.get("done")
                    return {"staged": staged if isinstance(staged, dict) else {},
                            "done": done if isinstance(done, dict) else {}}
        except (OSError, ValueError):
            log.debug("classflow state unreadable", exc_info=True)
        return {"staged": {}, "done": {}}

    def _save(self) -> None:
        if not self._state_path:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_name(self._state_path.name + ".tmp")
            tmp.write_text(json.dumps({"staged": self._staged, "done": self._done}))
            os.replace(tmp, self._state_path)     # atomic: never half a file
        except OSError:
            log.debug("classflow state save failed", exc_info=True)

    # ---------------------------------------------------------- calendar
    def _events(self) -> list:
        cal = self._get_calendar() if callable(self._get_calendar) else self._get_calendar
        if cal is None:
            return []
        try:
            conf = getattr(cal, "configured", True)
            if callable(conf):
                conf = conf()
            if not conf:
                return []
            return list(cal.events())
        except Exception:                   # noqa: BLE001 - the cache is best-effort
            log.debug("classflow: calendar unavailable", exc_info=True)
            return []

    # -------------------------------------------------------------- gates
    def _quiet_reason(self) -> str:
        """DND / quiet hours / away -- but NOT the calendar leg. The class
        being staged is itself the running calendar event, so asking the
        full reason() would refuse to stage every class there is."""
        quiet = getattr(self._services, "quiet", None)
        reason = getattr(quiet, "reason", None)
        if not callable(reason):
            return ""
        try:
            return str(reason(calendar=False) or "")
        except TypeError:                   # a policy without the keyword
            return ""
        except Exception:                   # noqa: BLE001 - policy boundary
            log.debug("classflow: quiet reason failed", exc_info=True)
            return ""

    def at_desk(self) -> bool:
        try:
            idle_min = float(self._get("class_flow.idle_min", DEFAULT_IDLE_MIN))
        except (TypeError, ValueError):
            idle_min = DEFAULT_IDLE_MIN
        return desk_mod.at_desk(idle_min, turns_path=self._turns_path)

    # --------------------------------------------------------------- tick
    def tick(self) -> int:
        """Stage classes that have just started, restore ones that ended.
        Returns how many were staged this pass."""
        restored = self._restore_finished()
        if not self.enabled:
            return 0
        events = self._events()
        if not events:
            return 0
        names = courses_mod.recurring_courses(events)
        if not names:
            return 0
        staged = 0
        grace = timedelta(minutes=GRACE_MIN)
        for ev in events:
            start = getattr(ev, "start", None)
            if start is None or getattr(ev, "all_day", False):
                continue
            course = courses_mod.course_for(getattr(ev, "title", ""), names)
            if not course:
                continue
            now = self._now(start.tzinfo)
            if not (start <= now < start + grace):
                continue
            key = self._key(ev)
            with self._lock:
                if key in self._staged or key in self._done:
                    continue
            # A refused gate is NOT remembered: he may sit down, or lift the
            # do-not-disturb, a minute into the hour, and GRACE_MIN already
            # bounds how long the offer stands. Only the log is deduplicated.
            reason = self._quiet_reason() or \
                ("" if self.at_desk() else "not at the desk")
            if reason:
                self._log_skip(key, course, reason)
                continue
            self._stage(key, ev, course)
            staged += 1
        self._prune()
        if staged or restored:
            self._save()
        return staged

    def _key(self, ev) -> str:
        title = " ".join(str(getattr(ev, "title", "") or "").split())
        return f"{title}|{ev.start.isoformat()}"

    def _log_skip(self, key: str, course: str, why: str) -> None:
        """Say why once per class, not once every 45 s for five minutes."""
        if key in self._skips:
            return
        self._skips.add(key)
        log.info("classflow: %r not staged (%s)", course, why)

    # -------------------------------------------------------------- stage
    def _stage(self, key: str, ev, course: str) -> dict:
        """Do the staging; record exactly what was done so it can be undone."""
        did: dict = {"course": course, "start": ev.start.isoformat(),
                     "end": getattr(ev, "end", ev.start).isoformat(),
                     "notes_path": "", "armed": False, "paused": False}
        rows = []
        try:
            path = self._prime_notes(ev, course)
            if path:
                did["notes_path"] = str(path)
                # the card gets the file name; the full path is in the log,
                # where a line can be as long as it likes
                rows.append(NOTES_ROW.format(path=path.name))
            if path and self._get("class_flow.open_notes", False):
                self._opener(path)
            if path and self._get("class_flow.auto_notes", False):
                did["armed"] = bool(self._arm_notes(ev, course))
            if self._get("class_flow.duck_music", True):
                did["paused"] = self._duck()
                if did["paused"]:
                    rows.append(MUSIC_ROW)
        except Exception:                   # noqa: BLE001 - staging boundary
            # Half a staged desk is worse than none: put back whatever
            # landed before the failure and leave the class unstaged.
            log.exception("classflow: staging %r failed; restoring", course)
            self._undo(did)
            with self._lock:
                self._done[key] = did["start"]
            self._save()
            return did
        with self._lock:
            self._staged[key] = did
        self._save()
        log.info("classflow: staged %r (notes=%s armed=%s music=%s)", course,
                 did["notes_path"] or "-", did["armed"], did["paused"])
        self._card(course, rows)
        if did["armed"]:
            # He configured auto-notes, but he is still told the room is
            # being written down: consent in the config, notice out loud.
            self._speak(lecture_mod.START_LINE.format(course=course))
        return did

    def _prime_notes(self, ev, course: str) -> Optional[Path]:
        """Today's dated notes file, created with its header if absent.
        Capture stays OFF: this only means the page is open."""
        try:
            path = lecture_mod.note_path(self._cfg, course, ev.start)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text(lecture_mod.HEADER.format(
                    course=course, date=ev.start.date().isoformat()))
            return path
        except OSError:
            log.info("classflow: notes folder unwritable for %r", course, exc_info=True)
            return None

    def _arm_notes(self, ev, course: str) -> bool:
        commander = self._get_commander() if callable(self._get_commander) \
            else self._get_commander
        if commander is None or getattr(commander, "lecture_course", None):
            return False                    # already noting something
        try:
            capture = lecture_mod.LectureNotes(self._cfg, course,
                                               notes=getattr(self._services, "notes", None))
        except Exception:                   # noqa: BLE001 - file boundary
            log.exception("classflow: cannot open notes for %r", course)
            return False
        # The same two attributes commander._h_lecture_start sets, in the
        # same order: the commander owns the mode flag, this owns the file.
        commander._lecture = capture
        commander.lecture_course = capture.course
        log.info("classflow: lecture notes armed for %r -> %s", course, capture.path)
        return True

    def _duck(self) -> bool:
        """Pause Spotify if something is actually playing. Returns whether
        there is anything to put back."""
        spotify = getattr(self._services, "spotify", None)
        if spotify is None or music_mod.playing(spotify) is not True:
            return False
        ok, _kind = music_mod.pause(spotify)
        return bool(ok)

    @staticmethod
    def _xdg_open(path) -> None:
        subprocess.Popen(["xdg-open", str(path)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)

    # ------------------------------------------------------------ restore
    def _restore_finished(self) -> int:
        """Undo every stage whose event has ended."""
        now = self._now()
        ready = []
        with self._lock:
            for key, did in list(self._staged.items()):
                end = self._parse(did.get("end"))
                if end is None or self._passed(end, now):
                    ready.append((key, did))
        for key, did in ready:
            self._undo(did)
            with self._lock:
                self._staged.pop(key, None)
                self._done[key] = str(did.get("start") or "")
        if ready:
            self._save()
        return len(ready)

    def _undo(self, did: dict) -> None:
        """Put back everything ``did`` changed. Safe to call twice."""
        if did.get("paused"):
            spotify = getattr(self._services, "spotify", None)
            ok, kind = music_mod.resume(spotify)
            did["paused"] = not ok
            if ok:
                log.info("classflow: music resumed")
            else:
                # Not retried: the stage is being closed either way, and a
                # thread that keeps poking a dead Connect device forever is
                # worse than a paused speaker he can start himself.
                log.warning("classflow: music left paused (%s)", kind or "no spotify")
        if did.get("armed"):
            commander = self._get_commander() if callable(self._get_commander) \
                else self._get_commander
            capture = getattr(commander, "_lecture", None) if commander else None
            course = str(did.get("course") or "")
            # Only ours: he may have opened notes for something else since.
            if capture is not None and getattr(capture, "course", "") == course:
                try:
                    log.info("classflow: %s", capture.close())
                except Exception:           # noqa: BLE001 - file boundary
                    log.exception("classflow: closing notes failed")
                commander._lecture = None
                commander.lecture_course = None
            did["armed"] = False

    def _on_wake(self, _ev=None) -> None:
        """The safety net: a stage whose end tick never ran (the app was
        down, the thread was stopped) is undone the next time he speaks --
        never one that is still running, or the music would come back on in
        the middle of the lecture.

        The bus delivers on the Tk thread, and putting the music back is a
        network call, so the work goes to a worker; the cheap check that
        there is any work at all stays here so a wake word does not spawn a
        thread for nothing."""
        with self._lock:
            if not self._staged:
                return
        def run():
            try:
                if self._restore_finished():
                    log.info("classflow: restored a finished stage at the wake word")
            except Exception:               # noqa: BLE001 - worker boundary
                log.debug("classflow: wake restore failed", exc_info=True)
        self._bg(run)

    def restore_all(self) -> None:
        """Undo everything still staged, whatever the clock says (quit)."""
        with self._lock:
            items = list(self._staged.items())
        for key, did in items:
            self._undo(did)
            with self._lock:
                self._staged.pop(key, None)
                self._done[key] = str(did.get("start") or "")
        if items:
            self._save()

    # -------------------------------------------------------------- misc
    @staticmethod
    def _parse(value) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None

    def _passed(self, when: datetime, now: datetime) -> bool:
        ref = self._now(when.tzinfo) if when.tzinfo else now.replace(tzinfo=None)
        try:
            return when <= ref
        except TypeError:
            return True

    def _prune(self) -> None:
        cutoff = (self._now() - timedelta(days=2)).isoformat()
        with self._lock:
            stale = [k for k, v in self._done.items()
                     if not isinstance(v, str) or v < cutoff]
            for k in stale:
                self._done.pop(k, None)
        if stale:
            self._save()

    def _card(self, course: str, rows: list) -> None:
        lines = [STAGED_LINE.format(course=course)] + [r for r in rows if r]
        try:
            self._publish(BriefingReady(sections={"class": lines}, spoken=""))
        except Exception:                   # noqa: BLE001 - bus boundary
            log.exception("classflow: card publish failed")

    def _speak(self, text: str) -> None:
        say = getattr(self._services, "speak", None)
        if not text or not callable(say):
            return
        try:
            try:
                say(text, proactive=True, kind="message")
            except TypeError:
                say(text)                   # a bare test seam takes text only
        except Exception:                   # noqa: BLE001 - speech boundary
            log.exception("classflow: speak failed")

    def _thread_bg(self, fn) -> None:
        t = threading.Thread(target=fn, daemon=True, name="classflow-work")
        with self._lock:
            self._workers = [w for w in self._workers if w.is_alive()]
            self._workers.append(t)
        t.start()

    # ------------------------------------------------------------ thread
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="classflow")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            bus.unsubscribe(HotwordDetected, self._on_wake)
        except Exception:                   # noqa: BLE001 - bus boundary
            log.debug("classflow: unsubscribe failed", exc_info=True)
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)
        # Quit must not leave the music paused and a lecture recording.
        try:
            self.restore_all()
        except Exception:                   # noqa: BLE001 - teardown boundary
            log.exception("classflow: restore on stop failed")
        with self._lock:
            workers = list(self._workers)
        for w in workers:
            if w is not threading.current_thread():
                w.join(timeout=2.0)

    def _run(self) -> None:
        if self._stop.wait(25.0):           # let the calendar refresh first
            return
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                log.exception("classflow tick failed")
            if self._stop.wait(self.tick_s):
                return
