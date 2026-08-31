"""Weekly memory garden: promoting what Jarvis OBSERVED into what he KNOWS.

Three stores already exist and none of them talk to each other. The
activity journal (jarvis/context.py, one JSONL file per day, kept 90) holds
every exchange, tool call, Claude result and window title -- what he
observed. facts.json plus the chromadb index (jarvis/memory.py) hold what
Hunter TOLD him. Nothing ever moved the first into the second, so six
months of use compounded into nothing.

Once a week -- at the first tick after the ISO week has closed, in the
small hours while Hunter sleeps -- this thread renders the week's journal
as a digest, hands it to the resident local model with a strict extraction
prompt (brain.extract_facts) and files what comes back through the ORDINARY
memory.remember path, tagged ``source="garden"``. Monday's first wake gets
one line; "memory report" reads the list; "forget the last garden pass"
takes it all back.

Four rules keep a hallucinating model from poisoning long-term memory:

* it never overwrites a key that already exists -- what Hunter said out
  loud always wins over what Jarvis inferred;
* at most ``garden.max_facts`` (4) land per week, so a bad week is a small
  mess rather than a large one;
* every promoted fact carries provenance, so `memory.facts_from("garden")`
  can list exactly what Jarvis made up about him;
* the undo is one sentence of voice, and it is the reason the state file
  records the values as well as the keys: a fact Hunter has since replaced
  by voice loses the tag and must survive the undo.

It is skipped, not queued, whenever the model is lent to a trainer
(brain.is_lent) or busy with a turn: the week's journal is still there next
tick, and a 26B extraction must never sit in front of a real question.
Delivery is quiet-gated for free by riding the first-wake briefing path,
which already refuses to speak inside quiet hours.
"""
from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from jarvis.config import PATHS
# One definition of "the week that just closed", shared with the weekly
# self-review so the garden and the report can never disagree about which
# seven days they are talking about.
from jarvis.dayreview import week_key as _week_key_for_date
from jarvis.logs import get_logger

log = get_logger("garden")

SOURCE = "garden"                # the provenance tag on every promoted fact
TICK_S = 900.0                   # a quarter hour, like the day reviewer
FIRST_TICK_S = 60.0
DEFAULT_MAX_FACTS = 4
# The pass wants a sleeping house and an idle GPU: before this hour, local
# time. A week that closed longer ago than LATE_DAYS runs at any hour --
# a box that is only ever awake in daylight would otherwise never garden.
DEFAULT_RUN_BEFORE_HOUR = 6
LATE_DAYS = 2
KNOWN_FACTS_CAP = 40             # how many existing keys the prompt carries
MIN_ROWS = 12                    # a week this thin has nothing to promote
DIGEST_CHARS = 6000
REPORT_ITEMS = 6

_WORDS = {0: "nothing", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
          6: "six", 7: "seven", 8: "eight", 9: "nine"}


def _word(n: int) -> str:
    return _WORDS.get(n, str(n))


def week_key(when: datetime) -> str:
    """The ISO week that has CLOSED as of ``when`` -- "2026-W34".  Called
    on a Monday it names the week just ended, which is the week the pass
    reads; called on a Wednesday it names the same one, so a late pass
    still files against the right key and never runs twice."""
    return _week_key_for_date(when.date())


def week_window(when: datetime) -> tuple[datetime, datetime]:
    """(since, until) for the closed week: Monday 00:00 to Monday 00:00."""
    midnight = when.replace(hour=0, minute=0, second=0, microsecond=0)
    monday = midnight - timedelta(days=when.weekday())
    return monday - timedelta(days=7), monday


def _norm(text) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(text or "").lower()).strip()


def _cfg(cfg, dotted: str, default):
    if cfg is None:
        return default
    try:
        value = cfg.get(dotted, default)
    except Exception:                              # noqa: BLE001
        return default
    return default if value is None else value


def filed_line(n: int, name: str = "sir") -> str:
    """The Monday-morning line.  Empty when nothing was filed: a pass that
    found nothing is a success, not an announcement."""
    if n <= 0:
        return ""
    thing = "thing" if n == 1 else "things"
    it = "it" if n == 1 else "them"
    return (f"I filed {_word(n)} {thing} from this week, {name}; "
            f"say memory report to hear {it}.")


class MemoryGarden:
    """The Sunday-night pass and the two voice commands that answer for it.

    ``extract`` is the model seam (brain.extract_facts); ``busy`` and
    ``lent`` are callables the app fills in from the brain so the thread
    never imports it.  All three are injected so the whole pass runs
    offline in tests.
    """

    def __init__(self, memory, context=None, cfg=None, extract: Optional[Callable] = None,
                 busy: Optional[Callable] = None, lent: Optional[Callable] = None,
                 state_path: Optional[Path] = None, now: Optional[Callable] = None):
        self._memory = memory
        self._context = context
        self._cfg = cfg
        self._extract = extract
        self._busy = busy
        self._lent = lent
        self._state_path = Path(state_path) if state_path else \
            PATHS.MEMORY_DIR / "garden.json"
        self._now = now or datetime.now
        self._state = self._load()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------- state
    def _load(self) -> dict:
        try:
            data = json.loads(self._state_path.read_text())
            if isinstance(data, dict):
                return data
        except (OSError, ValueError):
            log.debug("garden state unreadable", exc_info=True)
        return {}

    def _save(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_name(self._state_path.name + ".tmp")
            tmp.write_text(json.dumps(self._state, indent=1))
            os.replace(tmp, self._state_path)      # atomic: never half a file
        except OSError:
            log.debug("garden state save failed", exc_info=True)

    @property
    def enabled(self) -> bool:
        return bool(_cfg(self._cfg, "garden.enabled", True))

    @property
    def max_facts(self) -> int:
        try:
            return max(1, min(int(_cfg(self._cfg, "garden.max_facts",
                                       DEFAULT_MAX_FACTS)), 6))
        except (TypeError, ValueError):
            return DEFAULT_MAX_FACTS

    # -------------------------------------------------------- scheduling
    def due(self, now: Optional[datetime] = None) -> bool:
        """True when the closed week has no pass yet AND the hour is right
        (or the week is late enough that the hour stops mattering)."""
        if not self.enabled:
            return False
        now = now or self._now()
        if self._state.get("week") == week_key(now):
            return False
        try:
            before = int(_cfg(self._cfg, "garden.run_before_hour",
                              DEFAULT_RUN_BEFORE_HOUR))
        except (TypeError, ValueError):
            before = DEFAULT_RUN_BEFORE_HOUR
        return now.hour < before or now.weekday() >= LATE_DAYS

    def _model_free(self) -> str:
        """"" when the model may be used, else why it may not."""
        for fn, why in ((self._lent, "lent to a trainer"), (self._busy, "busy")):
            try:
                if fn is not None and fn():
                    return why
            except Exception:                      # noqa: BLE001
                log.debug("garden model gate failed", exc_info=True)
        return ""

    # ------------------------------------------------------------- pass
    def _journal_digest(self, since: datetime, until: datetime) -> tuple[str, int]:
        from jarvis.tools.journal import digest
        ctx = self._context
        if ctx is None or not hasattr(ctx, "journal_rows"):
            return "", 0
        try:
            rows = ctx.journal_rows(since, until)
        except Exception:                          # noqa: BLE001
            log.exception("garden: journal read failed")
            return "", 0
        if len(rows) < MIN_ROWS:
            return "", len(rows)
        return digest(rows, "the past week", max_chars=DIGEST_CHARS), len(rows)

    def _known(self) -> dict:
        """A COPY of the fact store.

        get_all_facts() hands back JarvisMemory's own dict, not a copy: the
        pass adds each fact it files to this map so the next proposal is
        diffed against it, and writing into the live store that way dropped
        the "time" key remember() had just written (facts.json on disk and
        the store in memory then disagreed, and `recall(since=...)` lost
        its timestamp). Read-only is not enough here -- it must be a copy."""
        try:
            facts = self._memory.get_all_facts() or {}
        except Exception:                          # noqa: BLE001
            log.exception("garden: fact listing failed")
            return {}
        return dict(facts) if isinstance(facts, dict) else {}

    def _is_new(self, key: str, value: str, facts: dict) -> bool:
        """A proposal is new when no existing key normalises to the same
        words and no existing value already contains the same sentence.
        Hand-told facts are never overwritten -- that is the whole
        safety story of the feature."""
        nkey, nval = _norm(key), _norm(value)
        if not nkey or not nval:
            return False
        for k, entry in facts.items():
            if _norm(k) == nkey:
                return False
            other = _norm(entry.get("value", "") if isinstance(entry, dict) else entry)
            if other and (other == nval or nval in other or other in nval):
                return False
        return True

    def run_pass(self, now: Optional[datetime] = None) -> dict:
        """One pass, schedule ignored (the tick and the tests both call it).
        Returns the record it filed in the state file."""
        now = now or self._now()
        key = week_key(now)
        since, until = week_window(now)
        record = {"week": key, "at": now.isoformat(timespec="seconds"),
                  "filed": [], "proposed": 0, "spoken": False, "skipped": ""}
        why = self._model_free()
        if why:
            log.info("garden: skipped, the model is %s", why)
            record["skipped"] = why
            return record                          # NOT recorded: retry next tick
        text, rows = self._journal_digest(since, until)
        if not text:
            log.info("garden: %s has %d journal rows; nothing to garden", key, rows)
            record["skipped"] = "thin week"
            with self._lock:
                self._state = record               # a thin week is still done
                self._save()
            return record
        facts = self._known()
        known = [f"{k}: {str((e or {}).get('value', ''))[:80]}"
                 for k, e in list(facts.items())[:KNOWN_FACTS_CAP]
                 if isinstance(e, dict)]
        try:
            proposals = self._extract(text, known=known, limit=self.max_facts) \
                if self._extract is not None else []
        except Exception:                          # noqa: BLE001
            log.exception("garden: extraction failed")
            proposals = []
        record["proposed"] = len(proposals or [])
        for item in (proposals or [])[:self.max_facts * 2]:
            if len(record["filed"]) >= self.max_facts:
                break
            fact_key = " ".join(str(item.get("key", "")).split()).lower()
            value = " ".join(str(item.get("value", "")).split())
            if not self._is_new(fact_key, value, facts):
                continue
            try:
                self._memory.remember(fact_key, value, source=SOURCE)
            except TypeError:
                # A memory without provenance (an old stub): file it plainly
                # rather than lose the week, and say so.
                log.warning("garden: memory.remember takes no source; "
                            "filing %r untagged", fact_key)
                self._memory.remember(fact_key, value)
            except Exception:                      # noqa: BLE001
                log.exception("garden: filing %r failed", fact_key)
                continue
            # the local copy only: the next proposal is diffed against what
            # this pass has already filed
            facts[fact_key] = {"value": value, "source": SOURCE}
            record["filed"].append({"key": fact_key, "value": value})
        log.info("garden: %s proposed %d, filed %d", key, record["proposed"],
                 len(record["filed"]))
        with self._lock:
            self._state = record
            self._save()
        return record

    def tick(self) -> Optional[dict]:
        if not self.due():
            return None
        return self.run_pass()

    # ----------------------------------------------------------- speaking
    def pending_line(self, name: str = "sir") -> str:
        """The line the first wake of the day owes him, or "" when the last
        pass filed nothing or has already been spoken."""
        with self._lock:
            if self._state.get("spoken"):
                return ""
            filed = self._state.get("filed") or []
        return filed_line(len(filed), name)

    def mark_spoken(self) -> None:
        with self._lock:
            if not self._state.get("spoken"):
                self._state["spoken"] = True
                self._save()

    def _surviving(self) -> list[dict]:
        """The last pass's facts that memory still holds under this tag --
        one Hunter has since overwritten by voice is no longer ours."""
        with self._lock:
            filed = list(self._state.get("filed") or [])
        if not filed:
            return []
        try:
            tagged = dict(self._memory.facts_from(SOURCE))
        except Exception:                          # noqa: BLE001
            log.debug("garden: provenance listing unavailable", exc_info=True)
            return filed
        out = []
        for item in filed:
            entry = tagged.get(item.get("key"))
            if isinstance(entry, dict) and _norm(entry.get("value")) == \
                    _norm(item.get("value")):
                out.append(item)
        return out

    def report_line(self, name: str = "sir") -> str:
        """"Memory report": what the last pass put in long-term memory."""
        items = self._surviving()
        if not items:
            return f"I've filed nothing from the journal yet, {name}."
        values = [str(i.get("value", "")).rstrip(".") for i in items[:REPORT_ITEMS]]
        head = f"From last week I filed {_word(len(items))}, {name}: "
        return head + "; ".join(values) + "."

    def undo(self, name: str = "sir") -> str:
        """"Forget the last garden pass": every fact of it that is still
        mine goes; anything Hunter has re-told since stays."""
        items = self._surviving()
        if not items:
            return f"There's nothing from the last pass to forget, {name}."
        gone = 0
        for item in items:
            try:
                self._memory.forget(item.get("key"))
                gone += 1
            except Exception:                      # noqa: BLE001
                log.exception("garden: forgetting %r failed", item.get("key"))
        with self._lock:
            self._state["filed"] = []
            self._state["spoken"] = True
            self._save()
        log.info("garden: undo removed %d facts", gone)
        thing = "thing" if gone == 1 else "things"
        return f"Forgotten, {name}; {_word(gone)} {thing} out of memory."

    # ----------------------------------------------------------- thread
    def start(self) -> None:
        # Alive-guard + clear, like presence.py: a stopped instance can be
        # started again (tests, a config reload), and a dead thread must
        # not block a fresh one.
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="garden")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            # a pass in flight holds the local model; it must not outlive
            # stop_assistant into the teardown
            t.join(timeout=5.0)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        if self._stop.wait(FIRST_TICK_S):
            return
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                log.exception("garden tick failed")
            if self._stop.wait(TICK_S):
                return


__all__ = ["MemoryGarden", "SOURCE", "week_key", "week_window", "filed_line",
           "DEFAULT_MAX_FACTS"]
