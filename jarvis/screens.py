"""Where his screens are, learned from how he uses them. Coordinates only.

Hunter asked: "for the gesture control, cant we train the camera to
understand where the third monitor is or something?" -- and that question
revealed there are THREE physical screens, not two.

THE CAMERA CANNOT SEE THE SCREENS, AND THAT IS THE FIRST THING TO SAY.
Every line of evidence in this repo says the lens points at HIM:
``AttentionTracker`` measures whether his head is turned toward the camera,
``faceenrol`` walks HIS yaw range, ``camera.cone_centre_deg`` ships 0.0 (the
cone is aimed down the lens axis), and docs/vision.md s9 mounts the camera
beside a monitor at eye level pointed across at his face. So this module
does NOT learn where the screens are. It learns WHERE HE REACHES AND WHERE
HE LOOKS when he is using each one, which is the honest reading of his idea
and the better feature: it survives him moving a monitor, moving his chair
or re-seating himself, and it removes the "tell me which side HPCOMPUTER is
on" ritual that ``gesture.sinks`` ships empty waiting for.

THEN THE ARITHMETIC COLLAPSED THE PROBLEM. There are three screens but only
TWO MACHINES: the Spark drives the right-hand screen, HPCOMPUTER drives the
middle and the left. The destination of a cast is a MACHINE, not a screen.
Enumerating (source machine x throw direction) gives four rows and only two
of them can fire:

    Spark (right screen)       throw left   -> HPCOMPUTER    FIRES  <- his ex 1
    Spark (right screen)       throw right  -> nothing there  REFUSE
    HPCOMPUTER (middle/left)   throw right  -> Spark         FIRES  <- his ex 2
    HPCOMPUTER (middle/left)   throw left   -> nothing there  REFUSE

Both of HIS OWN EXAMPLES are the two firing rows, so this is his rule, not a
substitution for it. Two consequences fall straight out and both are pinned
in tests/test_screens.py rather than argued here:

1. The source only has to resolve to a MACHINE -- two zones across the
   frame, not three. Three zones would have been marginal (zone half-width
   0.762 hand-units against the only placement number this repo has ever
   measured, 0.60, leaves 20 mm of margin); two zones give 1.143 units and
   are 97% correct even at the worst placement error I can justify.
2. A one-step source error CANNOT cast the wrong desktop. The two valid
   directions are DISJOINT between the two machines, so a source flip
   always flips into a refusal. That, and not the tightness of the gesture,
   is why firing straight through with no confirmation is safe.

THE CORRECTION I OWE. I was handed ``anchor_drift_u = 0.60`` as "his
measured placement error". It is not: gesture.py documents it as the DWELL
STILLNESS radius, justified as roughly 7x the landmark noise floor. How
accurately he puts his hand at a named screen has never been measured, on
him or on anyone, in this project. ``scripts/screen_selfcheck.py`` is the
instrument that turns it into a number; until he runs it, every separation
figure here is arithmetic over a constant measured for a different purpose.

TWO DESIGN DECISIONS THAT MATTER MORE THAN THEY LOOK.

* THE CLASSIFIER LIVES ON RAW ``yaw_t``, NEVER ON ``yaw_deg``. ``yaw_deg``
  is ``atan(yaw_t / NOSE_RATIO)`` and NOSE_RATIO = 0.35 is documented in
  visionrig.py as an assumption about adult anatomy, not a measurement of
  him. Working in yaw_t removes that assumption from the decision entirely:
  a wrong nose ratio warps the axis monotonically and cannot make two
  separated clusters overlap. Degrees stay for the console readout, and
  ``yaw_t_from_deg`` is the exact inverse used to get back to the raw
  measure from the only field ``campreview.PreviewFace`` carries.
* MEDIANS AND IQRs, NEVER MEANS AND STANDARD DEVIATIONS. The label
  ("which machine is he on") is a proxy and it is sometimes wrong -- he can
  look at the Spark while his mouse rests on Windows. A minority of wrong
  labels must move nothing, and that is what the median buys.

WHAT THIS MODULE WILL NOT DO. It holds no frame, no landmark array, no
image, no clock but the one passed in, and no transport. It never delivers
anything: it names a machine and a margin, and ``jarvis/gesturecast.py``
decides what that means. Refusing to arm is a first-class outcome here and
not an error, because the fallback -- the board -- is a real, working
destination.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from jarvis.gesture import MIN_PALM_DIAG_PX, to_his_frame
from jarvis.logs import get_logger
from jarvis.visionrig import NOSE_RATIO

log = get_logger("screens")

VERSION = 1

# Config keys. Both ship absent, which means OFF and UNLEARNED.
OPTION_MAP = "gesture.screens"
OPTION_ENABLED = "gesture.cast_screens"

# The two machines. This tuple is the destination set, and the misfire
# analysis in the design is only valid while it has exactly two entries --
# see ``build`` and test_a_third_machine_refuses_to_arm.
SPARK = "spark"
HPCOMPUTER = "hpcomputer"
MACHINES = (HPCOMPUTER, SPARK)

# His specification, enumerated. The grab names the SOURCE, the throw names
# the DESTINATION, and "the next screen in that direction FROM the one he
# grabbed" reduces to exactly these two rows once the destination is read as
# a machine. Everything not in here REFUSES: there is nothing that way.
ROUTES = {(SPARK, "left"): HPCOMPUTER,
          (HPCOMPUTER, "right"): SPARK}

# How confident the source has to be before a cast may fire. Under this the
# throw goes to the board and nothing is asked of him -- a REFUSAL, not a
# confirmation. This is the "make the gesture harder to trigger" option,
# chosen over reinstating a prompt he overruled.
MIN_MARGIN_SIGMA = 2.0
# The same bar, applied pairwise at arming time.
MIN_PAIR_SIGMA = 2.0
# Refuse to conclude on fewer than this per screen. The sample-size
# arithmetic says 21 calls a 1.0-sigma separation and 9 calls a 1.5-sigma
# one, so 25 is comfortable and covers a mid-session seat shift.
MIN_SAMPLES = 25
# A sample counts only once the label has held steady this long, because
# "which machine is the mouse on" flickers and a flicker is not a session.
LABEL_STABLE_S = 3.0

# sigma = IQR / 1.349 for a normal. The clusters are not guaranteed normal;
# this is a robust SCALE, used only to put the two axes in the same units.
IQR_TO_SIGMA = 1.349
# Spread floors. Two clusters 0.02 units apart with a zero IQR would divide
# by nothing and read as infinitely separated, so a map learned from three
# identical frames could arm itself. These are the smallest spreads worth
# believing: a fifth of the dwell stillness radius on the hand axis, and the
# yaw swing of one detect pixel at his desk on the yaw axis.
MIN_SIGMA_U = 0.12
MIN_SIGMA_T = 0.02

UNLEARNED_LINE = ("I haven't learnt where your screens are yet, sir. "
                  "Everything goes to the board.")
UNSEPARATED_LINE = ("I can't tell your {a} screen from the {b} one by where "
                    "you reach, sir. Everything goes to the board until you "
                    "tell me a side.")
LAYOUT_CHANGED_LINE = ("Your screens have moved, sir. I've forgotten where "
                       "they were; throws go to the board until I've "
                       "learnt them again.")


# ------------------------------------------------------------- geometry
def hand_x_u(cx: float, palm_diag: float, frame_w: float,
             mirrored: bool = False) -> float:
    """The palm's lateral position in HAND-UNITS, signed into HIS frame.

    Positive is HIS RIGHT. The sign comes from ``gesture.to_his_frame`` and
    from nowhere else: a second copy of that mapping is precisely how the
    side of the room gets inverted, which happened twice during the first
    design pass with entirely plausible-looking output both times.

    A collapsed hand (``palm_diag`` at or below the floor) has NO OPINION
    and returns 0.0 rather than an unbounded one.
    """
    unit = float(palm_diag or 0.0)
    if unit <= MIN_PALM_DIAG_PX:
        return 0.0
    his_right, _up = to_his_frame(float(cx) - float(frame_w) / 2.0, 0.0,
                                  bool(mirrored))
    return his_right / unit


def yaw_t_from_deg(deg: float, nose_ratio: float = NOSE_RATIO) -> float:
    """``yaw_deg`` back to the RAW ratio it was derived from.

    ``visionrig.HeadModel.yaw_deg`` is ``atan(t / nose_ratio)``, so this is
    its exact inverse. It exists because ``campreview.PreviewFace`` carries
    only the derived degrees, and every decision in this module is taken on
    the raw measure -- see the module docstring. Negative is turned toward
    HIS RIGHT.
    """
    try:
        d = float(deg)
    except (TypeError, ValueError):
        return 0.0
    if d != d or abs(d) >= 89.9:
        return 0.0
    return float(nose_ratio) * math.tan(math.radians(d))


def sigma_u(iqr: float) -> float:
    """A robust spread on the hand axis, floored. See MIN_SIGMA_U."""
    return max(abs(float(iqr or 0.0)) / IQR_TO_SIGMA, MIN_SIGMA_U)


def sigma_t(iqr: float) -> float:
    return max(abs(float(iqr or 0.0)) / IQR_TO_SIGMA, MIN_SIGMA_T)


def _pooled(a: float, b: float) -> float:
    return math.sqrt((float(a) ** 2 + float(b) ** 2) / 2.0)


def _split(lo_med: float, lo_sig: float,
           hi_med: float, hi_sig: float) -> float:
    """The point between two clusters that is equally many sigmas from
    each. With equal spreads it is the midpoint; with unequal ones it sits
    nearer the tighter cluster, which is where the honest boundary is."""
    total = float(lo_sig) + float(hi_sig)
    if total <= 0.0:
        return (float(lo_med) + float(hi_med)) / 2.0
    return (float(lo_med) * float(hi_sig)
            + float(hi_med) * float(lo_sig)) / total


# --------------------------------------------------------------- the map
@dataclass(frozen=True)
class Screen:
    """One physical screen, as the numbers he generated using it.

    ``name`` is a short label he would recognise ("right", "middle",
    "left"); ``machine`` is which box drives it. Everything else is a
    median and an inter-quartile range -- never a mean and never a trace of
    what was on it.
    """

    name: str
    machine: str
    hand_x_u_p50: float = 0.0
    hand_x_u_iqr: float = 0.0
    yaw_t_p50: float = 0.0
    yaw_t_iqr: float = 0.0
    n: int = 0

    def numbers_only(self) -> dict:
        return {"name": str(self.name), "machine": str(self.machine),
                "hand_x_u_p50": round(float(self.hand_x_u_p50), 4),
                "hand_x_u_iqr": round(float(self.hand_x_u_iqr), 4),
                "yaw_t_p50": round(float(self.yaw_t_p50), 4),
                "yaw_t_iqr": round(float(self.yaw_t_iqr), 4),
                "n": int(self.n)}


@dataclass(frozen=True)
class ScreenMap:
    """The learned room. Armed or not, and it says why not.

    DISARMED IS NOT BROKEN. With no map the feature falls back to
    direction-only, which -- because the two valid directions are disjoint
    between the two machines -- still routes both of his examples
    correctly. The map's whole job is to be the veto.
    """

    screens: tuple = ()
    boundary_u: float = 0.0
    boundary_t: float = 0.0
    sigma_u: float = MIN_SIGMA_U
    sigma_t: float = MIN_SIGMA_T
    margin_sigma: float = 0.0
    armed: bool = False
    reason: str = ""
    unseparated: tuple = ()
    right_machine: str = ""
    at: float = 0.0
    source: str = ""
    layout: str = ""
    probe_r: float = 0.0
    version: int = VERSION

    @property
    def machines(self) -> tuple:
        return tuple(sorted({s.machine for s in self.screens}))

    def machine_of(self, name: str) -> str:
        for s in self.screens:
            if s.name == name:
                return s.machine
        return ""

    def numbers_only(self) -> dict:
        return to_config(self)


@dataclass(frozen=True)
class Verdict:
    """What a grab resolved to, and how confidently.

    ``machine`` is "" whenever the answer is "I don't know" -- an unarmed
    map, a grab on the boundary, or the head and the hand naming different
    machines. That is the cheap, reversible outcome this whole lane chooses
    everywhere: the throw goes to the board and nothing is asked of him.
    """

    machine: str = ""
    margin_sigma: float = 0.0
    hand_sigma: float = 0.0
    yaw_sigma: float = 0.0
    yaw_used: bool = False
    agreed: bool = True
    why: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.machine) and self.margin_sigma >= MIN_MARGIN_SIGMA

    def numbers_only(self) -> dict:
        return {"machine": self.machine,
                "margin_sigma": round(float(self.margin_sigma), 3),
                "hand_sigma": round(float(self.hand_sigma), 3),
                "yaw_sigma": round(float(self.yaw_sigma), 3),
                "yaw_used": bool(self.yaw_used), "agreed": bool(self.agreed),
                "why": self.why}


def _by_machine(screens: Sequence[Screen]) -> dict:
    out: dict = {}
    for s in screens:
        out.setdefault(s.machine, []).append(s)
    return out


def separation(a: Screen, b: Screen) -> tuple:
    """``(hand sigmas, yaw sigmas)`` between two screens.

    Pooled within-cluster spread on each axis, floored. Two axes rather
    than one because the design's whole hypothesis is that head yaw is a
    genuine SECOND axis -- possibly the stronger one -- and a pair only has
    to separate on ONE of them to be usable.
    """
    du = abs(float(a.hand_x_u_p50) - float(b.hand_x_u_p50))
    dt = abs(float(a.yaw_t_p50) - float(b.yaw_t_p50))
    su = _pooled(sigma_u(a.hand_x_u_iqr), sigma_u(b.hand_x_u_iqr))
    st = _pooled(sigma_t(a.yaw_t_iqr), sigma_t(b.yaw_t_iqr))
    return du / su, dt / st


def build(screens: Sequence[Screen], *, at: float = 0.0, source: str = "",
          layout: str = "", probe_r: float = 0.0) -> ScreenMap:
    """Screens in, an armed-or-not map out. The arming rules, in order.

    1. EXACTLY TWO MACHINES. Not one (there is nothing to route between)
       and never three: the disjoint-direction safety property that lets a
       cast fire with no confirmation dies the moment a third target
       exists, so a third machine refuses to arm rather than quietly making
       the misfire analysis void.
    2. ENOUGH SAMPLES PER SCREEN.
    3. NO CROSS-MACHINE PAIR TOO CLOSE ON BOTH AXES. A pair that is too
       close but sits INSIDE one machine costs nothing -- middle and left
       are both HPCOMPUTER and the design never needed to tell them apart.
    4. THE MACHINE BOUNDARY ITSELF CLEARS THE BAR on at least one axis.
    """
    rows = tuple(s for s in screens if isinstance(s, Screen) and s.machine)
    groups = _by_machine(rows)
    if sorted(groups) != sorted(MACHINES):
        return ScreenMap(screens=rows, at=float(at), source=str(source),
                         layout=str(layout), probe_r=float(probe_r),
                         reason="I need to see both machines used, and only "
                                "those two machines, before I can tell them "
                                "apart (there must be exactly two machines)")
    thin = [s.name for s in rows if int(s.n) < MIN_SAMPLES]
    if thin:
        return ScreenMap(screens=rows, at=float(at), source=str(source),
                         layout=str(layout), probe_r=float(probe_r),
                         reason="I haven't watched you use %s enough yet "
                                "(I want %d looks at each screen)"
                                % (", ".join(sorted(thin)), MIN_SAMPLES))

    unsep = []
    blocked = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            du, dt = separation(a, b)
            if du < MIN_PAIR_SIGMA and dt < MIN_PAIR_SIGMA:
                unsep.append((a.name, b.name))
                if a.machine != b.machine:
                    blocked.append((a.name, b.name))

    stats = {}
    for name, group in groups.items():
        stats[name] = (
            statistics.median([s.hand_x_u_p50 for s in group]),
            _pooled(*(sigma_u(s.hand_x_u_iqr) for s in group[:2]))
            if len(group) > 1 else sigma_u(group[0].hand_x_u_iqr),
            statistics.median([s.yaw_t_p50 for s in group]),
            _pooled(*(sigma_t(s.yaw_t_iqr) for s in group[:2]))
            if len(group) > 1 else sigma_t(group[0].yaw_t_iqr),
        )
    a_name, b_name = sorted(groups)
    au, asu, at_, ast = stats[a_name]
    bu, bsu, bt_, bst = stats[b_name]
    right = a_name if au > bu else b_name
    boundary_u = _split(min(au, bu), asu if au < bu else bsu,
                        max(au, bu), bsu if au < bu else asu)
    boundary_t = _split(min(at_, bt_), ast if at_ < bt_ else bst,
                        max(at_, bt_), bst if at_ < bt_ else ast)
    pooled_u = _pooled(asu, bsu)
    pooled_t = _pooled(ast, bst)
    margin = max(abs(au - bu) / pooled_u / 2.0,
                 abs(at_ - bt_) / pooled_t / 2.0)

    reason = ""
    if blocked:
        a, b = blocked[0]
        reason = UNSEPARATED_LINE.format(a=a, b=b)
    elif margin < MIN_PAIR_SIGMA:
        reason = ("your two machines sit too close together for me to tell "
                  "them apart yet")
    return ScreenMap(screens=rows, boundary_u=boundary_u,
                     boundary_t=boundary_t, sigma_u=pooled_u,
                     sigma_t=pooled_t, margin_sigma=margin,
                     armed=not reason, reason=reason,
                     unseparated=tuple(unsep), right_machine=right,
                     at=float(at), source=str(source), layout=str(layout),
                     probe_r=float(probe_r))


def unarmed_line(m: Optional[ScreenMap]) -> str:
    """One short line for why nothing is being routed, or "" when it is."""
    if m is None or not m.screens:
        return UNLEARNED_LINE
    if m.armed:
        return ""
    if m.reason.startswith("I can't tell"):
        return m.reason
    return UNLEARNED_LINE


def score(m: Optional[ScreenMap], hand_u: float,
          yaw_t: Optional[float] = None) -> Verdict:
    """A grab's numbers -> which machine he grabbed at, and how confidently.

    The hand decides and the head checks. Whichever axis is stronger sets
    the margin (the design's pass bar allows for yaw turning out to be the
    ONLY axis rather than the second one, and this reads whichever wins
    without a rewrite). A CONFIDENT DISAGREEMENT between the two refuses
    the grab outright: that is the tripwire standing between a map learned
    from a systematically skewed label set and a cast to the wrong machine.
    A weak yaw reading ABSTAINS rather than vetoing.
    """
    if m is None or not m.armed:
        return Verdict(why="nothing learned")
    x = float(hand_u)
    hand_margin = abs(x - m.boundary_u) / m.sigma_u
    hand_machine = m.right_machine if x > m.boundary_u else \
        [n for n in m.machines if n != m.right_machine][0]

    yaw_used = False
    yaw_margin = 0.0
    yaw_machine = ""
    if yaw_t is not None:
        t = float(yaw_t)
        yaw_margin = abs(t - m.boundary_t) / m.sigma_t
        # Negative yaw_t is turned toward HIS RIGHT; the right-hand
        # machine's cluster is whichever side of the yaw boundary its own
        # median fell, which is derived, never assumed.
        right_is_low_t = _machine_yaw_median(m, m.right_machine) < m.boundary_t
        low = m.right_machine if right_is_low_t else \
            [n for n in m.machines if n != m.right_machine][0]
        high = [n for n in m.machines if n != low][0]
        yaw_machine = low if t < m.boundary_t else high
        yaw_used = True

    if yaw_used and yaw_margin >= MIN_MARGIN_SIGMA and \
            hand_margin >= MIN_MARGIN_SIGMA and yaw_machine != hand_machine:
        return Verdict(margin_sigma=0.0, hand_sigma=hand_margin,
                       yaw_sigma=yaw_margin, yaw_used=True, agreed=False,
                       why="your hand and your head disagree")
    margin = hand_margin
    if yaw_used and yaw_machine == hand_machine:
        margin = max(hand_margin, yaw_margin)
    return Verdict(machine=hand_machine, margin_sigma=margin,
                   hand_sigma=hand_margin, yaw_sigma=yaw_margin,
                   yaw_used=yaw_used, agreed=True)


def _machine_yaw_median(m: ScreenMap, machine: str) -> float:
    rows = [s.yaw_t_p50 for s in m.screens if s.machine == machine]
    return statistics.median(rows) if rows else 0.0


def route(source: str, direction: str) -> str:
    """(source machine, throw direction) -> destination machine, or "".

    HIS RULE, and the whole of it. "" is not a failure: it is "there is
    nothing that way", which is a refusal he hears as one short line.
    """
    src = str(source or "").strip().lower()
    word = str(direction or "").strip().lower()
    return ROUTES.get((src, word), "")


def layout_changed(m: Optional[ScreenMap], layout: str) -> bool:
    """Did a monitor get added, removed or moved?

    An exact tripwire, from the machine that owns the layout. Nothing said
    (an empty string) is NOT news -- the Windows poller may simply not be
    running, and a map must not disarm itself because nobody is talking.
    """
    now = str(layout or "").strip()
    if not now or m is None or not m.layout:
        return False
    return now != str(m.layout).strip()


# -------------------------------------------------------------- the store
def to_config(m: ScreenMap) -> dict:
    """The stored form: numbers and short names only.

    This goes into ``~/.config/jarvis/assistant.json`` under
    ``gesture.screens``. There is no frame, no window title, no path and no
    per-minute activity trace in it, and nothing here could reconstruct
    what he was doing. ``visionrig.assert_numbers_only`` is asserted over
    it in the suite rather than promised in this comment.
    """
    return {"version": int(m.version), "at": round(float(m.at), 3),
            "armed": bool(m.armed), "source": str(m.source),
            "boundary_u": round(float(m.boundary_u), 4),
            "boundary_t": round(float(m.boundary_t), 4),
            "margin_sigma": round(float(m.margin_sigma), 3),
            "layout": str(m.layout), "probe_r": round(float(m.probe_r), 4),
            "screens": [s.numbers_only() for s in m.screens]}


def from_config(raw) -> Optional[ScreenMap]:
    """A stored map back, or None for "nothing learned".

    ARMED IS RE-DERIVED, NEVER TAKEN ON TRUST. A file hand-edited to
    ``armed: true`` with one machine in it must not route, so the rows are
    fed back through ``build`` and the arming rules run again.
    """
    if not isinstance(raw, dict):
        return None
    if int(raw.get("version", 0) or 0) != VERSION:
        return None
    rows = raw.get("screens")
    if not isinstance(rows, list) or not rows:
        return None
    screens = []
    for row in rows:
        if not isinstance(row, dict):
            return None
        name = str(row.get("name") or "").strip()
        machine = str(row.get("machine") or "").strip().lower()
        if not name or not machine:
            return None
        try:
            screens.append(Screen(
                name=name, machine=machine,
                hand_x_u_p50=float(row.get("hand_x_u_p50", 0.0)),
                hand_x_u_iqr=float(row.get("hand_x_u_iqr", 0.0)),
                yaw_t_p50=float(row.get("yaw_t_p50", 0.0)),
                yaw_t_iqr=float(row.get("yaw_t_iqr", 0.0)),
                n=int(row.get("n", 0))))
        except (TypeError, ValueError):
            return None
    return build(screens, at=float(raw.get("at", 0.0) or 0.0),
                 source=str(raw.get("source") or ""),
                 layout=str(raw.get("layout") or ""),
                 probe_r=float(raw.get("probe_r", 0.0) or 0.0))


def load(get_option: Optional[Callable]) -> Optional[ScreenMap]:
    if not callable(get_option):
        return None
    try:
        raw = get_option(OPTION_MAP, None)
    except Exception:                          # noqa: BLE001 - config edge
        log.debug("screens: the map could not be read", exc_info=True)
        return None
    return from_config(raw)


def save(m: ScreenMap, set_option: Optional[Callable]) -> bool:
    if not callable(set_option):
        return False
    try:
        return bool(set_option(OPTION_MAP, to_config(m)))
    except Exception:                          # noqa: BLE001 - config edge
        log.warning("screens: the map could not be saved", exc_info=True)
        return False


def enabled(get_option: Optional[Callable]) -> bool:
    """OFF BY DEFAULT, and it stays off until he says otherwise.

    Casting a screen puts a desktop on a monitor in his office; the board
    cast this feature sits beside costs three seconds of his attention. The
    two do not deserve the same default.
    """
    if not callable(get_option):
        return False
    try:
        return bool(get_option(OPTION_ENABLED, False))
    except Exception:                          # noqa: BLE001 - config edge
        return False


# ------------------------------------------------------------ the learner
@dataclass
class _Bucket:
    hand: list = field(default_factory=list)
    yaw: list = field(default_factory=list)


def _iqr(values: Sequence[float]) -> float:
    if len(values) < 4:
        return 0.0
    ordered = sorted(float(v) for v in values)
    half = len(ordered) // 2
    lo = statistics.median(ordered[:half])
    hi = statistics.median(ordered[-half:])
    return abs(hi - lo)


class ScreenLearner:
    """Rows in, a map out. No clock of its own; every call carries its own
    ``at``.

    THE LABEL IS WEAK AND THIS DOES NOT PRETEND OTHERWISE. "Which machine
    is the mouse on" is a proxy for "which screen is he looking at", and he
    can look at the Spark while his mouse rests on Windows. Two defences,
    both cheap: a sample counts only once the label has held steady for
    ``LABEL_STABLE_S``, and every statistic is a MEDIAN or an IQR, so a
    minority of wrong labels moves nothing.
    """

    def __init__(self, *, stable_s: float = LABEL_STABLE_S,
                 cap: int = 2000) -> None:
        self.stable_s = float(stable_s)
        self.cap = int(cap)
        self._rows: dict = {}
        self._label = ""
        self._since = 0.0
        self.seen = 0
        self.skipped = 0

    def observe(self, *, at: float, label: str, hand_u: float,
                yaw_t: Optional[float] = None) -> bool:
        """One labelled sample. True when it was kept."""
        name = str(label or "").strip()
        t = float(at)
        if not name:
            self._label, self._since = "", t
            return False
        if name != self._label:
            self._label, self._since = name, t
            self.skipped += 1
            return False
        if (t - self._since) < self.stable_s:
            self.skipped += 1
            return False
        bucket = self._rows.setdefault(name, _Bucket())
        if len(bucket.hand) >= self.cap:
            bucket.hand.pop(0)
            if bucket.yaw:
                bucket.yaw.pop(0)
        bucket.hand.append(float(hand_u))
        if yaw_t is not None:
            bucket.yaw.append(float(yaw_t))
        self.seen += 1
        return True

    def counts(self) -> dict:
        return {name: len(b.hand) for name, b in self._rows.items()
                if b.hand}

    def screen(self, name: str, machine: str) -> Screen:
        bucket = self._rows.get(str(name), _Bucket())
        hand = bucket.hand
        yaw = bucket.yaw
        return Screen(name=str(name), machine=str(machine),
                      hand_x_u_p50=statistics.median(hand) if hand else 0.0,
                      hand_x_u_iqr=_iqr(hand),
                      yaw_t_p50=statistics.median(yaw) if yaw else 0.0,
                      yaw_t_iqr=_iqr(yaw), n=len(hand))

    def build(self, machines: dict, *, at: float = 0.0, source: str = "",
              layout: str = "", probe_r: float = 0.0) -> ScreenMap:
        """``{screen name: machine}`` -> a map. A label with no machine is
        DROPPED, never guessed at: an unknown screen is not evidence."""
        rows = [self.screen(name, machines[name])
                for name in sorted(self._rows)
                if name in machines and machines[name] in MACHINES]
        return build(rows, at=float(at), source=str(source),
                     layout=str(layout), probe_r=float(probe_r))

    def numbers_only(self) -> dict:
        return {"seen": int(self.seen), "skipped": int(self.skipped),
                "counts": self.counts(), "stable_s": float(self.stable_s)}


__all__ = [
    "HPCOMPUTER", "IQR_TO_SIGMA", "LABEL_STABLE_S", "LAYOUT_CHANGED_LINE",
    "MACHINES", "MIN_MARGIN_SIGMA", "MIN_PAIR_SIGMA", "MIN_SAMPLES",
    "MIN_SIGMA_T", "MIN_SIGMA_U", "OPTION_ENABLED", "OPTION_MAP", "ROUTES",
    "SPARK", "UNLEARNED_LINE", "UNSEPARATED_LINE", "VERSION", "Screen",
    "ScreenLearner", "ScreenMap", "Verdict", "build", "enabled",
    "from_config", "hand_x_u", "layout_changed", "load", "route", "save",
    "score", "separation", "sigma_t", "sigma_u", "to_config",
    "unarmed_line", "yaw_t_from_deg",
]
