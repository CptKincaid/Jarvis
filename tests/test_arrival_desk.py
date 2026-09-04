"""GREET AT THE DOOR, ASK AT THE DESK -- the pure half.

His flat is a corridor: front door -> KITCHEN -> office. The office is
only reachable through the kitchen, so the kitchen going occupied after an
absence is the door opening and the office is where he lands. Today both
halves of the arrival cue fire on that first step, which puts "shall I go
through your mail" on him while he is still taking his shoes off.

So the cue SPLITS. panel -> earcon -> greeting stay at the door; the
catch-up is armed there and delivered when he SETTLES. This file pins the
rules with no phone, no radar, no lens and no app -- the same promise
jarvis/arrival.py has always made -- and tests/test_arrival_desk_app.py
pins the wiring.

NOTHING HERE EVER SEES A FRAME. The camera reaches this feature as an
identity LABEL and a score; there is no argument anywhere in the module
under test that an image could be passed through.
"""
import threading

from jarvis import arrival as arrival_mod


class FakeVerdict:
    """The shape jarvis/zones.Verdict presents: a zone, the rule that
    decided it, and the room. Duck-typed on purpose -- arrival.py must not
    import zones.py, which owns a log file and a config reader."""

    def __init__(self, zone, rule="band", room="office"):
        self.zone, self.rule, self.room = zone, rule, room


class FakeCamera:
    """jarvis/zones.CameraOpinion: a name and a yes/no, and no field an
    image could live in."""

    def __init__(self, known=True, label="Hunter"):
        self.known, self.label = known, label


DESK = arrival_mod.DEFAULT_DESK_ZONE
OFF_DESK = "empty space"


# ==================================================================
# THE SPLIT ITSELF
# ==================================================================
def test_the_door_and_the_desk_together_are_still_the_whole_cue():
    """The split may not lose a step. DOOR_STEPS + DESK_STEPS is exactly
    ARRIVAL_STEPS, in order, so a step added to the contract and to
    neither half fails here instead of silently never running."""
    assert arrival_mod.DOOR_STEPS + arrival_mod.DESK_STEPS == \
        arrival_mod.ARRIVAL_STEPS


def test_the_ordering_contract_is_unchanged():
    assert arrival_mod.ARRIVAL_STEPS == ("panel", "earcon", "greeting",
                                         "catch-up")


def test_deferring_leaves_the_greeting_where_it_was():
    """He still gets panel -> earcon -> greeting at the door, in order."""
    assert arrival_mod.arrival_plan(returned=True, defer_catch_up=True) == \
        ["panel", "earcon", "greeting"]


def test_not_deferring_is_the_cue_exactly_as_it_shipped():
    assert arrival_mod.arrival_plan(returned=True) == \
        ["panel", "earcon", "greeting", "catch-up"]


def test_deferring_still_honours_the_earcon_switch():
    assert arrival_mod.arrival_plan(returned=True, cue=False,
                                    defer_catch_up=True) == ["panel", "greeting"]


def test_a_quiet_house_is_panel_only_whether_or_not_it_defers():
    """quiet.py's answer is honoured exactly as it always was: the panel
    comes up and the voice waits for the policy's own tick."""
    for defer in (False, True):
        assert arrival_mod.arrival_plan(returned=True, quiet_reason="a meeting",
                                        defer_catch_up=defer) == ["panel"]


def test_the_desk_plan_is_the_catch_up_and_nothing_else():
    assert arrival_mod.settle_plan() == ["catch-up"]


def test_the_desk_plan_is_EMPTY_while_the_policy_is_holding_his_tongue():
    """THE ONE NEW WAY TO SPEAK OVER A QUIET HOUR. At the door the plan is
    panel-only so the catch-up never ran during quiet hours at all;
    deferred, it would arrive at the desk an hour later and _say passes it
    with proactive=False, which pierces the hold. So the desk asks the
    policy too, and a held catch-up stays OWED rather than being spoken."""
    assert arrival_mod.settle_plan(quiet_reason="quiet hours") == []


def test_you_are_out_is_stale_at_the_desk_too_and_defers_nothing():
    """The same rule arrival_plan has always had: he is demonstrably in
    the room, so "you're out" is the one reason that is wrong by the time
    it is read."""
    assert arrival_mod.settle_plan(quiet_reason="you're out") == ["catch-up"]


# ==================================================================
# WHETHER TO DEFER AT ALL -- a deferral no leg can fire is a lost catch-up
# ==================================================================
def test_with_no_leg_live_the_catch_up_is_NOT_deferred():
    """The failure this exists to refuse: no zone source and no camera
    means nothing can ever say he settled, so deferring would be the
    catch-up silently never happening. It asks at the door instead."""
    assert arrival_mod.defer_catch_up(legs=()) is False


def test_one_live_leg_is_enough_to_defer():
    assert arrival_mod.defer_catch_up(legs=("radar",)) is True
    assert arrival_mod.defer_catch_up(legs=("camera",)) is True


def test_the_split_can_be_switched_off_with_both_legs_live():
    assert arrival_mod.defer_catch_up(legs=("radar", "camera"),
                                      enabled=False) is False


# ==================================================================
# SETTLED: which leg, and it may never lie about which
# ==================================================================
def test_the_desk_band_settles_him_by_the_RADAR():
    assert arrival_mod.settle(room="office",
                              verdict=FakeVerdict(DESK)) == "radar"


def test_a_band_that_is_not_the_desk_settles_nothing():
    """He is in the office by the door, still walking. 0.75-2.25 m is
    "empty space" on the live map and it is not where he sits."""
    assert arrival_mod.settle(room="office", verdict=FakeVerdict(OFF_DESK)) == ""


def test_the_kitchen_never_settles_him_however_the_verdict_reads():
    """The catch-up is owed at his DESK. A kitchen zone map that happens
    to name a band "at the desk" is not his desk."""
    assert arrival_mod.settle(room="kitchen",
                              verdict=FakeVerdict(DESK, room="kitchen")) == ""


def test_a_camera_ruled_verdict_is_reported_as_the_CAMERA():
    """THE LIE THIS REFUSES. zones.verdict lets the camera overrule the
    radar, and a camera-ruled verdict carries the desk zone -- so reading
    the zone alone would credit the radar for a decision the lens made,
    with his back to the radar and no range reading involved at all."""
    assert arrival_mod.settle(room="office",
                              verdict=FakeVerdict(DESK, rule="camera")) == "camera"


def test_a_bare_camera_opinion_settles_him_with_no_radar_at_all():
    """The office module unplugged, or reading nothing with his back to
    it: the lens naming him in the office is the whole of the evidence."""
    assert arrival_mod.settle(room="office", camera=FakeCamera()) == "camera"


def test_an_identity_LABEL_is_accepted_where_an_opinion_is():
    """jarvis/app._eye_identity hands out a NAME and nothing else. That
    string is the entire camera-side interface of this feature."""
    assert arrival_mod.settle(room="office", camera="Hunter") == "camera"


def test_a_lens_that_recognised_NOBODY_settles_nothing():
    """known=False is "I looked and named no one", which is not a claim
    that the chair is empty and is certainly not a claim it is him."""
    assert arrival_mod.settle(room="office", camera=FakeCamera(known=False)) == ""


def test_a_lens_with_no_opinion_settles_nothing():
    for nothing in (None, "", "   "):
        assert arrival_mod.settle(room="office", camera=nothing) == ""


def test_with_BOTH_legs_true_at_once_the_camera_is_named():
    """Both are honest and the naming has to be deterministic. The lens
    identified HIM; the radar saw a body in a band. The stronger claim is
    the one the log records."""
    assert arrival_mod.settle(room="office", verdict=FakeVerdict(DESK),
                              camera=FakeCamera()) == "camera"


def test_a_plain_zone_STRING_is_accepted_as_the_verdict():
    assert arrival_mod.settle(room="office", verdict=DESK) == "radar"
    assert arrival_mod.settle(room="office", verdict=OFF_DESK) == ""


def test_the_verdict_names_its_own_room_when_the_caller_does_not():
    assert arrival_mod.settle(room="", verdict=FakeVerdict(DESK)) == "radar"
    assert arrival_mod.settle(room="", verdict=FakeVerdict(DESK,
                                                           room="kitchen")) == ""


def test_the_room_is_slugged_exactly_as_the_fabric_slugs_it():
    """A hand-written config key and a fabric event name have to meet --
    the same rule _room_key already enforces for the door room."""
    for spelling in ("Office", "  office  ", "OFFICE"):
        assert arrival_mod.settle(room=spelling, camera="Hunter") == "camera"


def test_nothing_settles_him_with_no_room_and_no_verdict():
    assert arrival_mod.settle(room="", camera="Hunter") == ""


def test_a_verdict_that_EXPLODES_on_being_read_settles_nothing():
    """A broken sensor lane may not cost him the deferred catch-up by
    raising through a bus subscriber; it costs him this observation."""
    class Boom:
        @property
        def zone(self):
            raise RuntimeError("the zone lane exploded")

    assert arrival_mod.settle(room="office", verdict=Boom()) == ""


def test_the_desk_zone_and_room_are_configurable():
    assert arrival_mod.settle(room="study", verdict=FakeVerdict("in the chair"),
                              desk_room="study", desk_zone="in the chair") == "radar"


# ==================================================================
# ONE DELIVERY, NEVER TWO
# ==================================================================
def test_an_unarmed_watch_settles_nothing():
    """He did not come home; he got up for coffee and sat back down. No
    catch-up is owed and none is delivered."""
    watch = arrival_mod.DeskWatch()
    assert watch.armed is False
    assert watch.observe(room="office", verdict=FakeVerdict(DESK)) == ""


def test_the_first_settle_after_the_door_fires_exactly_once():
    watch = arrival_mod.DeskWatch()
    watch.arm()
    assert watch.observe(room="office", verdict=FakeVerdict(DESK)) == "radar"
    assert watch.armed is False
    assert watch.observe(room="office", verdict=FakeVerdict(DESK)) == ""


def test_BOTH_LEGS_FIRING_OVER_AND_OVER_DELIVER_ONCE():
    """The pin. He sits down: the radar commits the desk band, the lens
    recognises him, and both keep on saying so for as long as he sits
    there. He is asked about his mail ONE time."""
    watch = arrival_mod.DeskWatch()
    watch.arm()
    fired = [watch.observe(room="office", verdict=FakeVerdict(DESK),
                           camera=FakeCamera()) for _ in range(50)]
    assert [leg for leg in fired if leg] == ["camera"]


def test_TWO_LEGS_ON_TWO_THREADS_DELIVER_ONCE():
    """They are two different lanes and neither owns the Tk pump, so the
    once-per-arm guarantee cannot rest on the GIL falling the right way."""
    watch = arrival_mod.DeskWatch()
    watch.arm()
    legs, start = [], threading.Barrier(2)

    def leg(**kw):
        start.wait()
        for _ in range(200):
            got = watch.observe(room="office", **kw)
            if got:
                legs.append(got)

    threads = [threading.Thread(target=leg, kwargs={"verdict": FakeVerdict(DESK)}),
               threading.Thread(target=leg, kwargs={"camera": FakeCamera()})]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
        assert not t.is_alive()
    assert len(legs) == 1, "the catch-up was delivered %d times" % len(legs)


def test_arming_twice_is_still_one_delivery():
    """Two sentinels crossing on the same walk through the door both
    reach _greet_return; the damper stops the second greeting, and this
    stops a second catch-up if it ever did not."""
    watch = arrival_mod.DeskWatch()
    watch.arm()
    watch.arm()
    assert watch.observe(room="office", camera="Hunter") == "camera"
    assert watch.observe(room="office", camera="Hunter") == ""


def test_a_SECOND_homecoming_is_owed_a_second_catch_up():
    watch = arrival_mod.DeskWatch()
    watch.arm()
    assert watch.observe(room="office", camera="Hunter") == "camera"
    watch.arm()
    assert watch.observe(room="office", camera="Hunter") == "camera"


def test_clearing_the_watch_gives_up_the_catch_up():
    watch = arrival_mod.DeskWatch()
    watch.arm()
    watch.clear()
    assert watch.armed is False
    assert watch.observe(room="office", camera="Hunter") == ""


def test_a_settle_that_did_not_happen_leaves_it_armed():
    """The important half of the latch: only a real settle spends it. He
    walks through the office to the bedroom, the radar places him in the
    empty space, and the catch-up is still owed."""
    watch = arrival_mod.DeskWatch()
    watch.arm()
    assert watch.observe(room="office", verdict=FakeVerdict(OFF_DESK)) == ""
    assert watch.observe(room="kitchen", camera="Hunter") == ""
    assert watch.armed is True


def test_the_watch_carries_the_room_and_zone_it_was_built_with():
    watch = arrival_mod.DeskWatch(room="study", zone="in the chair")
    watch.arm()
    assert watch.observe(room="study", verdict=FakeVerdict("in the chair",
                                                           room="study")) == "radar"
