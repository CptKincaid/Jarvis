"""Address thinning at the JOIN (jarvis/address.py) and its four seams.

Every rule in the module docstring has a test here that fails without it, and
so does every corruption that got the first attempt reverted:

  finding 1  content corruption  "Sir Isaac Newton", "Yes Sir, I Can Boogie",
                                 and unquoted third-party interpolations
  finding 2  word mangling       the possessive "sir's"
  finding 3  character loss      the summons on a reminder and an alarm

The composites at the bottom are built by calling the REAL builders --
quiet.digest, QuietPolicy.free, the app's arrival actions, and the speak-queue
watcher -- because the whole claim of this module is that those four joins are
where sirs stack and a finished line is not.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

import jarvis.app as app_mod
from jarvis import address, arrival as arrival_mod
from jarvis import soundbar
from jarvis import speak_queue as sq
from jarvis.address import (SUMMONS, TRAILING, count_sirs, join_fragments,
                            thin_fragments, vocative_spans)
from jarvis.config import CONFIG
from jarvis.presence import WELCOME_LINE
from jarvis.quiet import AWAY_PREFIX, BUSY_PREFIX, FREE_LINE, digest
from jarvis.tools import health
from jarvis.tools import timekeeper as tk
from tests.test_quiet import Clock, FakeCfg, _policy   # noqa: F401


@pytest.fixture(autouse=True)
def default_knobs():
    """Every test starts from the shipped defaults, and no test leaks its
    knob into the next one (the config is MODULE state)."""
    address.configure(enabled=True, per_burst=1)
    yield
    address.configure(enabled=True, per_burst=1)


def sirs(text):
    return count_sirs(text)


# The three held lines behind every measured catch-up: two real timekeeper
# reminders and one watchdog warning.
HELD = [
    (0.0, tk.REMINDER_LINE.format(text="Call the bursar back."), "reminder"),
    (0.0, tk.REMINDER_LINE.format(text="Move the car before four."), "reminder"),
    (0.0, "Memory is getting tight, sir: 12 GB free.", "warning"),
]


# ======================================================================
# Rule B -- only a trailing vocative is droppable
# ======================================================================
class TestOnlyATrailingVocativeIsDroppable:

    @pytest.mark.parametrize("line,expected", [
        # the FRAGMENT-FINAL shapes the corpus uses, with their counts
        ("The build passed, sir.", "The build passed."),            # 517
        ("Shall I hand that over, sir?", "Shall I hand that over?"),  # 45
        ("Very good, sir", "Very good"),                # end of the fragment
        ("Not at all, sir!", "Not at all!"),
        ("Good night, sir;", "Good night;"),
    ])
    def test_each_trailing_shape_is_dropped_and_repaired(self, line, expected):
        # First fragment spends the budget, so the second is the one thinned.
        assert thin_fragments(["Very good, sir.", line])[1] == expected

    @pytest.mark.parametrize("line,expected", [
        # ...and the CLAUSE-final shapes, with their corpus counts. The
        # sentence runs on behind the vocative, so the words behind it are
        # the same speaker's -- this is the register soundbar, health,
        # focus, runwatch, timekeeper, spotify and notes speak in, 31% of
        # the corpus's trailing vocatives and most of the catch-up digest.
        ("Good evening, sir; the day was survived.",                 # 170
         "Good evening; the day was survived."),
        ("While you were busy, sir: one warning.",                   # 29
         "While you were busy: one warning."),
        ("Right away, sir — the build is queued.",                   # 19
         "Right away — the build is queued."),
    ])
    def test_a_clause_final_vocative_is_dropped_too(self, line, expected):
        assert thin_fragments(["Very good, sir.", line])[1] == expected

    @pytest.mark.parametrize("line", [
        # ...but a NEW SENTENCE behind the vocative is left alone: that is
        # the shape of somebody else's sentence embedded in one of Jarvis's,
        # and on a rendered fragment nothing says which it is.
        "Checking right now, sir. One moment.",
        "New mail from Bob: Thank you, sir. Shall I read it?",
        "Not at all, sir! The build is clean.",
        "Shall I hand that over, sir? It is ready.",
        # ...and so is the ASCII hyphen, which the corpus never uses and a
        # music service does ("Yes, Sir - Remastered").
        "Yes, sir - the build passed",                               # 0
        "Yes, sir -- the build passed",
    ])
    def test_a_vocative_with_a_new_sentence_behind_it_is_left_alone(self, line):
        """DEFECT D1. What this buys is every quoted subject, note body and
        track title that ends a sentence inside one of Jarvis's."""
        assert thin_fragments(["Very good, sir.", line])[1] == line

    def test_a_medial_vocative_is_left_alone(self):
        """", sir," mid-sentence is not a sign-off, and dropping it is how a
        track listed as "Yes, Sir, I Can Boogie" loses a word."""
        line = "I have the result, sir, but the model didn't get to the words."
        assert vocative_spans(line) == []
        assert thin_fragments(["Very good, sir.", line])[1] == line

    def test_the_span_starts_at_the_attaching_comma(self):
        spans = vocative_spans("The build passed, sir.")
        assert spans == [(16, 21, TRAILING)]


# ======================================================================
# Rule B -- finding 3: the summons is character, and it always stays
# ======================================================================
class TestTheSummonsIsNeverRemoved:
    """The first attempt classified by POSITION and stripped these, turning a
    butler into a clock radio."""

    @pytest.mark.parametrize("line", [
        tk.REMINDER_LINE.format(text="Call the bursar back."),
        tk.ALARM_LINE.format(time="7:00 am", label=tk.ALARM_DEFAULT_LABEL),
        tk.TIMER_LINE.format(n="ten minute"),
        tk.TIMER_LABEL_LINE.format(n="ten minute", label="pasta"),
    ])
    def test_a_sentence_initial_summons_survives_a_spent_burst(self, line):
        out = thin_fragments(["Welcome back, sir.", "Very good, sir.", line])
        assert out[-1] == line
        assert out[-1].startswith("Sir,")

    def test_a_summons_is_recognised_as_an_address_not_as_prose(self):
        line = tk.ALARM_LINE.format(time="7:00 am", label="Time to get up.")
        assert [k for _, _, k in vocative_spans(line)] == [SUMMONS]

    def test_only_index_0_is_a_summons(self):
        """DEFECT D2. A "Sir," after a full stop is where REMINDER_LINE and
        ALARM_LINE put THIRD-PARTY text, so it is not a summons and it may
        not spend the burst's budget. Both authored summonses are at index
        0, so nothing real is lost."""
        line = "That is done. Sir, this is your reminder. Call the bursar."
        assert vocative_spans(line) == []
        assert thin_fragments(["A moment.", line, "The build passed, sir."]) \
            == ["A moment.", line, "The build passed, sir."]

    def test_a_summons_spends_the_budget_so_a_later_sign_off_goes(self):
        """It has addressed him; the tic behind it is the one that grates."""
        out = thin_fragments(["A moment.",
                              "Sir, this is your reminder. Stand up.",
                              "The build passed, sir."])
        assert out[-1] == "The build passed."

    def test_a_summons_and_a_sign_off_in_ONE_fragment_is_interpolation(self):
        """DEFECT D1, rule C. No authored line carries two (938 of 938), so
        the sign-off here came out of REMINDER_LINE's {text} slot and it is
        the user's own words, not Jarvis's."""
        line = "Sir, this is your reminder. Call the bursar back, sir."
        out = thin_fragments(["A moment.", line])
        assert out[1] == line


# ======================================================================
# Finding 1 -- content corruption
# ======================================================================
class TestThirdPartyWordsAreNotJarvisWords:
    """Every one of these is a real, UNQUOTED interpolation site in this repo,
    and every one of them is what a \\bsir\\b match destroys."""

    @pytest.mark.parametrize("line", [
        # tools/notes.py:550 -- a note body, read back verbatim
        "Your note says Sir Isaac Newton wrote the Principia.",
        "Sir Isaac Newton wrote the Principia",
        # tools/spotify.py:136 -- track and artist
        "Now playing Yes Sir, I Can Boogie by Baccara.",
        "Now playing Yes Sir I Can Boogie by Baccara.",
        "Now playing Yes, Sir, I Can Boogie by Baccara.",
        # tools/calendar.py:988 -- an event title
        "At four you have Sir Humphrey's retirement drinks.",
        # a mail subject
        "New mail from the department: Sir Alex has replied.",
        # anything gemma4 generates
        "The Sirius launch window opens on Tuesday.",
    ])
    def test_third_party_text_is_never_an_address(self, line):
        assert vocative_spans(line) == []
        # ...and it survives a burst whose budget is long spent.
        out = thin_fragments(["Welcome back, sir.", "Very good, sir.", line])
        assert out[-1] == line

    def test_a_fragment_may_carry_both_and_only_jarvis_s_own_goes(self):
        line = "Now playing Yes Sir, I Can Boogie by Baccara, sir."
        out = thin_fragments(["Very good, sir.", line])
        assert out[1] == "Now playing Yes Sir, I Can Boogie by Baccara."

    def test_an_honorific_before_a_name_is_vetoed_explicitly(self):
        assert address._is_before_a_name(" Isaac Newton wrote it") is True
        assert address._is_before_a_name(". The build passed") is False


# ======================================================================
# Finding 2 -- word mangling
# ======================================================================
class TestThePossessiveIsNotAnAddress:

    @pytest.mark.parametrize("line", [
        "That is sir's coffee.",
        "That is, sir's coffee, on the desk.",
        "The note says sir’s coffee is cold.",
    ])
    def test_a_possessive_is_never_matched(self, line):
        assert not [s for s in vocative_spans(line) if s[2] == TRAILING]

    def test_the_possessive_survives_while_the_sign_off_beside_it_goes(self):
        line = "That is sir's coffee, sir."
        out = thin_fragments(["Very good, sir.", line])
        assert out[1] == "That is sir's coffee."

    def test_the_possessive_veto_is_explicit(self):
        assert address._is_possessive("'s coffee") is True
        assert address._is_possessive("’s coffee") is True
        assert address._is_possessive(". The build passed") is False


# ======================================================================
# Rule C -- keep the first, drop later ones
# ======================================================================
class TestKeepTheFirstDropTheRest:

    def test_the_first_survives_and_the_rest_go(self):
        out = thin_fragments(["Welcome back, sir.", "The build passed, sir.",
                              "Memory is tight, sir."])
        assert out == ["Welcome back, sir.", "The build passed.",
                       "Memory is tight."]

    def test_the_first_is_kept_even_when_it_is_not_the_first_fragment(self):
        out = thin_fragments(["The panel is up.", "The build passed, sir.",
                              "Memory is tight, sir."])
        assert out == ["The panel is up.", "The build passed, sir.",
                       "Memory is tight."]

    def test_the_only_address_in_a_burst_is_never_removed(self):
        out = thin_fragments(["The panel is up.", "The build passed.",
                              "Memory is tight, sir."])
        assert out[-1] == "Memory is tight, sir."
        assert sirs(" ".join(out)) == 1

    def test_a_lone_fragment_is_never_rewritten_at_any_setting(self):
        line = "Good night, sir."
        assert thin_fragments([line]) == [line]
        address.configure(enabled=True, per_burst=1)
        assert thin_fragments([line]) == [line]

    def test_the_pass_is_idempotent(self):
        once = thin_fragments(["Welcome back, sir.", "The build passed, sir."])
        assert thin_fragments(once) == once


# ======================================================================
# The knob (persona.address_thinning / address_per_burst)
# ======================================================================
class TestTheKnob:

    def test_disabled_speaks_every_fragment_as_written(self):
        address.configure(enabled=False)
        frags = ["Welcome back, sir.", "The build passed, sir."]
        assert thin_fragments(frags) == frags

    def test_a_larger_allowance_keeps_that_many(self):
        address.configure(enabled=True, per_burst=2)
        out = thin_fragments(["Welcome back, sir.", "The build passed, sir.",
                              "Memory is tight, sir."])
        assert sirs(" ".join(out)) == 2

    def test_the_allowance_is_clamped_at_one_so_it_cannot_be_muted(self):
        address.configure(enabled=True, per_burst=0)
        assert address.per_burst() == 1
        out = thin_fragments(["Welcome back, sir.", "The build passed, sir."])
        assert sirs(" ".join(out)) == 1

    def test_set_config_reads_the_persona_keys(self):
        address.set_config(FakeCfg({"persona": {"address_thinning": False,
                                                "address_per_burst": 3}}))
        assert address.enabled() is False and address.per_burst() == 3
        address.set_config(FakeCfg({}))
        assert address.enabled() is True and address.per_burst() == 1

    def test_a_broken_config_falls_back_to_the_defaults(self):
        address.set_config(SimpleNamespace())          # no .get at all
        assert address.enabled() is True and address.per_burst() == 1


# ======================================================================
# The four real joins
# ======================================================================
def _digest_app(quiet):
    a = object.__new__(app_mod.JarvisApp)
    a.assistant = SimpleNamespace(get=lambda k, d=None: d, user_name="Hunter")
    a._init_assistant_state()
    a.tts = SimpleNamespace(spoken=[])
    a.tts.speak = a.tts.spoken.append
    a.quiet = quiet
    a.services = SimpleNamespace(panel_wake=None)
    return a


class TestTheFourJoins:
    """Before/after on the composites the measurement named, built by the
    real builders rather than by a fixture string."""

    def test_quiet_digest_keeps_the_prefix_s_sir_and_drops_the_lines(self):
        address.configure(enabled=False)
        before = digest(HELD, BUSY_PREFIX)
        address.configure(enabled=True, per_burst=1)
        after = digest(HELD, BUSY_PREFIX)
        assert sirs(before) == 4 and sirs(after) == 2
        assert after.startswith(BUSY_PREFIX + ": two reminders and one warning.")
        # The warning is the clause register the round-3 rule exempted: one
        # sentence, one speaker, so "12 GB free" is health.py's own words.
        assert "Memory is getting tight: 12 GB free." in after
        # One summons announces the list; the second copy of it is a chant.
        assert after.count("Sir, this is your reminder.") == 1
        assert after.count("This is your reminder.") == 1

    def test_quiet_free_keeps_the_acknowledgement_and_thins_the_digest(self):
        def build():
            p = _policy(FakeCfg({"quiet": {"hours": {"start": "23:00",
                                                     "end": "07:00"}}}),
                        Clock(datetime(2026, 8, 31, 23, 30)))
            for _, text, kind in HELD:
                p.hold(text, kind)
            return p.free()

        address.configure(enabled=False)
        before = build()
        address.configure(enabled=True, per_burst=1)
        after = build()
        assert sirs(before) == 5 and sirs(after) == 2
        assert after.startswith(FREE_LINE + " While you were busy: ")

    def test_the_arrival_cue_is_one_burst_across_two_say_calls(self):
        def build():
            p = _policy()
            p._last_reason = "you're out"
            for _, text, kind in HELD:
                p.hold(text, kind)
            a = _digest_app(p)
            arrival_mod.run(["greeting", "catch-up"], a._arrival_actions())
            return a.tts.spoken

        CONFIG.talkback = True
        address.configure(enabled=False)
        before = build()
        address.configure(enabled=True, per_burst=1)
        after = build()
        assert sirs(" ".join(before)) == 5 and sirs(" ".join(after)) == 2
        # The welcome is untouched; the digest behind it lost its opening one.
        assert after[0] == WELCOME_LINE
        assert after[1].startswith("While you were out: ")
        assert before[1].startswith(AWAY_PREFIX + ": ")

    def test_the_speak_queue_watcher_thins_the_lines_it_joins(self, tmp_path):
        lines = ["Welcome back, sir.", "The build passed, sir.",
                 "Memory is getting tight, sir.", "Quiet hours are off, sir."]
        qfile = tmp_path / "speak_queue.txt"
        got = []
        sq.set_sink(got.append)
        try:
            w = sq.Watcher(qfile)
            qfile.write_text("\n".join(lines) + "\n")
            w.poll_once()
        finally:
            sq.set_sink(None)
        assert sirs(" ".join(lines)) == 4
        assert got == ["Welcome back, sir. The build passed. "
                       "Memory is getting tight. Quiet hours are off."]

    def test_a_single_queued_line_is_spoken_verbatim(self, tmp_path):
        qfile = tmp_path / "speak_queue.txt"
        got = []
        sq.set_sink(got.append)
        try:
            w = sq.Watcher(qfile)
            qfile.write_text("Nothing was held back, sir.\n")
            w.poll_once()
        finally:
            sq.set_sink(None)
        assert got == ["Nothing was held back, sir."]


# ======================================================================
# The per-line claim: there is nothing to cut on a finished line
# ======================================================================
def test_no_authored_line_is_changed_when_it_is_spoken_alone():
    """The measurement's central finding, asserted rather than asserted-to:
    every authored spoken line in the repo carries exactly one "sir", so a
    per-line pass has nothing to do and this module does not offer one."""
    import pathlib
    import re
    root = pathlib.Path(app_mod.__file__).parent
    lines = []
    for path in root.rglob("*.py"):
        if path.name == "address.py":
            continue
        for m in re.finditer(r'"([^"\\]{6,200}?\bsir\b[^"\\]{0,200}?)"',
                             path.read_text(), re.I):
            text = m.group(1)
            if text.count("{") == text.count("}"):
                lines.append(text)
    assert len(lines) > 300, "the corpus scan found nothing; the regex broke"
    assert [ln for ln in lines if thin_fragments([ln]) != [ln]] == []


# ======================================================================
# The seam in the app
# ======================================================================
class TestTheAppSeam:

    def test_thin_address_takes_a_list_of_fragments_not_a_finished_line(self):
        a = object.__new__(app_mod.JarvisApp)
        assert a._thin_address(["Welcome back, sir.", "The build passed, sir."]) \
            == ["Welcome back, sir.", "The build passed."]

    def test_a_failure_speaks_the_fragments_as_written(self, monkeypatch):
        a = object.__new__(app_mod.JarvisApp)
        monkeypatch.setattr(address, "thin_fragments",
                            lambda *_, **__: (_ for _ in ()).throw(RuntimeError))
        frags = ["Welcome back, sir.", "The build passed, sir."]
        assert a._thin_address(frags) == frags

    def test_say_no_longer_rewrites_anything_on_the_way_out(self, monkeypatch):
        """The door to the TTS speaks what it is handed. A rendered line is
        where Jarvis's words and a track title are indistinguishable, which
        is exactly why the pass moved off it."""
        a = object.__new__(app_mod.JarvisApp)
        a.assistant = SimpleNamespace(get=lambda k, d=None: d,
                                      user_name="Hunter")
        a._init_assistant_state()
        a.tts = SimpleNamespace(spoken=[])
        a.tts.speak = a.tts.spoken.append
        a.quiet = None
        monkeypatch.setattr(CONFIG, "talkback", True)
        line = "Now playing Yes Sir, I Can Boogie by Baccara, sir."
        a._say("Welcome back, sir.")
        a._say(line)
        assert a.tts.spoken == ["Welcome back, sir.", line]


# ======================================================================
# join_fragments / count_sirs
# ======================================================================
def test_join_fragments_is_thin_then_join():
    assert join_fragments(["Welcome back, sir.", "The build passed, sir."]) \
        == "Welcome back, sir. The build passed."


def test_count_sirs_counts_what_a_listener_hears():
    assert count_sirs("Sir Isaac Newton wrote it, sir.") == 2
    assert count_sirs("The sirloin is ready.") == 0
    assert count_sirs("") == 0


# ======================================================================
# D1 -- third-party text at the END of a fragment
# ======================================================================
class TestThirdPartyTextAtTheEndOfAFragment:
    """A rendered fragment is an authored TEMPLATE with its slots filled in,
    and four of those templates put somebody else's words last. Each of these
    lost a word to the previous pass."""

    @pytest.mark.parametrize("line", [
        # quoted speech in anything read back
        'He said, "Thank you, sir." and left.',
        # tools/notes.py search_text / list_text -- the body is interpolated LAST
        "You have one note about milk, sir: buy milk, sir.",
        "Two notes mention milk, sir: buy milk, sir; and call the dairy.",
        # tools/spotify.py LIKE_LINE -- the track title
        "Saved Yes, Sir! to your Liked Songs, sir.",
        # a mail subject
        "New mail from Bob, sir: Thank you, sir.",
        # jarvis/mailwatch.py:139 -- sender, then the subject words
        "Mail from Bob, sir — re: Thank you, sir.",
    ])
    def test_the_word_at_the_end_is_not_jarvis_s_to_take(self, line):
        out = thin_fragments(["Welcome back, sir.", line])
        assert out[1] == line

    @pytest.mark.parametrize("line", [
        # The surface the clause rule opens: a template with NO "sir" of its
        # own, whose slot hands back a title that ends in one. Rule C cannot
        # help -- there is only one address in the fragment and it is not
        # Jarvis's -- so the punctuation has to.
        "Now playing Yes, Sir! by Baccara.",             # a new SENTENCE
        "Now playing Yes, Sir - Remastered by Baccara.",  # the ASCII hyphen
        "Now playing Yes Sir, I Can Boogie by Baccara.",  # a medial comma
        "Reading the note: Sir Isaac Newton biography.",  # the honorific
    ])
    def test_a_sir_free_template_cannot_lose_its_title(self, line):
        assert thin_fragments(["Welcome back, sir.", line])[1] == line

    def test_the_rule_is_the_three_shapes_and_nothing_cleverer(self):
        """The vocative ends the fragment, or it ends a clause the same
        sentence runs on from -- and it is not inside a quotation. Failing
        all three, the fragment holds two addresses and one came out of a
        slot, and rule C leaves it alone."""
        assert address.is_fragment_final("The build passed, sir.", 21) is True
        assert address.is_fragment_final('He said "no, sir." and left.', 17) \
            is False
        clause = "Memory is tight, sir: 3 free."
        sentence = "Checking now, sir. One moment."
        assert address.is_clause_final(clause, vocative_spans(clause)[0][1]) \
            is True
        assert address.is_clause_final(sentence,
                                       vocative_spans(sentence)[0][1]) is False
        assert len(vocative_spans("New mail from Bob, sir: Thank you, sir.")) == 2

    @pytest.mark.parametrize("line", [
        # reported speech: the vocative belongs to whoever is being quoted,
        # and the clause rule would otherwise reach straight into it.
        'He wrote "yes, sir; the parcel arrived" in the ticket.',
        'The subject reads “thank you, sir: much obliged”.',
        'He said, "Thank you, sir." and left.',
    ])
    def test_a_vocative_inside_a_quotation_is_never_taken(self, line):
        assert thin_fragments(["Very good, sir.", line])[1] == line
        assert thin_fragments(["Very good, sir.",
                               address.authored(line)])[1] == line

    def test_the_apostrophe_is_not_a_quote(self):
        """The shield counts quotation marks, never the apostrophe: "I'm
        coming out of..." opens every second line soundbar.py speaks."""
        line = "I'm coming out of the monitor, sir; the soundbar has dropped."
        assert address.inside_a_quotation(line, 30) is False
        assert thin_fragments(["Very good, sir.", line])[1] == \
            "I'm coming out of the monitor; the soundbar has dropped."

    def test_a_join_site_may_certify_its_own_words(self):
        """A certificate buys the one shape the rules refuse on their own --
        a NEW SENTENCE behind the vocative. The clause separators need none,
        which is why quiet.py's catch-up prefix is now thinned either way.
        Nothing that carries an interpolation may be wrapped."""
        line = "That's the lot, sir. Say the word and I'll read them again."
        assert thin_fragments(["Very good, sir.", address.authored(line)])[1] \
            == "That's the lot. Say the word and I'll read them again."
        assert thin_fragments(["Very good, sir.", line])[1] == line


# ======================================================================
# D2 -- the burst invariant, stated and proved
# ======================================================================
class TestTheInvariant:

    def test_a_title_at_a_sentence_start_cannot_cost_him_his_sign_off(self):
        """"Sir, With Love" arrives through REMINDER_LINE's {text} and
        ALARM_LINE's {label}. It used to be read as a summons, spend the
        budget, and take Jarvis's real sign-off with it."""
        frags = ["Sir, With Love starts at eight.", "The build passed, sir."]
        assert thin_fragments(frags) == frags
        assert vocative_spans(frags[0]) == []

    def test_the_first_fragment_of_a_burst_is_never_rewritten(self):
        for first in ["Welcome back, sir.", "Sir, this is your reminder. Up.",
                      "Sir, With Love starts at eight.", ", sir."]:
            out = thin_fragments([first, "The build passed, sir.",
                                  "Memory is tight, sir."])
            assert out[0] == first

    def test_a_burst_that_went_in_with_an_address_comes_out_with_one(self):
        out = thin_fragments(["The panel is up.", "The build passed, sir."])
        assert sirs(" ".join(out)) == 1

    def test_the_guard_holds_even_if_the_rules_are_wrong(self, monkeypatch):
        """Belt and braces: whatever the classifier decides, a burst never
        comes out mute. Here the removal is sabotaged into eating both."""
        monkeypatch.setattr(address, "_thin",
                            lambda frags, budget: ["The build passed."] * 2)
        frags = ["Welcome back, sir.", "The build passed, sir."]
        assert thin_fragments(frags) == frags


def _words(text):
    import re as _re
    return _re.findall(r"[a-z0-9']+", text.lower())


def test_the_invariant_holds_across_the_fuzz():
    """~35k bursts built out of the shapes this module has ever met. Two
    mechanical claims: a fragment only ever LOSES one "sir" and never a
    word of anything else, and a burst that had an address keeps one."""
    import itertools
    pool = [
        "Welcome back, sir.", "The build passed, sir.", "Very good, sir",
        "Good evening, sir; the day was survived.",
        "While you were busy, sir: one warning.",
        "Sir, this is your reminder. Call the bursar back.",
        "Sir, it's 7:00 am. Time to get up.",
        "Sir, With Love starts at eight.",
        "Now playing Yes Sir, I Can Boogie by Baccara, sir.",
        "That is sir's coffee, sir.",
        'He said, "Thank you, sir." and left.',
        'He wrote "yes, sir; the parcel arrived" in the ticket.',
        "You have one note about milk, sir: buy milk, sir.",
        "I'm coming out of the monitor, sir; the soundbar has dropped.",
        "Mail from Bob, sir — re: Thank you, sir.",
        "Now playing Yes, Sir! by Baccara.",
        "Sir Isaac Newton wrote the Principia.",
        ", sir.", "", "The panel is up.", "Sir?",
    ]
    n = 0
    for burst in itertools.product(pool, repeat=3):
        frags = list(burst)
        out = thin_fragments(frags)
        n += 1
        assert len(out) == len(frags)
        assert out[0] == frags[0]
        for src, got in zip(frags, out):
            wi, wo = _words(src), _words(got)
            if wi == wo:
                continue
            assert len(wo) == len(wi) - 1, (src, got)
            assert any(wi[:i] + wi[i + 1:] == wo
                       for i, w in enumerate(wi) if w == "sir"), (src, got)
            assert len(vocative_spans(src)) == 1, (src, got)
        if any(sirs(f) for f in frags):
            assert any(sirs(f) for f in out), (frags, out)
    assert n > 4000


# ======================================================================
# D4 -- the chant: five held reminders in one catch-up
# ======================================================================
class TestTheChant:

    FIVE = [(0.0, tk.REMINDER_LINE.format(text=t), "reminder") for t in
            ("Stand up.", "Call the bank.", "Move the car.",
             "Feed the cat.", "Take the tablets.")]

    def test_the_first_summons_announces_the_list_and_the_rest_are_thinned(self):
        address.configure(enabled=False)
        before = digest(self.FIVE, BUSY_PREFIX)
        address.configure(enabled=True, per_burst=1)
        after = digest(self.FIVE, BUSY_PREFIX)
        assert sirs(before) == 6            # THE tic he complained about
        assert sirs(after) == 2
        assert after.count("Sir, this is your reminder.") == 1
        assert after.count("This is your reminder.") == 4
        # the announcement is the FIRST one, not a later one
        assert after.index("Sir, this is your reminder.") < \
            after.index("This is your reminder.")

    def test_a_summons_that_is_not_a_repeat_keeps_its_summons(self):
        """An alarm in the middle of a digest is still an alarm."""
        out = thin_fragments([tk.REMINDER_LINE.format(text="Stand up."),
                              tk.ALARM_LINE.format(time="7:00 am",
                                                   label="Time to get up."),
                              tk.REMINDER_LINE.format(text="Call the bank.")])
        assert out[1].startswith("Sir, it's 7:00 am.")
        assert out[2].startswith("This is your reminder.")

    def test_a_thinned_summons_is_a_whole_sentence_again(self):
        out = thin_fragments(["Sir, this is your reminder. Stand up.",
                              "Sir, this is your reminder. Call the bank."])
        assert out[1] == "This is your reminder. Call the bank."


# ======================================================================
# The burst the clause rule was reopened for
# ======================================================================
class TestTheWarningDigest:
    """The shape that occurred twice in the live log 30 s apart (14:34:31,
    14:35:01) and that the round-3 fragment-final rule cut NOTHING out of:
    the soundbar dropping and coming back, and a memory warning. All three
    are the "X, sir; Y." register, and all three are constants of the module
    that speaks them -- ``jarvis/soundbar.py`` and ``jarvis/tools/health.py``
    -- rendered here by the real templates rather than by a fixture string."""

    HELD = [
        (0.0, soundbar.DROP_LINE.format(now="the monitor",
                                        preferred="the soundbar"), "warning"),
        (0.0, soundbar.BACK_LINE.format(Preferred="The soundbar",
                                        now="the monitor"), "warning"),
        (0.0, health.WARN_LINE.format(free=3, hogs=""), "warning"),
    ]

    def test_the_digest_says_it_once_instead_of_four_times(self):
        address.configure(enabled=False)
        before = digest(self.HELD, BUSY_PREFIX)
        address.configure(enabled=True, per_burst=1)
        after = digest(self.HELD, BUSY_PREFIX)
        assert sirs(before) == 4 and sirs(after) == 1
        # ...and not one word besides the three sign-offs has moved.
        assert after == (
            "While you were busy, sir: three warnings. "
            "I'm coming out of the monitor; the soundbar has dropped. "
            "The soundbar is back; I'm still coming out of the monitor. "
            "Memory is getting tight: 3 gigabytes free.")


# ======================================================================
# D5 -- a fragment that thins away to punctuation
# ======================================================================
class TestAFragmentThatThinsToNothing:

    def test_a_bare_vocative_line_is_dropped_from_the_burst(self):
        assert join_fragments(["Feed the cat, sir.", ", sir."]) \
            == "Feed the cat, sir."

    def test_the_speak_queue_never_hands_the_sink_a_lone_full_stop(self, tmp_path):
        """speak_queue.txt is a documented user-writable surface, so its
        fragments are arbitrary lines, not authored ones."""
        qfile = tmp_path / "speak_queue.txt"
        got = []
        sq.set_sink(got.append)
        try:
            w = sq.Watcher(qfile)
            qfile.write_text("Welcome back, sir.\n, sir.\nFeed the cat, sir.\n")
            w.poll_once()
        finally:
            sq.set_sink(None)
        assert got == ["Welcome back, sir. Feed the cat."]
        assert ". ." not in got[0]

    def test_thin_fragments_itself_stays_one_to_one(self):
        """The arrival cue indexes into the result, so the DROP happens in
        the join, not in the pass."""
        out = thin_fragments(["Feed the cat, sir.", ", sir."])
        assert len(out) == 2 and out[1] == "."


# ======================================================================
# D6 -- a burst is thinned once, as fragments
# ======================================================================
class TestTheDigestIsThinnedOnceAsFragments:

    def _held(self):
        p = _policy()
        p._last_reason = "you're out"
        for _, text, kind in HELD:
            p.hold(text, kind)
        return p

    def test_the_policy_hands_out_fragments(self):
        frags = self._held().release_fragments()
        assert len(frags) == 1 + len(HELD)
        assert all(sirs(f) <= 1 for f in frags)

    def test_release_is_the_joined_form_of_release_fragments(self):
        a, b = self._held(), self._held()
        assert a.release() == join_fragments(b.release_fragments())

    def test_the_arrival_cue_thins_fragments_and_never_a_finished_string(self,
                                                                        monkeypatch):
        seen = []
        real = address.thin_fragments

        def spy(fragments, **kw):
            seen.append(list(fragments))
            return real(fragments, **kw)
        monkeypatch.setattr(address, "thin_fragments", spy)
        CONFIG.talkback = True
        a = _digest_app(self._held())
        arrival_mod.run(["greeting", "catch-up"], a._arrival_actions())
        assert len(seen) == 1, "one burst, one pass"
        assert len(seen[0]) == 1 + 1 + len(HELD)     # welcome + head + lines
        assert all(sirs(f) <= 1 for f in seen[0]), \
            "a fragment with two sirs is a finished string, not a fragment"


# ======================================================================
# D7 -- the pass is total
# ======================================================================
class TestThePassNeverRaises:

    @pytest.mark.parametrize("frags", [
        ["ok, sir.", 5],
        ["ok, sir.", None, object()],
        [b"bytes, sir.", "ok, sir."],
        "not a list at all",
    ])
    def test_a_hostile_burst_degrades_to_the_input(self, frags):
        out = thin_fragments(frags)
        assert list(out) == list(frags)
        join_fragments(frags)                  # and the string form is fine

    def test_a_broken_rule_degrades_instead_of_killing_the_catch_up(self,
                                                                   monkeypatch):
        monkeypatch.setattr(address, "_thin",
                            lambda *_, **__: (_ for _ in ()).throw(RuntimeError))
        frags = ["Welcome back, sir.", "The build passed, sir."]
        assert thin_fragments(frags) == frags
        assert join_fragments(frags) == "Welcome back, sir. The build passed, sir."

    def test_the_three_bare_join_sites_survive_it(self, monkeypatch, tmp_path):
        """quiet.py:468, quiet.py:593 and speak_queue.py:153 call
        join_fragments with no guard of their own."""
        monkeypatch.setattr(address, "_thin",
                            lambda *_, **__: (_ for _ in ()).throw(RuntimeError))
        assert digest(HELD, BUSY_PREFIX).startswith(BUSY_PREFIX)
        p = _policy(FakeCfg({"quiet": {"hours": {"start": "23:00",
                                                 "end": "07:00"}}}),
                    Clock(datetime(2026, 8, 31, 23, 30)))
        for _, text, kind in HELD:
            p.hold(text, kind)
        assert p.free().startswith(FREE_LINE)
        got = []
        sq.set_sink(got.append)
        try:
            qfile = tmp_path / "speak_queue.txt"
            w = sq.Watcher(qfile)
            qfile.write_text("Welcome back, sir.\nThe build passed, sir.\n")
            w.poll_once()
        finally:
            sq.set_sink(None)
        assert got == ["Welcome back, sir. The build passed, sir."]


# ======================================================================
# D8 -- what ships
# ======================================================================
def test_the_shipped_default_is_on():
    from jarvis.assistant_config import DEFAULTS
    assert DEFAULTS["persona"]["address_thinning"] is True
    assert DEFAULTS["persona"]["address_per_burst"] == 1
    address.set_config(None)
    assert address.enabled() is True and address.per_burst() == 1


# ======================================================================
# D3 -- the first-wake morning briefing is a join too
# ======================================================================
def _briefing_app(tmp_path, week="", garden=""):
    a = object.__new__(app_mod.JarvisApp)
    a.assistant = SimpleNamespace(
        get=lambda k, d=None: {"briefing.on_first_wake": True,
                               "briefing.after": "06:00"}.get(k, d),
        user_name="Hunter")
    a._init_assistant_state()
    a.brain = SimpleNamespace(is_busy=False)
    a.said = []
    a._say = a.said.append
    a.chats = []
    a.services = SimpleNamespace(
        brain=SimpleNamespace(chat=lambda t, **kw: a.chats.append((t, kw))))
    a._briefing_state_path = lambda: tmp_path / "briefing.json"
    a._pending_week_line = lambda: week
    a._pending_garden_line = lambda: garden
    return a


class TestTheMorningBriefingBurst:
    """Once the longest burst in the live log: 14:33:49-14:34:29, seven TTS
    segments over 40 seconds, four sirs. Two of those segments have gone --
    the day review left the path on 2026-09-02 and the whole thing now
    waits on a yes -- but what remains is still consecutive _say calls with
    nothing between them, so it is still ONE burst and still thinned."""

    WEEK = "Last week: 41 turns, median wait 1.3 seconds, sir."

    def test_the_hand_over_is_thinned_against_the_weekly_line(self, tmp_path):
        a = _briefing_app(tmp_path, week=self.WEEK)
        a._deliver_first_wake_briefing()
        assert a.said == [self.WEEK, "Your briefing for today."]
        assert sirs(" ".join(a.said)) == 1
        assert len(a.chats) == 1

    def test_with_nothing_owed_the_hand_over_keeps_its_own(self, tmp_path):
        a = _briefing_app(tmp_path)
        a._deliver_first_wake_briefing()
        assert a.said == ["Your briefing for today, sir."]

    def test_the_garden_line_is_in_the_same_burst(self, tmp_path):
        a = _briefing_app(tmp_path, week=self.WEEK,
                          garden="I filed three things from this week, sir.")
        a._deliver_first_wake_briefing()
        assert a.said == [self.WEEK, "I filed three things from this week.",
                          "Your briefing for today."]
        assert sirs(" ".join(a.said)) == 1
