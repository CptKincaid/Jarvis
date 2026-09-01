"""Spoken undo ("scratch that") and named lists (specs 5 and 14).

Firewall: a real Commander and a real NotesStore under tmp_path, the
timekeeper and the desktop mocked, no Tk, no network, no /tmp/vss_voice.
The intent log is redirected by tests/conftest.py and again here.
"""
import types
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from jarvis.commander import Commander, IntentClassifier, UNDO_WINDOW_S, undo_kind
from jarvis.config import CONFIG
from jarvis.router import Router
from jarvis.tools.notes import NotesStore, canon_list, list_kind, list_name


class Cfg:
    def __init__(self, **over):
        self.data = {"claude.big_model": "fable", "briefing.enabled": False}
        self.data.update(over)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def setup_line(self, section):
        return f"I'll need {section} set up, sir."


@pytest.fixture
def rich(tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent_log.json")
    for key, val in (("voice_cmds", True), ("jarvis_mode", True),
                     ("auto_type", True), ("talkback", False)):
        monkeypatch.setattr(CONFIG, key, val)
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    svc = types.SimpleNamespace(
        desktop=MagicMock(), workflows=MagicMock(), brain=MagicMock(),
        memory=MagicMock(), context=MagicMock(), tts=MagicMock(), assistant=Cfg(),
        timekeeper=MagicMock(), approvals=MagicMock(), claude=MagicMock(),
        notes=NotesStore(tmp_path / "notes.db"))
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.context.interpret_intent.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    svc.approvals.pending.return_value = []
    svc.claude.active_project = "jarvis"
    svc.brain.local_line.return_value = "Right away, sir."
    tk = svc.timekeeper
    tk.ringing = None
    tk.cancel.return_value = 1
    tk.parse_when.return_value = datetime.now() + timedelta(hours=9)
    tk.describe_due.return_value = "at 7:00 am tomorrow"
    tk.add_timer.return_value = types.SimpleNamespace(id="tm-1")
    tk.add_alarm.return_value = types.SimpleNamespace(id="al-1")
    tk.add_reminder.return_value = types.SimpleNamespace(id="rm-1")
    svc.router = Router(svc.assistant, classify=lambda t: ("local", 0.2))
    return Commander(svc), svc


# ================================================================= undo
def test_undo_kind_reads_the_phrase_and_leaves_others_alone():
    for yes in ("scratch that", "undo that", "undo", "no, scratch that",
                "scratch that last one", "take that back", "belay that",
                "actually, scratch that", "undo the last one"):
        assert undo_kind(yes), yes
    for no in ("scratch my head", "undo the last commit", "never mind",
               "scratch buy milk off the list", "cancel the timer", ""):
        assert not undo_kind(no), no


def test_scratch_that_cancels_the_timer_it_just_set(rich):
    c, svc = rich
    res = c.handle("set a timer for five minutes", source="typed")
    assert res.undo is not None
    svc.timekeeper.cancel.assert_not_called()

    res = c.handle("scratch that", source="typed")
    svc.timekeeper.cancel.assert_called_once_with(which="tm-1", kind="timer")
    assert res.reply == "Timer scrapped, sir." and res.speak
    assert res.status == "Undone"


def test_the_undo_targets_the_item_by_id_not_the_last_one(rich):
    """Two timers, then "scratch that": the SECOND one dies, by id."""
    c, svc = rich
    c.handle("set a timer for five minutes", source="typed")
    svc.timekeeper.add_timer.return_value = types.SimpleNamespace(id="tm-2")
    c.handle("set a timer for ten minutes", source="typed")
    c.handle("scratch that", source="typed")
    svc.timekeeper.cancel.assert_called_once_with(which="tm-2", kind="timer")


def test_scratch_that_cancels_an_alarm_and_a_reminder(rich):
    c, svc = rich
    c.handle("set an alarm for seven", source="typed")
    res = c.handle("scratch that", source="typed")
    svc.timekeeper.cancel.assert_called_once_with(which="al-1", kind="alarm")
    assert res.reply == "Alarm cancelled, sir."

    svc.timekeeper.cancel.reset_mock()
    c.handle("remind me to call mum in ten minutes", source="typed")
    res = c.handle("undo that", source="typed")
    svc.timekeeper.cancel.assert_called_once_with(which="rm-1", kind="reminder")
    assert res.reply == "Reminder cancelled, sir."


def test_scratch_that_forgets_the_note_it_just_took(rich):
    c, svc = rich
    c.handle("take a note that the boiler is broken", source="typed")
    assert svc.notes.count("note") == 1
    res = c.handle("scratch that", source="typed")
    assert svc.notes.count("note") == 0
    assert res.reply == "Note struck out, sir."


def test_scratch_that_takes_the_todo_back_off_the_list(rich):
    c, svc = rich
    c.handle("add buy milk to my list", source="typed")
    assert svc.notes.count("todo") == 1
    c.handle("scratch that", source="typed")
    assert svc.notes.count("todo") == 0


def test_with_nothing_to_undo_the_phrase_keeps_its_typing_meaning(rich):
    """The dictation action is the fallback, not the casualty: this is why
    _try_undo returns None rather than an "I have nothing" reply."""
    c, svc = rich
    res = c.handle("scratch that", source="typed")
    svc.desktop.handle_action.assert_called_once_with("delete_last_sentence")
    assert res.status == "Action: delete_last_sentence"


def test_delete_that_is_never_the_undo(rich):
    """"delete that" stays a dictation action even with an undo pending."""
    c, svc = rich
    c.handle("set a timer for five minutes", source="typed")
    c.handle("delete that", source="typed")
    svc.desktop.handle_action.assert_called_once_with("delete_last_sentence")
    svc.timekeeper.cancel.assert_not_called()


def test_an_undo_runs_once(rich):
    c, svc = rich
    c.handle("set a timer for five minutes", source="typed")
    c.handle("scratch that", source="typed")
    svc.timekeeper.cancel.reset_mock()
    c.handle("scratch that", source="typed")
    svc.timekeeper.cancel.assert_not_called()
    svc.desktop.handle_action.assert_called_once_with("delete_last_sentence")


def test_a_stale_undo_is_not_run(rich):
    c, svc = rich
    c.handle("set a timer for five minutes", source="typed")
    closure, stamp = c._last_undo
    c._last_undo = (closure, stamp - UNDO_WINDOW_S - 1)
    c.handle("scratch that", source="typed")
    svc.timekeeper.cancel.assert_not_called()


def test_an_undo_that_throws_says_so_rather_than_lying(rich):
    c, svc = rich
    c.handle("set a timer for five minutes", source="typed")
    svc.timekeeper.cancel.side_effect = RuntimeError("db gone")
    res = c.handle("scratch that", source="typed")
    assert res.reply == "I couldn't take that back, sir."
    assert res.status == "Undo failed"


# ========================================================== named lists
def test_list_name_helpers():
    assert canon_list("the Shopping List") == "shopping"
    assert canon_list("my packing list") == "packing"
    assert canon_list("the list") == ""
    assert list_kind("the shopping list") == "list:shopping"
    assert list_kind("my task list") == ""          # reserved: that's to-dos
    assert list_name("list:shopping") == "shopping"
    assert list_name("todo") is None


def test_add_milk_to_the_shopping_list_makes_a_shopping_list(rich):
    c, svc = rich
    res = c.handle("add milk to the shopping list", source="typed")
    assert res.reply == "Added to your shopping list, sir." and res.speak
    assert svc.notes.list_names() == ["shopping"]
    assert svc.notes.count("list:shopping") == 1
    assert svc.notes.count("todo") == 0             # NOT the generic list


def test_a_reserved_name_still_reaches_the_todo_list(rich):
    c, svc = rich
    c.handle("add buy milk to my task list", source="typed")
    assert svc.notes.count("todo") == 1 and svc.notes.list_names() == []


def test_several_items_in_one_breath(rich):
    c, svc = rich
    res = c.handle("add milk, eggs and bread to the shopping list", source="typed")
    assert res.reply == "Three added to your shopping list, sir."
    assert svc.notes.count("list:shopping") == 3
    # …but an errand with an "and" in it stays one item
    c.handle("add pick up the dry cleaning and post the forms to the errands list",
             source="typed")
    assert svc.notes.count("list:errands") == 1


def test_reading_a_list_that_does_not_exist_says_so(rich):
    c, svc = rich
    res = c.handle("read my packing list", source="typed")
    assert res.reply == "You haven't a packing list, sir."
    assert svc.notes.list_names() == []              # and does not invent it


def test_read_my_list_back(rich):
    c, svc = rich
    for item in ("milk", "eggs", "bread"):
        svc.notes.add("list:shopping", item)
    res = c.handle("what's on the shopping list", source="typed")
    assert res.reply == "Three on your shopping list, sir: milk, eggs, and bread."
    assert res.speak and res.status == "shopping list"


def test_the_phone_path_needs_no_wake_word(rich):
    """`ssh spark jarvis "what's on the shopping list"` -- source "cli",
    no jarvis prefix, so it has to come through unprefixed Tier 1."""
    c, svc = rich
    svc.notes.add("list:shopping", "milk")
    res = c.handle("what's on the shopping list", source="cli")
    assert res.reply == "You have one item on your shopping list, sir: milk."
    svc.brain.chat.assert_not_called()


def test_strike_one_item_off_by_name_and_by_ordinal(rich):
    c, svc = rich
    for item in ("milk", "eggs", "bread"):
        svc.notes.add("list:shopping", item)
    res = c.handle("take milk off the shopping list", source="typed")
    assert res.reply == "Off the shopping list, sir; two left."
    assert [i["text"] for i in svc.notes.list("list:shopping")] == ["eggs", "bread"]
    # the ordinals index the order he was just read
    c.handle("cross the second one off the shopping list", source="typed")
    assert [i["text"] for i in svc.notes.list("list:shopping")] == ["eggs"]


def test_striking_something_that_is_not_there(rich):
    c, svc = rich
    svc.notes.add("list:shopping", "milk")
    res = c.handle("take the caviar off the shopping list", source="typed")
    assert res.reply == "I couldn't find that on your shopping list, sir."
    assert svc.notes.count("list:shopping") == 1


def test_scratch_that_puts_a_struck_item_back(rich):
    c, svc = rich
    svc.notes.add("list:shopping", "milk")
    c.handle("take milk off the shopping list", source="typed")
    res = c.handle("scratch that", source="typed")
    assert res.reply == "Back on the shopping list, sir."
    assert [i["text"] for i in svc.notes.list("list:shopping")] == ["milk"]


def test_scratch_that_takes_a_list_item_back_off(rich):
    c, svc = rich
    c.handle("add milk to the shopping list", source="typed")
    res = c.handle("scratch that", source="typed")
    assert res.reply == "Off the shopping list again, sir."
    assert svc.notes.count("list:shopping") == 0


# ------------------------------------------------- destructive read-back
def test_clearing_a_list_is_read_back_and_a_yes_runs_it(rich):
    c, svc = rich
    for item in ("milk", "eggs", "bread"):
        svc.notes.add("list:shopping", item)
    res = c.handle("clear the shopping list", source="typed")
    assert res.reply == "Clear all three off your shopping list, sir?"
    assert res.status == "Confirm?" and svc.notes.count("list:shopping") == 3

    res = c.handle("yes", source="typed")
    assert res.reply == "The shopping list is clear, sir."
    assert res.status == "Cleared 3 from the shopping list"
    assert svc.notes.count("list:shopping") == 0


def test_a_new_subject_drops_the_clear_offer(rich):
    c, svc = rich
    for item in ("milk", "eggs"):
        svc.notes.add("list:shopping", item)
    c.handle("clear the shopping list", source="typed")
    c.handle("what's on the shopping list", source="typed")
    c.handle("yes", source="typed")
    assert svc.notes.count("list:shopping") == 2


def test_scratch_that_restores_a_whole_wiped_list(rich):
    c, svc = rich
    for item in ("milk", "eggs", "bread"):
        svc.notes.add("list:shopping", item)
    c.handle("clear the shopping list", source="typed")
    c.handle("yes", source="typed")
    res = c.handle("scratch that", source="typed")
    assert res.reply == "All three back on the shopping list, sir."
    assert [i["text"] for i in svc.notes.list("list:shopping")] == \
        ["milk", "eggs", "bread"]


def test_read_back_off_clears_outright(rich):
    c, svc = rich
    svc.assistant.data["confirm.read_back"] = False
    for item in ("milk", "eggs"):
        svc.notes.add("list:shopping", item)
    res = c.handle("clear the shopping list", source="typed")
    assert res.reply == "The shopping list is clear, sir."
    assert svc.notes.count("list:shopping") == 0


def test_clearing_an_empty_list_asks_nothing(rich):
    c, svc = rich
    svc.notes.add("list:shopping", "milk")
    c.handle("take milk off the shopping list", source="typed")
    res = c.handle("clear the shopping list", source="typed")
    assert res.reply == "Your shopping list is empty already, sir."


def test_what_lists_do_i_have(rich):
    c, svc = rich
    res = c.handle("what lists do i have", source="typed")
    assert res.reply == "You haven't any lists yet, sir."
    svc.notes.add("list:shopping", "milk")
    svc.notes.add("list:packing", "socks")
    res = c.handle("what lists do i have", source="typed")
    assert res.reply == "Two lists, sir: shopping and packing."


def test_the_built_in_notes_and_todos_are_untouched(rich):
    """The named lists share the store; they must not leak into it."""
    c, svc = rich
    svc.notes.add("list:shopping", "milk")
    svc.notes.add("todo", "call the bank")
    svc.notes.add("note", "the boiler is broken")
    assert c.handle("what's on my todo list", source="typed").reply == \
        "You have one to-do, sir: call the bank."
    assert c.handle("show my notes", source="typed").reply == \
        "You have one note, sir: the boiler is broken."
    assert svc.notes.count("list:shopping") == 1


# ------------------------------------------------------- the Tier-2 tool
def test_the_notes_tool_reaches_named_lists(tmp_path):
    """The model's own route in: notes(action=…, list="shopping")."""
    from types import SimpleNamespace

    from jarvis.tools.notes import make_tools

    store = NotesStore(tmp_path / "notes.db")
    spec, = make_tools(None, SimpleNamespace(notes=store))

    res = spec.handler(action="add", text="milk", list="the shopping list")
    assert res.speak == "Added to your shopping list, sir."
    assert store.count("list:shopping") == 1 and store.count("todo") == 0

    res = spec.handler(action="list", list="shopping")
    assert res.speak == "You have one item on your shopping list, sir: milk."

    # reading one that does not exist neither invents it nor lies
    res = spec.handler(action="list", list="packing")
    assert not res.ok and res.speak == "You haven't a packing list, sir."
    assert store.list_names() == ["shopping"]


def test_the_tool_reads_a_named_list_wipe_back(rich, tmp_path):
    from types import SimpleNamespace

    from jarvis.tools.notes import make_tools

    c, svc = rich
    store = svc.notes
    spec, = make_tools(None, SimpleNamespace(notes=store))
    for item in ("milk", "eggs", "bread"):
        store.add("list:shopping", item)

    res = spec.handler(action="remove", which="all", list="shopping")
    assert res.speak == "Clear all three off your shopping list, sir?"
    assert store.count("list:shopping") == 3

    out = c.handle("yes", source="typed")
    assert out.reply == "The shopping list is clear, sir."
    assert out.status == "Cleared 3 from the shopping list"
    assert store.count("list:shopping") == 0


# ------------------------------------------------------- the intent gate
def test_a_spoken_list_command_is_never_called_background_chat(rich, monkeypatch):
    """The wake word is consumed before the text arrives, so these reach
    the classifier bare -- which has called every short phrase NO three
    times before now. The Tier-1 probe must spare them the guess."""
    c, svc = rich

    def boom(*a, **k):
        raise AssertionError("the classifier was consulted")
    monkeypatch.setattr(c.intent, "classify", boom)

    c.handle("add milk to the shopping list", source="voice")
    assert svc.notes.count("list:shopping") == 1
    res = c.handle("what's on the shopping list", source="voice")
    assert res.reply == "You have one item on your shopping list, sir: milk."


def test_scratch_that_by_voice_never_reaches_the_classifier(rich, monkeypatch):
    """Two words: the gate would drop it. The undo rung runs first."""
    c, svc = rich
    c.handle("set a timer for five minutes", source="voice")

    def boom(*a, **k):
        raise AssertionError("the classifier was consulted")
    monkeypatch.setattr(c.intent, "classify", boom)
    res = c.handle("scratch that", source="voice")
    svc.timekeeper.cancel.assert_called_once_with(which="tm-1", kind="timer")
    assert res.reply == "Timer scrapped, sir."


# =========================================== compound turns (review 2026-08-31)
# _try_multi runs two Tier-1 clauses in one turn, and it built its combined
# CommandResult by hand -- copying reply/speak/status/done/ack and dropping
# both clauses' undo closures, while the ONE _pending_destructive slot was
# overwritten by whichever clause read back last. Two separate silent losses.
def test_scratch_that_takes_back_a_whole_compound_turn(rich):
    """Both halves come back, last thing created first."""
    c, svc = rich
    res = c.handle("set a timer for ten minutes and add milk to my todo list",
                   source="typed")
    assert res.handled and res.undo is not None
    assert svc.notes.count("todo") == 1

    res = c.handle("scratch that", source="typed")
    svc.timekeeper.cancel.assert_called_once_with(which="tm-1", kind="timer")
    assert svc.notes.count("todo") == 0
    assert res.status == "Undone"
    # One burst, one sign-off: the two clauses' undo lines are joined and
    # thinned together (jarvis/address.py).
    assert "Timer scrapped." in res.reply


def test_a_compound_does_not_leave_the_previous_turns_undo_armed(rich):
    """The regression: a compound carried no undo, so handle() left the
    PREVIOUS turn's closure armed for the rest of UNDO_WINDOW_S and
    "scratch that" struck out the note from a turn ago while the two things
    just created stayed."""
    c, svc = rich
    c.handle("take a note that the boiler is broken", source="typed")
    assert svc.notes.count("note") == 1
    c.handle("set a timer for ten minutes and add milk to my todo list",
             source="typed")
    c.handle("scratch that", source="typed")
    assert svc.notes.count("note") == 1        # NOT what the undo was about
    assert svc.notes.count("todo") == 0
    svc.timekeeper.cancel.assert_called_once_with(which="tm-1", kind="timer")


def test_a_compound_of_two_read_backs_asks_once_and_one_yes_runs_both(rich):
    """Two clauses, one _pending_destructive slot: the second stash used to
    discard the first, so the single "yes" cancelled the alarms and the
    shopping list was never cleared -- with both questions read aloud."""
    c, svc = rich
    for item in ("milk", "eggs", "bread"):
        svc.notes.add("list:shopping", item)
    svc.timekeeper.list.return_value = [object(), object(), object()]
    svc.timekeeper.cancel.return_value = 3

    res = c.handle("clear the shopping list and cancel all the alarms",
                   source="typed")
    # One question, asked once: the two read-backs are one burst.
    assert res.reply == ("Clear all three off your shopping list, sir? "
                         "Cancel all three alarms?")
    assert svc.notes.count("list:shopping") == 3      # nothing done yet
    svc.timekeeper.cancel.assert_not_called()

    res = c.handle("yes", source="typed")
    assert svc.notes.count("list:shopping") == 0
    svc.timekeeper.cancel.assert_called_once_with("all", "alarm")
    assert res.reply == "The shopping list is clear, sir. Cancelled 3."


def test_a_compound_that_ran_nothing_leaves_a_standing_question_alone(rich):
    """_try_multi clears the slot to read each clause's stash; if no clause
    ran it must put back whatever question was already on the table."""
    c, svc = rich
    stash = (lambda: None, "Cancel the timer, sir?", 0.0)
    c._pending_destructive = stash
    # both clauses are Tier-1 (so the split is taken) but the ladder hands
    # back nothing for either
    said = "clear the shopping list and cancel all the alarms"
    assert c._try_multi(said, lambda part: None) is None
    assert c._pending_destructive is stash


# ================================================ double-speech (2026-08-31)
# Three separate doublings from one test session, three separate causes.
# Kept together because they were reported as one symptom: "it said it twice".

def test_an_item_already_on_the_list_is_not_added_again(rich):
    """#31: "add milk to the shopping list" then "add milk, eggs, and bread
    to the shopping list" read back as "milk, milk, eggs, ...".
    """
    c, svc = rich
    c.handle("add milk to the shopping list", source="typed")
    res = c.handle("add milk, eggs, and bread to the shopping list",
                   source="typed")
    assert [r["text"] for r in svc.notes.list("list:shopping", limit=20)] == \
        ["milk", "eggs", "bread"]
    assert res.reply == ("Two added to your shopping list, sir; "
                         "milk was already there.")


def test_the_oxford_comma_is_a_separator_not_an_item(rich):
    """#31: "it kept the and bread i said" -- the list held "and bread"."""
    c, svc = rich
    c.handle("add milk, eggs, and bread to the shopping list", source="typed")
    assert [r["text"] for r in svc.notes.list("list:shopping", limit=20)] == \
        ["milk", "eggs", "bread"]
    # read back as three things, with the conjunction only where it belongs
    res = c.handle("what's on the shopping list", source="typed")
    assert res.reply.endswith(": milk, eggs, and bread.")
    assert "and and" not in res.reply


def test_a_duplicate_inside_one_breath_is_stored_once(rich):
    c, svc = rich
    res = c.handle("add milk, milk, and eggs to the shopping list",
                   source="typed")
    assert [r["text"] for r in svc.notes.list("list:shopping", limit=20)] == \
        ["milk", "eggs"]
    assert res.reply == ("Two added to your shopping list, sir; "
                         "milk was already there.")


def test_adding_only_things_already_there_says_so(rich):
    c, svc = rich
    svc.notes.add("list:shopping", "milk")
    res = c.handle("add milk to the shopping list", source="typed")
    assert res.reply == "Milk is already on your shopping list, sir."
    assert svc.notes.count("list:shopping") == 1


def test_undo_after_a_deduped_add_takes_back_only_the_new_items(rich):
    """The undo must not strike the milk that was already on the list."""
    c, svc = rich
    svc.notes.add("list:shopping", "milk")
    c.handle("add milk, eggs, and bread to the shopping list", source="typed")
    assert [r["text"] for r in svc.notes.list("list:shopping", limit=20)] == \
        ["milk", "eggs", "bread"]
    c.handle("scratch that", source="typed")
    assert [r["text"] for r in svc.notes.list("list:shopping", limit=20)] == \
        ["milk"]


def test_one_compound_utterance_gets_one_acknowledgement(rich, monkeypatch):
    """#57 "two noted said though": "My mom is Heather and my dad is Ali."
    spoke "Noted, sir: your mom is Heather." and then "Noted: your dad is
    Ali." -- two utterances for one breath."""
    c, svc = rich
    monkeypatch.setattr(CONFIG, "talkback", True)   # or _speak returns early
    svc.memory.add_person.side_effect = lambda alias, name, email=None: {
        "alias": alias, "name": name, "email": email}
    res = c.handle("my mom is Heather and my dad is Ali", source="typed")
    assert svc.memory.add_person.call_count == 2        # both facts kept
    assert res.reply == "Noted, sir: your mom is Heather and your dad is Ali."
    assert res.speak                                    # spoken by the JOIN
    # and NOT spoken by the handlers themselves: two eager _speak calls for
    # one breath is exactly the "two noted said" he heard.
    assert svc.tts.speak.call_args_list == []


def test_merge_acks_leaves_two_different_answers_alone():
    from jarvis.commander import _merge_acks
    assert _merge_acks(["Noted, sir: your mom is Heather.",
                        "Noted, sir: your dad is Ali."]) == \
        "Noted, sir: your mom is Heather and your dad is Ali."
    # different lead-ins are two answers, not one
    assert _merge_acks(["Noted, sir: your mom is Heather.",
                        "Timer set, sir: ten minutes."]) is None
    # no lead-in at all
    assert _merge_acks(["Done, sir.", "Done, sir."]) is None
    assert _merge_acks(["Noted, sir: your mom is Heather."]) is None


def test_whisper_stutter_is_not_asked_twice():
    """#31 verbatim: 'What are on both lists? What are on both lists?' --
    one question, decoded twice by Whisper."""
    from jarvis.transcriber import collapse_repeats
    assert collapse_repeats("What are on both lists? What are on both lists?") \
        == "What are on both lists?"
    assert collapse_repeats("Set a timer. Set a timer. Then call mom.") == \
        "Set a timer. Then call mom."
    # emphasis is left alone: short repeats are speech, not a decode artefact
    assert collapse_repeats("No. No.") == "No. No."
    assert collapse_repeats("Add milk to the shopping list") == \
        "Add milk to the shopping list"


def test_one_reply_never_speaks_the_same_line_twice():
    """#144: the calendar confirmation was SPOKEN TWICE -- the add_event
    tool's own confirmation and the model's reply are the same authored
    sentence, and a tag batch carrying both spoke both."""
    import jarvis.app as app_mod
    line = "Added hello, Tuesday at 4:30 PM, to your calendar, sir."
    app = types.SimpleNamespace(
        _turn_finished=lambda: None, _say=MagicMock(), _last_source="voice",
        _active_turn_id="", _quiet_turn=False, context=MagicMock(),
        _followup_after_speech=False, _thin_address=lambda frags: list(frags))
    app_mod.JarvisApp._on_brain_tags(app, [("SPEAK", line), ("SPEAK", line)])
    assert app._say.call_args_list == [((line,),)]

    # two DIFFERENT lines in one reply are both still spoken
    app._say.reset_mock()
    app_mod.JarvisApp._on_brain_tags(
        app, [("SPEAK", line), ("SPEAK", "Anything else, sir?")])
    assert app._say.call_count == 2

    # and the same line in a LATER reply speaks again
    app._say.reset_mock()
    app_mod.JarvisApp._on_brain_tags(app, [("SPEAK", line)])
    assert app._say.call_count == 1
