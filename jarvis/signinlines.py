"""What Jarvis SAYS when he recognises somebody, and when he refuses to.

PURE, TOTAL, DETERMINISTIC, like ``jarvis/recognise.py`` beside it. No clock,
no config, no model, no logger, no threshold. It takes what the legs said and
returns a string, so every sentence in this feature can be tested with no
microphone and no lens -- which on this box is the only way it can be tested.

THE ONE SENTENCE THAT IS HIS, VERBATIM, AND IT IS THE SIGN-IN LINE FOR
ANYBODY HE HAS ENROLLED
    "Voice and identity recognized, welcome back Hunter. How may I be of
    assistance today?"

    His wording, fixed, with the first name of whoever was recognised in
    it -- Hunter, or Mara. The first version reserved it for the owner and
    greeted a guest with "Hello Mara, I recognize you."; he asked for his
    sentence to be THE line for a recognised person, so it is. Scope is not
    a greeting's business: ``gate.allowed_for`` still holds a KNOWN person
    to the time, the weather and the music a second later.

    SETTLED 2026-09-04 FOR BOTH BRANCHES (voice-multispeaker and
    people-signin): THIS MODULE OWNS THE LINE AND THE RULE. The rule: his
    exact words are spoken TO THE PERSON RECOGNISED, addressed by FIRST
    NAME -- ``first_name(person)`` -- and never by an honorific; "sir" and
    "ma'am" live elsewhere (jarvis/honorific.py). Both branches import
    ``BOTH_LEGS_LINE`` and ``first_name`` from here and neither carries its
    own copy; tests/test_signin_lines.py fails on a second copy anywhere
    under jarvis/, on purpose, so the integration merge cannot disagree.

    It claims BOTH legs, so it may only be said when both legs actually
    confirmed. Four preconditions, all required, and ``both_legs_line``
    returns "" if any one of them is missing:

      1. the voice leg NAMED a label -- past the accept bar AND the margin,
         which is voicegallery.identify's job and is already decided by the
         time a name arrives here;
      2. the face leg NAMED a label -- past camera.identity_min, already
         applied by eye.Attention;
      3. it is the SAME label;
      4. both legs were RUNNING. A dark camera under the night curfew is not
         a confirmation and neither is an abstention. Saying a sentence that
         claims two instruments when one of them never ran is the kind of
         small lie that makes the whole feature untrustworthy.

    Miss any one and a DIFFERENT and honest sentence is said, naming which
    leg actually saw them -- for him and for a guest alike.

SILENCE IS ALSO A SENTENCE
    Under the abstain window, nothing about identity is said at all -- the
    words are simply answered. An abstention is a FAIL-OPEN, not a
    recognition; narrating it would be claiming an instrument that did not
    run. With nobody enrolled there is no identity line either, exactly as a
    fresh box behaves today.

AND ONE THING THAT MAY NEVER BE SAID
    Nothing here may be worded as security. ``jarvis/identity.py`` opens by
    saying a recording defeats the voice check and a photograph defeats the
    face check, and that the user was told so. These are greetings.

THE ONE HONEST TENSION, SAID OUT LOUD RATHER THAN HIDDEN
    ``NEAR_MISS_LINE`` tells the room that the speaker sounds like somebody
    enrolled without saying who. That is a small piece of information a
    stranger would not otherwise have, and it is the deliberate price of not
    flipping a coin between two people. It is the ONLY line here that says
    anything at all about an unnamed speaker, it never names anybody, and it
    is not said when the accept bar failed -- an unrelated voice gets the
    ordinary unknown line and learns nothing.
"""
from __future__ import annotations

# HIS WORDING, FIXED, for ANY recognised person. ``{first}`` is Person.name's
# first token.
BOTH_LEGS_LINE = ("Voice and identity recognized, welcome back {first}. "
                  "How may I be of assistance today?")

# One leg, and it NAMES WHICH -- "welcome back anyway" rather than pretending
# the camera agreed.
VOICE_ONLY_LINE = ("I recognize your voice, {first}. The camera isn't "
                   "confirming right now — welcome back anyway.")
FACE_ONLY_LINE = ("I recognize your face, {first}. I haven't heard you yet — "
                  "welcome back.")

# The sentence that replaces a coin flip, and the whole reason the margin
# exists. It NAMES NOBODY.
NEAR_MISS_LINE = ("I can hear someone I know, but I can't tell which of you — "
                  "say a little more and I'll catch up.")

# A label with too few takes for its centroid to have settled. Logged always,
# SPOKEN ONLY IN SHADOW MODE, and never used to grant scope.
PROVISIONAL_LINE = ("I think that's {first}, but I've only heard them a few "
                    "times — I'll keep listening.")

MODE_SHADOW = "shadow"


def first_name(person, fallback: str = "") -> str:
    """What ``{first}`` renders as -- THE ONE RESOLVER OF THE FIRST NAME.

    A TYPED first name wins (people-signin's Person row carries ``first``,
    typed at enrolment; a row without the field, as on this branch, simply
    has none), else the first token of the display name, else the label
    capitalised, which is what ``Person.display`` already does -- a
    greeting with an empty name in it is worse than a greeting with a
    plain one -- else ``fallback``.
    """
    if person is None:
        return str(fallback or "")
    try:
        typed = str(getattr(person, "first", "") or "").strip()
        if typed:
            return typed
        name = str(getattr(person, "name", "") or "")
        if name.strip():
            return name.split()[0]
        return str(getattr(person, "display", lambda: "")() or fallback or "")
    except Exception:  # noqa: BLE001 - a name that cannot be read is no name
        return str(fallback or "")


def both_legs_line(legs, person) -> str:
    """His verbatim sentence, or "" -- and "" whenever any precondition is
    missing. Never a partial claim.

    Deliberately written as one boolean chain rather than several returns:
    every one of these is a way the sentence would be a lie, and keeping them
    in one place is what makes the test that enumerates them possible.
    """
    if person is None:
        return ""
    # ANY registry person: his wording is the sign-in line for whoever both
    # legs recognised. (An earlier version kept it for the owner and said
    # "Hello Mara, I recognize you." to a guest; his call was otherwise.)
    voice = str(getattr(legs, "voice_says", "") or "")
    face = str(getattr(legs, "face_says", "") or "")
    if not (voice and face):
        return ""
    if voice != face:
        return ""
    if voice != str(getattr(person, "label", "") or ""):
        return ""
    if not (getattr(legs, "voice_running", False)
            and getattr(legs, "face_running", False)):
        return ""
    return BOTH_LEGS_LINE.format(first=first_name(person))


def line_for(legs, person, *, near_miss: bool = False, provisional=None,
             mode: str = "") -> str:
    """The sentence to say, or "" for "say nothing about identity".

    ``person`` is the registry row for whoever was named, or None. ``near_miss``
    is a failed MARGIN -- two enrolled people too close to separate.
    ``provisional`` is a Person whose label won on score but has too few takes.

    ORDER IS THE SAFETY ARGUMENT. A confirmed identity outranks a near miss,
    a near miss outranks a provisional guess, and a provisional guess is only
    ever spoken in shadow mode. Nothing below can promote anybody: the only
    line that names a person is one built from a Person the caller already
    resolved through the registry, and recognise() has already dropped a label
    the registry does not hold.
    """
    if person is not None:
        both = both_legs_line(legs, person)
        if both:
            return both
        voice = str(getattr(legs, "voice_says", "") or "")
        face = str(getattr(legs, "face_says", "") or "")
        label = str(getattr(person, "label", "") or "")
        first = first_name(person)
        # One leg, owner or guest alike: the line names WHICH instrument
        # actually saw them rather than claiming both.
        if voice == label and getattr(legs, "voice_running", False):
            return VOICE_ONLY_LINE.format(first=first)
        if face == label and getattr(legs, "face_running", False):
            return FACE_ONLY_LINE.format(first=first)
        return ""
    if near_miss:
        return NEAR_MISS_LINE
    if provisional is not None and str(mode) == MODE_SHADOW:
        return PROVISIONAL_LINE.format(first=first_name(provisional))
    return ""
