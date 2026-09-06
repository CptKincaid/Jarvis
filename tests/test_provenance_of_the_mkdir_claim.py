"""WHERE THE STRONGEST SAFETY PROPERTY IN THIS LANE GETS ITS AUTHORITY.

Every byte this lane puts on HPCOMPUTER goes inside a directory taken with
one SFTP mkdir, and the whole safety argument rests on one sentence: mkdir
at a name that is already taken FAILS.  On POSIX that is measured here.  On
Windows it was, through five rounds, MODELLED -- inferred from
SSH_FXP_MKDIR over CreateDirectory -- because no probe of his machine is
allowed from this side.

2026-09-05 Hunter was asked directly whether creating a folder on
HPCOMPUTER fails when that name is already taken.  He answered YES.

That is a DIFFERENT AND STRONGER claim than a model, and a WEAKER one than a
measurement, and the source has to say which it is.  These tests pin that
distinction so it cannot drift in either direction: nobody may quietly
demote it back to "modelled", and nobody may quietly promote it to
"measured" -- nothing here has measured HPCOMPUTER and nothing here may.

NOTHING HERE OPENS A SOCKET, TOUCHES ~/Desktop, OR SPEAKS TO HPCOMPUTER.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REMOTE = (REPO / "jarvis/tools/remote.py").read_text()


def test_the_windows_mkdir_refusal_is_attributed_to_him():
    """His answer, dated, about his own machine -- named at the place the
    code explains why the claim is safe."""
    assert "CONFIRMED BY HIM" in REMOTE, (
        "the mkdir claim's authority is not stated where it is explained")
    assert "2026-09-05" in REMOTE, "his answer is undated"
    assert "HPCOMPUTER" in REMOTE


def test_that_property_is_no_longer_called_modelled():
    """It stopped being a model the moment he answered.  The old paragraph
    put the mkdir refusal under "HONESTLY MODELLED, NOT MEASURED"; leaving
    it there understates what is known and invites a sixth round to
    re-derive it."""
    para = REMOTE[REMOTE.index("HONESTLY"):REMOTE.index("HONESTLY") + 1400]
    lowered = para.lower()
    assert "modelled" not in lowered.split("confirmed by him")[-1][:600] \
        or "no longer" in lowered, para[:400]
    assert "he was asked" in lowered or "asked him" in lowered, (
        "the source does not say how the Windows behaviour became known")


def test_it_is_not_promoted_to_measured():
    """The line that must NOT appear.  A statement by him about his own box
    is not a local measurement, and this lane has twice been damaged by
    confident numbers with no source.  The code must say so in as many
    words."""
    assert "not a local measurement" in REMOTE.lower(), (
        "his answer must be recorded as HIS STATEMENT, explicitly not as "
        "something this side measured")
    # And the POSIX numbers keep their own, different, provenance.
    assert "MEASURED on this box 2026-09-05" in REMOTE, (
        "the local sftp-server measurements lost their attribution")


def test_the_degradation_argument_survives():
    """The reason this was shippable while it was only modelled: if Windows
    allowed mkdir over an existing directory, the inner name is still ours
    by shape, so it degrades to exactly the old behaviour and no further.
    His YES removes the need for that argument but not the argument."""
    assert "cannot be worse than what it replaces" in REMOTE
