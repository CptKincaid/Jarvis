"""jarvis/filephrase.py -- the spoken shapes a file name hides inside.

The module is 400 lines of the most dangerous code in the mail lane (it is
what turns "that file on my desktop" into ONE path) and it had no suite of
its own: everything it does was covered only incidentally, through
tests/test_send_file.py, which exercises it via outbox.prepare and can
therefore only see the cases the mail lane happens to reach.

So this file tests the resolver DIRECTLY, and it is organised by the way it
can be wrong rather than by function:

* the PARSER -- what a sentence actually specifies (folder, type, recency,
  the leftover name, a literal file name protected from the cue tables);
* the SHAPES -- folder cue, type cue, recency, literal, partial, explicit
  path, and the combinations of them;
* what it must REFUSE -- the deny list, dotfiles, system trees, symlinks
  out of the roots, folders, specials, oversize;
* the TIE BAND, which is the whole safety margin between "here is your
  file" and "which one, sir?";
* the JUST window, because "I just downloaded it" is a claim about time
  that has to be checked rather than believed.

No network, no transport, no config: filephrase takes its roots as an
argument and reads nothing else.
"""
import os
import time
from pathlib import Path

import pytest

from jarvis import filephrase
from jarvis.tools import filepick


# ------------------------------------------------------------- fixtures
def _write(path: Path, data: bytes, age_s: float = 0.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    when = time.time() - age_s
    os.utime(path, (when, when))
    return path


@pytest.fixture
def home(tmp_path):
    """Desktop / Documents / Downloads with known names and known mtimes.

    The ages are spread by hours so that "the newest" is unambiguous
    wherever a test relies on it, and TIE_WINDOW_S is only under test when
    a test says it is.
    """
    _write(tmp_path / "Desktop" / "lab_report.pdf", b"L" * 3000, 1 * 3600)
    _write(tmp_path / "Desktop" / "lab_report_final.pdf", b"F" * 4000, 2 * 3600)
    _write(tmp_path / "Desktop" / "notes.txt", b"N" * 800, 3 * 3600)
    _write(tmp_path / "Desktop" / "holiday.jpg", b"J" * 900, 4 * 3600)
    _write(tmp_path / "Downloads" / "invoice-4471.pdf", b"I" * 2000, 0.25 * 3600)
    _write(tmp_path / "Downloads" / "syllabus.docx", b"D" * 1500, 5 * 3600)
    _write(tmp_path / "Documents" / "Biosensors_Lab-Handout_v2.pdf",
           b"B" * 5000, 6 * 3600)
    # One level down, which DEFAULT_DEPTH=1 reaches and two levels does not.
    _write(tmp_path / "Documents" / "Fall2026" / "thermo_notes.pdf",
           b"T" * 1200, 7 * 3600)
    _write(tmp_path / "Documents" / "Fall2026" / "deep" / "buried.pdf",
           b"X" * 100, 8 * 3600)
    return tmp_path


@pytest.fixture
def roots(home):
    return [str(home / "Desktop"), str(home / "Documents"),
            str(home / "Downloads")]


# ==================================================================
# 1. The parser: what a sentence SPECIFIES
# ==================================================================
def test_a_folder_cue_leaves_no_name_behind():
    """"on my desktop" must be consumed WHOLE. A leftover "desktop" would
    be scored as part of the file name and match desktop-backup.zip."""
    ph = filephrase.parse("that file on my desktop")
    assert ph.folder == "Desktop"
    assert ph.name == ""
    assert ph.noun is True and ph.has_cue is True


def test_a_type_cue_narrows_without_naming():
    ph = filephrase.parse("the PDF I just downloaded")
    assert ph.suffixes == (".pdf",)
    assert ph.folder == "Downloads", "'downloaded' names the folder by verb"
    assert ph.just is True and ph.name == ""


def test_the_longest_type_cue_wins():
    """"word document" must be consumed before "document", or the leftover
    "word" becomes a name and every .docx stops matching."""
    ph = filephrase.parse("the word document on my desktop")
    assert ".docx" in ph.suffixes and ph.name == ""


def test_a_literal_filename_survives_the_cue_tables():
    """The regression the _FILENAME_RX park exists for: \\bpdf\\b matches
    inside "lab_report.pdf" because the dot is a word boundary, so without
    parking, the exact name he said is torn into "lab_report." plus a type
    filter and stops matching itself."""
    ph = filephrase.parse("lab_report.pdf")
    assert ph.name == "lab_report.pdf"
    assert ph.literals == ("lab_report.pdf",)


def test_a_name_plus_a_folder_keeps_both():
    ph = filephrase.parse("the lab report on my desktop")
    assert ph.folder == "Desktop" and ph.name == "lab report"


def test_recency_words_are_a_cue_and_not_a_name():
    ph = filephrase.parse("the latest one")
    assert ph.latest is True and ph.name == ""


def test_a_bare_recipient_specifies_nothing():
    """"email that to Heather" -- once the recipient is stripped off, the
    file half is "that", which is filler and nothing else."""
    ph = filephrase.parse("that")
    assert not ph.name and not ph.has_cue and not ph.noun


# ==================================================================
# 2. The shapes that must resolve
# ==================================================================
def test_a_literal_name_resolves_exactly(roots):
    m = filephrase.resolve("lab_report.pdf", roots=roots)
    assert m.ok and m.path.name == "lab_report.pdf"


def test_a_partial_name_resolves(roots):
    m = filephrase.resolve("the biosensors handout", roots=roots)
    assert m.ok and m.path.name == "Biosensors_Lab-Handout_v2.pdf"


def test_a_folder_cue_alone_resolves_when_the_folder_holds_one(tmp_path):
    only = _write(tmp_path / "Desktop" / "one.pdf", b"x" * 10)
    m = filephrase.resolve("that file on my desktop",
                           roots=[str(tmp_path / "Desktop")])
    assert m.ok and m.path == only.resolve()


def test_a_type_cue_narrows_the_folder(roots, home):
    """"the picture on my desktop" -- one image among four Desktop files."""
    m = filephrase.resolve("the picture on my desktop", roots=roots)
    assert m.ok and m.path.name == "holiday.jpg"


def test_recency_plus_type_resolves(roots):
    m = filephrase.resolve("the PDF I just downloaded", roots=roots)
    assert m.ok and m.path.name == "invoice-4471.pdf"


def test_the_latest_ranks_by_time(roots, home):
    m = filephrase.resolve("the latest file in my downloads", roots=roots)
    assert m.ok and m.path.name == "invoice-4471.pdf"


def test_depth_one_reaches_a_subfolder(roots):
    m = filephrase.resolve("thermo notes", roots=roots)
    assert m.ok and m.path.name == "thermo_notes.pdf"


def test_depth_one_does_not_reach_two_levels_down(roots):
    """The bound is what stops a spoken title reaching a stray notes.txt six
    levels into a git checkout -- and an unbounded walk of ~ is also seconds
    of stat() in the middle of a voice turn."""
    m = filephrase.resolve("buried", roots=roots)
    assert not m.ok and m.reason == "not-found"


def test_an_explicit_path_inside_the_roots_resolves(roots, home):
    want = home / "Desktop" / "notes.txt"
    m = filephrase.resolve(str(want), roots=roots)
    assert m.ok and m.path == want.resolve()


def test_an_explicit_path_outside_the_roots_is_allowed(tmp_path, roots):
    """He asked for "this file from this location"; a path he gave outright
    must work even when the location is not a search folder."""
    out = _write(tmp_path / "elsewhere" / "thesis.pdf", b"t" * 40)
    m = filephrase.resolve(str(out), roots=roots)
    assert m.ok and m.path == out.resolve()


def test_a_tilde_path_is_expanded(roots, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    want = _write(tmp_path / "elsewhere" / "x.pdf", b"y" * 20)
    m = filephrase.resolve("~/elsewhere/x.pdf", roots=roots)
    assert m.ok and m.path == want.resolve()


# ==================================================================
# 3. What it must REFUSE, however plainly he said it
# ==================================================================
def test_a_dot_folder_is_denied_even_as_an_explicit_path(tmp_path, roots):
    """~/.ssh, ~/.gnupg, ~/.config: every credential in this account lives
    behind a leading dot, and none of them is a document."""
    secret = _write(tmp_path / ".ssh" / "id_ed25519", b"k" * 50)
    assert filephrase.denied(secret) is True
    m = filephrase.resolve(str(secret), roots=roots)
    assert not m.ok and m.reason == "outside"


@pytest.mark.parametrize("path", [
    "/etc/shadow", "/etc/passwd", "/proc/self/environ", "/usr/bin/python3",
    "/var/log/syslog", "/boot/config",
])
def test_the_system_deny_list_refuses(path, roots):
    assert filephrase.denied(Path(path)) is True
    m = filephrase.resolve(path, roots=roots)
    assert not m.ok


def test_every_deny_root_is_actually_denied(roots):
    """Pinned as a whole so a root cannot be quietly dropped from the tuple
    without a test going red."""
    for root in filephrase.DENY_ROOTS:
        assert filephrase.denied(Path(root) / "anything") is True


def test_a_symlink_out_of_the_roots_is_never_a_candidate(tmp_path, home, roots):
    """The FIRST containment layer: _scan never offers a symlink at all, so
    a Desktop link pointing at a key cannot be reached by a spoken name."""
    secret = _write(tmp_path / "secret" / "id_ed25519", b"k" * 50)
    link = home / "Desktop" / "id_ed25519"
    link.symlink_to(secret)
    m = filephrase.resolve("id_ed25519", roots=roots)
    assert not m.ok and m.path is None


def test_a_folder_is_refused(home, roots):
    m = filephrase.resolve(str(home / "Desktop"), roots=roots)
    assert not m.ok and m.reason


def test_a_fifo_is_refused(tmp_path, roots):
    fifo = tmp_path / "Desktop" / "pipe"
    fifo.parent.mkdir(parents=True, exist_ok=True)
    os.mkfifo(fifo)
    m = filephrase.resolve(str(fifo), roots=roots)
    assert not m.ok and m.reason


def test_a_file_over_the_cap_is_refused_and_keeps_its_size(home, roots):
    big = _write(home / "Desktop" / "lecture.mp4", b"V" * (3 * 1024 * 1024))
    m = filephrase.resolve(str(big), roots=roots, max_mb=1)
    assert not m.ok and m.reason == "too-big"
    assert m.size == 3 * 1024 * 1024, "the refusal must be able to say how big"


def test_a_missing_path_says_not_found(roots):
    m = filephrase.resolve("/nowhere/at/all.pdf", roots=roots)
    assert not m.ok and m.reason == "not-found"


def test_nothing_said_is_empty_not_a_guess(roots):
    assert filephrase.resolve("", roots=roots).reason == "empty"
    assert filephrase.resolve("that", roots=roots).reason == "empty"


def test_no_roots_at_all_is_not_found():
    assert filephrase.resolve("anything", roots=[]).reason == "not-found"


def test_a_name_that_matches_nothing_is_not_found(roots):
    """A miss is a miss. There is no threshold at which a wrong file
    becomes acceptable."""
    m = filephrase.resolve("the widget specification", roots=roots)
    assert not m.ok and not m.ambiguous and m.reason == "not-found"


def test_a_one_letter_off_name_is_not_a_match(roots, home):
    """filepick's measured case: "budget" vs "gadget" scores 0.667, ABOVE
    the fuzzy floor, and is the wrong file. The prefix rule is what refuses
    it, and this lane must not have weakened that."""
    _write(home / "Desktop" / "gadget.png", b"g" * 100)
    m = filephrase.resolve("budget", roots=roots)
    assert not m.ok, "a mis-heard syllable must not resolve to a real file"


def test_a_named_folder_that_does_not_exist_is_not_found(tmp_path):
    m = filephrase.resolve("the file on my desktop",
                           roots=[str(tmp_path / "Documents")])
    assert not m.ok and m.reason == "not-found"


# ==================================================================
# 4. The tie band -- the whole safety margin
# ==================================================================
def test_two_similar_names_ask_instead_of_picking(roots):
    m = filephrase.resolve("the lab report", roots=roots)
    assert m.ambiguous and not m.ok
    assert {p.name for p in m.candidates} == {"lab_report.pdf",
                                              "lab_report_final.pdf"}


def test_the_band_is_wider_than_filepicks_on_purpose():
    """A copy to his own second machine is a mistake he can delete; an
    attachment in someone else's inbox is not. This lane buys certainty
    with a question and the shared one does not."""
    assert filephrase.TIE_SCORE > 0.02
    a = filepick.score("lab report", "lab_report.pdf")
    b = filepick.score("lab report", "lab_report_final.pdf")
    assert abs(a - b) <= filephrase.TIE_SCORE


def test_an_exact_filename_beats_a_near_neighbour(roots):
    """A literal name is not half-remembered, so the wide band must not
    turn the one unambiguous phrasing he has into a question."""
    m = filephrase.resolve("lab_report.pdf", roots=roots)
    assert m.ok and m.path.name == "lab_report.pdf"


def test_candidates_are_capped_and_the_real_total_is_reported(tmp_path):
    """Offering four of six invites him to pick from a list that need not
    contain the file he means -- so the count survives the cap and the
    caller can say the list was cut."""
    for i in range(6):
        _write(tmp_path / "Desktop" / f"report_{i}.pdf", b"r" * 100, i * 3600)
    m = filephrase.resolve("report", roots=[str(tmp_path / "Desktop")])
    assert m.ambiguous
    assert len(m.candidates) == filepick.MAX_CANDIDATES
    assert m.total == 6


def test_the_total_is_set_for_a_no_name_ambiguity_too(tmp_path):
    for i in range(5):
        _write(tmp_path / "Desktop" / f"f{i}.pdf", b"x" * 10, i * 3600)
    m = filephrase.resolve("that file on my desktop",
                           roots=[str(tmp_path / "Desktop")])
    assert m.ambiguous and m.total == 5
    assert len(m.candidates) == filepick.MAX_CANDIDATES


def test_a_two_way_ambiguity_reports_its_true_total(roots):
    m = filephrase.resolve("the lab report", roots=roots)
    assert m.total == 2 == len(m.candidates)


# ==================================================================
# 5. "I just downloaded it" is a claim, and it is checked
# ==================================================================
def test_just_is_refused_when_the_newest_is_old(tmp_path):
    old = filephrase.JUST_WINDOW_S + 600
    _write(tmp_path / "Downloads" / "a.pdf", b"a" * 10, old)
    _write(tmp_path / "Downloads" / "b.pdf", b"b" * 10, old + 600)
    m = filephrase.resolve("the PDF I just downloaded",
                           roots=[str(tmp_path / "Downloads")])
    assert m.ambiguous, "a file from last week is not the one he just got"


def test_files_written_together_are_never_split_by_timestamp(tmp_path):
    """Two arrivals inside TIE_WINDOW_S are not "the newest" and "an older
    one"; separating them by mtime is a coin flip."""
    now = time.time()
    for name in ("a.pdf", "b.pdf"):
        _write(tmp_path / "Downloads" / name, b"x" * 10)
    m = filephrase.resolve("the PDF I just downloaded",
                           roots=[str(tmp_path / "Downloads")], now=now)
    assert m.ambiguous


def test_a_clear_newest_inside_the_window_resolves(tmp_path):
    _write(tmp_path / "Downloads" / "new.pdf", b"n" * 10, 60)
    _write(tmp_path / "Downloads" / "old.pdf", b"o" * 10, 6 * 3600)
    m = filephrase.resolve("the PDF I just downloaded",
                           roots=[str(tmp_path / "Downloads")])
    assert m.ok and m.path.name == "new.pdf"


def test_without_a_recency_word_several_files_are_a_question(tmp_path):
    """No name, no recency: "the file in my downloads" with three in there
    has no safe resolution, however different their mtimes are."""
    for i, name in enumerate(("a.pdf", "b.pdf", "c.pdf")):
        _write(tmp_path / "Downloads" / name, b"x" * 10, i * 3600)
    m = filephrase.resolve("the file in my downloads",
                           roots=[str(tmp_path / "Downloads")])
    assert m.ambiguous


# ==================================================================
# 6. The module keeps its promises to the shared resolver
# ==================================================================
def test_the_skip_list_is_filepicks_own_object():
    """One rule, so ~/.ssh and ~/.config are out of reach from both lanes.
    Two lists that could drift is the failure the sharing prevents."""
    assert filephrase.SKIP_DIRS is filepick.SKIP_DIRS


def test_a_skipped_directory_is_never_walked(tmp_path):
    _write(tmp_path / "Desktop" / ".ssh" / "id_rsa.pdf", b"k" * 10)
    _write(tmp_path / "Desktop" / "real.pdf", b"r" * 10)
    m = filephrase.resolve("that file on my desktop",
                           roots=[str(tmp_path / "Desktop")])
    assert m.ok and m.path.name == "real.pdf"


def test_hidden_files_are_never_offered(tmp_path):
    _write(tmp_path / "Desktop" / ".hidden.pdf", b"h" * 10)
    _write(tmp_path / "Desktop" / "shown.pdf", b"s" * 10)
    m = filephrase.resolve("that file on my desktop",
                           roots=[str(tmp_path / "Desktop")])
    assert m.ok and m.path.name == "shown.pdf"


def test_resolve_never_reads_file_content(tmp_path, monkeypatch):
    """Resolving a name must not be the thing that loads a 2 GB video into
    memory, so nothing on this path may call read_bytes/read_text."""
    _write(tmp_path / "Desktop" / "big.pdf", b"z" * 5000)

    def _boom(self, *a, **k):
        raise AssertionError("filephrase read a file's CONTENT")

    monkeypatch.setattr(Path, "read_bytes", _boom)
    monkeypatch.setattr(Path, "read_text", _boom)
    m = filephrase.resolve("big", roots=[str(tmp_path / "Desktop")])
    assert m.ok


def test_the_module_imports_no_transport():
    """No smtplib, no requests, no socket: the roots are an argument and
    this module is the half of the mail lane that must stay pure."""
    import inspect
    src = inspect.getsource(filephrase)
    for banned in ("smtplib", "requests", "urllib", "socket", "subprocess"):
        assert banned not in src, f"filephrase must not import {banned}"
