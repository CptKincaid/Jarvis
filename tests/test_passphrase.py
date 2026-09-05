"""Both fallbacks: the spoken passphrase and the typed override code.

EVERY VALUE IN THIS FILE IS DELIBERATELY NOT A REAL ONE. The real phrase and
the real code are his, they are stored salted-hashed, and they must never
appear in a fixture, a log, or a config value anything prints.
"""
import logging

from jarvis import passphrase as pp

# Obvious non-values. If either of these is ever a plausible secret, the
# fixture has become a leak.
FAKE_PHRASE = "xxx-not-a-real-phrase-xxx"
FAKE_CODE = "xxx000"
OTHER = "yyy-also-not-real-yyy"


# ------------------------------------------------------------- hashing
def test_a_stored_secret_is_never_the_secret():
    stored = pp.hash_secret(FAKE_PHRASE)
    assert FAKE_PHRASE not in stored
    assert stored.startswith("scrypt$")
    assert pp.check_secret(FAKE_PHRASE, stored) is True
    assert pp.check_secret(OTHER, stored) is False


def test_the_salt_is_new_every_time():
    a, b = pp.hash_secret(FAKE_PHRASE), pp.hash_secret(FAKE_PHRASE)
    assert a != b
    assert pp.check_secret(FAKE_PHRASE, a) and pp.check_secret(FAKE_PHRASE, b)


def test_a_broken_or_empty_stored_value_is_a_no_never_a_crash():
    for junk in ("", None, "scrypt$", "scrypt$a$b$c$d$e", "plaintext",
                 "scrypt$16384$8$1$!!!$!!!"):
        assert pp.check_secret(FAKE_PHRASE, junk) is False


def test_an_empty_offering_never_matches_anything():
    stored = pp.hash_secret(FAKE_PHRASE)
    for empty in ("", None, "   "):
        assert pp.check_secret(empty, stored) is False


def test_the_scrypt_cost_is_the_one_that_was_measured():
    """19.7 ms per hash on this box at n=2**14 (measured 2026-09-03,
    hashlib.scrypt, mean of 3). It runs only when recognition has already
    failed AND the words were phrase-shaped, so it is off the turn budget
    for every ordinary sentence."""
    assert (pp.SCRYPT_N, pp.SCRYPT_R, pp.SCRYPT_P) == (2 ** 14, 8, 1)


# ------------------------------------------------- what a spoken phrase is
def test_spoken_normalisation_forgives_whisper_not_the_phrase():
    assert pp.normalise_spoken("  Open   The, Pod-bay. Doors!  ") == \
        "open the pod bay doors"
    assert pp.normalise_spoken(None) == ""


def test_a_normalised_phrase_matches_however_it_was_punctuated():
    stored = pp.hash_secret(pp.normalise_spoken(FAKE_PHRASE))
    assert pp.check_secret(pp.normalise_spoken("Xxx not a real phrase xxx!"),
                           stored) is True


def test_short_things_are_never_phrase_shaped():
    """The pre-filter is what keeps a 20 ms key derivation off every
    ordinary refused turn."""
    assert pp.phrase_shaped("yes") is False
    assert pp.phrase_shaped("") is False
    assert pp.phrase_shaped(FAKE_PHRASE) is True


def test_the_length_floors_are_stated_and_are_the_only_rule():
    """He was told, and accepted, that anyone who can already type at the
    Spark can edit the registry -- so the code is HIS recovery path, not a
    defence against them. No character-class policy is imposed."""
    assert pp.MIN_PHRASE_LEN == 12 and pp.MIN_CODE_LEN == 6
    assert pp.code_ok("aaaaaa")[0] is True          # letters only: fine
    assert pp.code_ok("123456")[0] is True          # digits only: fine
    assert pp.code_ok("aB3-x_")[0] is True
    assert pp.code_ok("abc")[0] is False
    assert pp.phrase_ok("too short")[0] is False


# ------------------------------------------------------- the two limiters
def test_the_limit_is_a_cool_off_and_not_a_lock():
    a = pp.Attempts(limit=2, window_s=100.0)
    assert a.allow(now=0.0) == (True, 0.0)
    a.record(now=0.0)
    a.record(now=1.0)
    ok, wait = a.allow(now=2.0)
    assert ok is False and 0 < wait <= 100.0
    assert a.allow(now=101.5)[0] is True


def test_burning_the_passphrase_never_locks_out_the_break_glass():
    """His explicit requirement. The fallback of last resort must not fail
    exactly when it is needed, so the two counters are separate objects
    with separate limits and share nothing."""
    phrase = pp.Attempts(limit=pp.PHRASE_LIMIT, window_s=pp.PHRASE_WINDOW)
    code = pp.Attempts(limit=pp.CODE_LIMIT, window_s=pp.CODE_WINDOW)
    for i in range(pp.PHRASE_LIMIT + 5):
        phrase.record(now=float(i))
    assert phrase.allow(now=10.0)[0] is False
    assert code.allow(now=10.0)[0] is True
    assert phrase is not code


def test_the_counters_live_in_memory_so_a_restart_clears_them():
    """A rate limiter that persists is itself a lockout."""
    import inspect
    src = inspect.getsource(pp.Attempts)
    for forbidden in ("open(", "Path", "json", "write_text", "PATHS"):
        assert forbidden not in src, forbidden


def test_the_clock_is_monotonic_so_an_ntp_step_cannot_extend_a_cool_off():
    import inspect
    src = inspect.getsource(pp)
    assert "time.monotonic" in src
    assert "time.time(" not in src and "datetime" not in src


# ------------------------------------------------------------- no leaks
def test_the_plaintext_never_reaches_a_log(caplog):
    with caplog.at_level(logging.DEBUG):
        stored = pp.hash_secret(FAKE_PHRASE)
        pp.check_secret(FAKE_PHRASE, stored)
        pp.check_secret(OTHER, stored)
        pp.phrase_ok(FAKE_PHRASE)
        pp.code_ok(FAKE_CODE)
        pp.normalise_spoken(FAKE_PHRASE)
    assert FAKE_PHRASE not in caplog.text
    assert FAKE_CODE not in caplog.text
    assert OTHER not in caplog.text


def test_the_module_logs_nothing_at_all():
    """The cheapest way to be sure a secret is not logged is to own no
    logger. A test, not a comment, because a future edit would not know.

    Read from the parsed module rather than the text, so the prose in the
    docstring explaining the rule cannot be mistaken for breaking it."""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(pp))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.Call):
            fn = node.func
            label = getattr(fn, "id", "") or getattr(fn, "attr", "")
            assert label not in ("print", "get_logger", "warning", "info",
                                 "debug", "exception"), label
    assert "logging" not in names and "jarvis" not in names


def test_nothing_here_carries_a_repr_that_could_hold_a_secret():
    a = pp.Attempts(limit=3, window_s=10.0)
    assert FAKE_PHRASE not in repr(a)


# ------------------------------------ round 2: the whitespace trap (R10)
def test_a_secret_typed_with_a_stray_space_still_works():
    """HIS BREAK-GLASS USED TO DIE ON ONE KEYSTROKE. hash_secret hashed the
    raw getpass string while check_override_code strips before checking, so
    a code set as "xxx000 " could be typed neither with the space nor
    without it -- and the CLI said "set." either way. Both ends strip now,
    so the two agree."""
    stored = pp.hash_secret(FAKE_CODE + " ")
    assert pp.check_secret(FAKE_CODE, stored) is True
    assert pp.check_secret(FAKE_CODE + " ", stored) is True
    assert pp.check_secret(" " + FAKE_CODE + "\n", stored) is True
    assert pp.check_secret(OTHER, stored) is False


def test_the_two_ends_hash_the_same_thing():
    """check_secret used to test the STRIPPED string for emptiness and then
    hash the RAW one; that asymmetry is what made the trap invisible."""
    assert pp.check_secret("  " + FAKE_PHRASE + "  ",
                           pp.hash_secret(FAKE_PHRASE)) is True
    assert pp.check_secret("", pp.hash_secret(FAKE_CODE)) is False
    assert pp.check_secret("   ", pp.hash_secret(FAKE_CODE)) is False
