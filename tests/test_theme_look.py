"""The UI look switch (jarvis/ui/theme.py, 2026-09-01).

"classic" must reproduce the 08-31 console token for token -- that is the
fallback Hunter asked for -- so the oracle is a JSON dump of every theme
name as at 85d5066 (before any restyle) and this file compares against it.
"holo" is the blue-holographic overhaul. Both come out of ONE derivation,
so a switch at runtime re-derives everything; the guard test at the bottom
keeps the other UI modules honest about reading theme.X at call time.

No Tk: theme imports tkinter.font but never creates a root.
"""
import ast
import json
import os
import pathlib
import re

import pytest

from jarvis.ui import theme

REPO = pathlib.Path(__file__).resolve().parents[1]
# Tracked next to the other fixtures: the first cut pointed at
# scratchpad/holo/classic/, which .gitignore drops, so on any other clone the
# exactness guard hard-failed -- and anyone could re-dump a drifted classic
# and the test would have blessed it (09-01 review).
ORACLE = REPO / "tests" / "fixtures" / "theme_tokens_85d5066.json"


@pytest.fixture(autouse=True)
def _restore_look():
    """Every test leaves the module in the import-time state (holo, scale
    1.0) so nothing downstream inherits a classic theme by accident."""
    yield
    theme.apply_scale(1.0)
    theme.select_look(theme.DEFAULT_LOOK)


def _oracle() -> dict:
    return json.loads(ORACLE.read_text())


def _current(key):
    value = getattr(theme, key)
    if isinstance(value, tuple):
        # JSON holds lists; the tokens stay tuples (CORE_BANDS is indexed,
        # SEAM_STEPS/TV_POOL are iterated, _DISPLAY_FACES is unpacked).
        return [list(v) if isinstance(v, tuple) else v for v in value]
    return value


# ------------------------------------------------------------ contract
def test_looks_and_default():
    assert theme.LOOKS == ("holo", "classic")
    assert theme.DEFAULT_LOOK == "holo"
    assert theme.LOOK in theme.LOOKS


def test_classic_reproduces_every_oracle_token():
    assert ORACLE.exists(), "the classic oracle JSON is a tracked fixture"
    oracle = _oracle()
    assert len(oracle) == 62
    assert theme.select_look("classic") == "classic"
    assert theme.LOOK == "classic"
    bad = {k: (v, _current(k)) for k, v in oracle.items() if _current(k) != v}
    assert not bad, bad


def test_round_trip_holo_classic_holo_is_consistent():
    holo_before = {k: _current(k) for k in theme.LOOK_TOKENS}
    assert theme.LOOK == "holo"
    assert theme.select_look("classic") == "classic"
    assert theme.BG == "#0d1b2a" and theme.LOOK == "classic"
    oracle = _oracle()
    assert all(_current(k) == v for k, v in oracle.items())
    assert theme.select_look("holo") == "holo"
    assert theme.LOOK == "holo" and theme.BG == "#050b14"
    assert theme.CYAN == "#35e0ff"
    assert {k: _current(k) for k in theme.LOOK_TOKENS} == holo_before
    # ...and classic still equals the oracle the second time round
    theme.select_look("classic")
    assert all(_current(k) == v for k, v in oracle.items())


def test_unknown_look_falls_back_to_holo():
    theme.select_look("classic")
    assert theme.select_look("nonsense") == "holo"
    assert theme.LOOK == "holo" and theme.BG == "#050b14"
    assert theme.select_look(None) == "holo"
    assert theme.select_look("") == "holo"


def test_select_look_is_case_insensitive():
    assert theme.select_look("CLASSIC") == "classic"
    assert theme.select_look(" Holo ") == "holo"


def test_every_look_token_is_derived_not_frozen():
    """Every colour that differs between the looks changes on a switch;
    the invariants (accent, amber, red, white focal) do not."""
    holo = {k: _current(k) for k in theme.LOOK_TOKENS}
    theme.select_look("classic")
    classic = {k: _current(k) for k in theme.LOOK_TOKENS}
    same = {k for k in holo if holo[k] == classic[k]}
    assert {"CYAN", "CYAN_DIM", "WARN", "ERR", "FOCAL", "OK"} <= same
    changed = set(holo) - same
    assert {"BG", "SURFACE", "RAISED", "LINE", "GLASS_EDGE", "RAIL", "FRAME",
            "INK", "MUTED", "FAINT", "TV_BG", "SEAM_STEPS", "CORE_BANDS",
            "STATE_COLORS"} <= changed
    # amber is reserved for warnings in BOTH looks
    assert holo["WARN"] == classic["WARN"] == "#ffb454"
    assert holo["STATE_COLORS"]["thinking"] == "#ffb454"


# ------------------------------------------------------------ holo anchors
def test_holo_is_glass_over_film_black():
    """The overhaul's brief: near-black navy ground, surfaces so thin they
    never read as slabs, brighter 1px strokes, text lifted for contrast."""
    assert theme.LOOK == "holo"
    assert theme.BG == "#050b14"
    assert theme.CYAN == "#35e0ff"
    holo = {k: _current(k) for k in ("SURFACE", "RAISED", "LINE", "GLASS_EDGE",
                                     "RAIL", "FRAME", "MUTED", "FAINT")}
    theme.select_look("classic")
    classic = {k: _current(k) for k in holo}

    def lum(h):
        r, g, b = theme._rgb(h)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def lift(h, ground):
        return lum(h) - lum(ground)

    # surfaces: less lift over their own ground than classic's
    for k in ("SURFACE", "RAISED"):
        assert lift(holo[k], "#050b14") < lift(classic[k], "#0d1b2a") * 0.7, k
    # strokes: MORE lift over their ground than classic's
    for k in ("LINE", "GLASS_EDGE", "RAIL", "FRAME"):
        assert lift(holo[k], "#050b14") > lift(classic[k], "#0d1b2a"), k
    # secondary text lifted ~10%
    for k in ("MUTED", "FAINT"):
        assert 1.05 < lum(holo[k]) / lum(classic[k]) < 1.20, k


def test_holo_transcript_ground_is_flat_so_frame_cards_match_it():
    """Tk has no alpha: a card's interior is ONE colour, card_look makes it
    TV_BG, and the 09-01 review measured the pool/gradient behind the holo
    cards turning each into a 7-25 level darker slab. So in holo the top
    lift is zero (the stage seam lands on TV_BG) and the transcript draws
    no pool/floor/gradient; classic keeps them (oracle)."""
    from jarvis.ui.views import card_look, ground_is_flat
    assert theme.LOOK == "holo"
    assert theme.TV_TOP == theme.TV_BG
    assert theme.SEAM_STEPS[-1] == theme.TV_BG and theme.SEAM_STEPS[0] != theme.TV_BG
    assert ground_is_flat() is True and ground_is_flat("classic") is False
    for role in ("jarvis", "user", "partial", "progress", "briefing"):
        assert card_look(role)["fill"] == theme.TV_BG, role
    theme.select_look("classic")
    assert ground_is_flat() is False
    assert theme.TV_TOP != theme.TV_BG
    assert theme.TV_TOP == _oracle()["TV_TOP"]
    assert list(theme.SEAM_STEPS) == list(_oracle()["SEAM_STEPS"])


# ------------------------------------------------------------ resolve_look
@pytest.mark.parametrize("env, option, expected", [
    ({"JARVIS_LOOK": "classic"}, None, "classic"),
    ({"JARVIS_LOOK": "classic"}, "holo", "classic"),       # env beats option
    ({"JARVIS_LOOK": "HOLO"}, "classic", "holo"),          # case-insensitive
    ({"JARVIS_LOOK": " Classic "}, None, "classic"),
    ({}, "classic", "classic"),                            # option next
    ({}, "Classic", "classic"),
    ({}, None, "holo"),                                    # default
    ({}, "", "holo"),
    ({}, "nonsense", "holo"),                              # junk -> holo
    ({"JARVIS_LOOK": "junk"}, "classic", "classic"),       # junk env falls through
    ({"JARVIS_LOOK": "junk"}, None, "holo"),
    ({"JARVIS_LOOK": ""}, "classic", "classic"),
    ({}, 7, "holo"),                                       # a non-string option
])
def test_resolve_look_precedence(env, option, expected):
    calls = []

    def get_option(key, default=None):
        calls.append(key)
        return option

    assert theme.resolve_look(env, get_option) == expected
    if env.get("JARVIS_LOOK", "").strip().lower() in theme.LOOKS:
        assert calls == []                  # a valid env never reads the config
    else:
        assert calls == ["console.look"]


def test_resolve_look_without_a_reader_or_with_a_broken_one():
    assert theme.resolve_look({}, None) == "holo"
    assert theme.resolve_look({}, "not callable") == "holo"

    def boom(key, default=None):
        raise RuntimeError("config on fire")

    assert theme.resolve_look({}, boom) == "holo"
    assert theme.resolve_look({"JARVIS_LOOK": "classic"}, boom) == "classic"


def test_resolve_look_defaults_to_the_process_environment(monkeypatch):
    monkeypatch.setenv("JARVIS_LOOK", "classic")
    assert theme.resolve_look() == "classic"
    monkeypatch.delenv("JARVIS_LOOK")
    assert theme.resolve_look() == "holo"
    assert os.environ.get("JARVIS_LOOK") is None


def test_resolve_look_does_not_switch():
    """resolve_look only ANSWERS; select_look applies."""
    assert theme.resolve_look({"JARVIS_LOOK": "classic"}) == "classic"
    assert theme.LOOK == "holo"


# ------------------------------------------------------------ scale
def test_apply_scale_after_select_look_scales_from_the_base():
    theme.select_look("classic")
    theme.apply_scale(2.0)
    assert (theme.PAD, theme.PAD_S, theme.PAD_L, theme.RADIUS, theme.CHAMFER) \
        == (32, 16, 48, 20, 20)
    theme.apply_scale(1.5)                      # idempotent: never compounds
    assert theme.PAD == 24
    theme.select_look("holo")
    theme.apply_scale(2.0)
    assert theme.PAD == 32 and theme.RADIUS == 20


def test_select_look_keeps_the_scale_already_applied():
    """MainWindow calls apply_scale once; a later switch (tests, a future
    live toggle) must not silently drop the window back to 1x spacing."""
    theme.apply_scale(2.0)
    theme.select_look("classic")
    assert theme.PAD == 32 and theme.CHAMFER == 20
    theme.select_look("holo")
    assert theme.PAD == 32


def test_spacing_and_type_scale_are_look_independent():
    holo = (theme.PAD, theme.PAD_S, theme.PAD_L, theme.RADIUS, theme.CHAMFER,
            theme.SIZE_WORDMARK, theme.SIZE_BODY, theme.SIZE_LABEL,
            theme.SIZE_CAPTION, theme.FOCAL_WORD_STATES)
    theme.select_look("classic")
    assert (theme.PAD, theme.PAD_S, theme.PAD_L, theme.RADIUS, theme.CHAMFER,
            theme.SIZE_WORDMARK, theme.SIZE_BODY, theme.SIZE_LABEL,
            theme.SIZE_CAPTION, theme.FOCAL_WORD_STATES) == holo
    assert "PAD" not in theme.LOOK_TOKENS and "SIZE_BODY" not in theme.LOOK_TOKENS


# ------------------------------------------------------------ the option key
def test_commander_writes_the_key_theme_reads():
    import jarvis.commander as commander
    assert commander.UI_LOOK_OPTION == theme.OPTION_KEY == "console.look"
    # ...and checks the same env var resolve_look lets outrank that key
    assert commander.UI_LOOK_ENV == theme.ENV_KEY == "JARVIS_LOOK"


def test_the_option_is_discoverable_in_the_config_defaults():
    """Only the voice command ever created console.look, so a hand-editing
    user had nothing to find in assistant.json (09-01 review). The default
    must resolve exactly as an absent key does."""
    from jarvis.assistant_config import DEFAULTS
    section, key = theme.OPTION_KEY.split(".")
    assert DEFAULTS[section][key] == theme.DEFAULT_LOOK
    assert theme.resolve_look({}, lambda k, d=None: DEFAULTS[section][key]) \
        == theme.resolve_look({}, lambda k, d=None: None) == theme.DEFAULT_LOOK


# ------------------------------------------------------------ call-time guard
# A def default or a class-body dict evaluated at import freezes whichever
# look was current at import -- always holo, since select_look runs in
# create(). In classic those widgets would then paint holo colours: the
# stage ground behind the sphere did exactly that in the first A1 shots
# (reactor.py:179 bg=theme.BG). The scan flags only LOOK_TOKENS: the type
# scale and the fonts are the same in both looks, so capturing them is fine.
UI_DIR = REPO / "jarvis" / "ui"

# Files another lane owns THIS round (W2, 2026-09-01) with captures it must
# turn into call-time reads. xfail(strict=True): the moment a file is clean
# the test XPASSes, fails the run, and the entry here must be deleted.
KNOWN_OFFENDERS = {
    # widgets.py fixed 2026-09-01 (lane B): Card/Meter/Chip/RoundButton
    # defaults resolve in __init__, RoundButton._kinds() / Toast._kind_fg()
    # build per call
    # reactor.py fixed 2026-09-01 (lane A2): Reactor.__init__(bg=None)
    # resolves theme.BG in the body
    # ambient.py, console_mode.py, board.py: cleaned by lane C 2026-09-01
    # (quiet_ink(), dim() resolving theme.BG at call time, tone_color()).
}


def _def_time_theme_captures(path: pathlib.Path) -> list:
    """(line, where, token) for every theme.<LOOK_TOKEN> evaluated at
    import: module-level assignments, class-body assignments and function
    default arguments (module- or class-level defs)."""
    tree = ast.parse(path.read_text(), filename=str(path))
    hits = []

    def token_in(node):
        for n in ast.walk(node):
            if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                    and n.value.id == "theme" and n.attr in theme.LOOK_TOKENS):
                return n.attr
        return None

    def defaults(fn, where):
        args = fn.args
        for d in list(args.defaults) + [d for d in args.kw_defaults if d]:
            tok = token_in(d)
            if tok:
                hits.append((fn.lineno, where, tok))

    def scan(body, where):
        for node in body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                tok = token_in(node.value)
                if tok:
                    hits.append((node.lineno, where, tok))
            elif isinstance(node, ast.ClassDef):
                scan(node.body, f"class {node.name}")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defaults(node, f"{where}.{node.name}")

    scan(tree.body, path.stem)
    return hits


def _ui_modules():
    return sorted(p.name for p in UI_DIR.glob("*.py")
                  if p.name not in ("theme.py", "__init__.py"))


@pytest.mark.parametrize("name", _ui_modules())
def test_ui_modules_read_theme_at_call_time(name, request):
    if name in KNOWN_OFFENDERS:
        request.applymarker(pytest.mark.xfail(
            strict=True,
            reason=f"{name}: def-time theme captures owned by another W2 lane; "
                   "remove it from KNOWN_OFFENDERS once fixed"))
    hits = _def_time_theme_captures(UI_DIR / name)
    assert not hits, (
        f"{name} freezes the import-time look: {hits} -- read theme.X inside "
        "the function/method body (e.g. fill = fill or theme.RAISED)")


def test_capture_scanner_sees_what_it_should(tmp_path):
    """The guard is only as good as its scanner."""
    src = tmp_path / "m.py"
    src.write_text(
        "from jarvis.ui import theme\n"
        "A = theme.BG\n"                          # module assign  -> hit
        "B = theme.SIZE_BODY\n"                   # type scale     -> fine
        "class W:\n"
        "    K = {'ok': theme.OK}\n"              # class dict     -> hit
        "    def __init__(self, fill=theme.RAISED, size=theme.SIZE_LABEL):\n"
        "        self.x = theme.CYAN\n"           # call-time      -> fine\n"
        "def f(ground: str = theme.BG) -> str:\n"  # module def     -> hit
        "    return theme.LINE\n")
    hits = _def_time_theme_captures(src)
    assert [(h[2], h[1]) for h in hits] == [
        ("BG", "m"), ("OK", "class W"), ("RAISED", "class W.__init__"), ("BG", "m.f")]


def test_no_hex_literals_leak_out_of_theme():
    """The docstring's rule -- every colour comes from theme -- read
    literally for the look switch: a quoted "#rrggbb" in another UI module
    is a colour the switch cannot reach, so classic would stop matching the
    oracle on screen while still matching it in this file. Clean today
    (avatar_bake's CLI defaults are bare hex PARAMETERS, not literals)."""
    rx = re.compile(r"[\"']#[0-9a-fA-F]{6}[\"']")
    leaks = {}
    for name in _ui_modules():
        text = (UI_DIR / name).read_text()
        found = sorted({m.group(0) for m in rx.finditer(text)})
        if found:
            leaks[name] = found
    assert not leaks, leaks
