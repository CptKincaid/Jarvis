"""DERIVE the answer grammars; do not list them.

2026-09-05: a filled pause in front of an answer ("Uh, yeah.") was thrown
away by every start-anchored answer grammar in the app. The first fix
stripped the fillers at six sites and published an eight-item inventory
as complete. An adversary found six more; an AST sweep with a hand-written
set of rung names found five more; this census, which derives the rungs
instead of naming them, found nine more on top (the send-ask lane's three
answer parsers, "that's enough" and "cancel that" over an open session or
card, the three enrolment controls over an open CAMERA, the day-shift
follow-up, and a hole in the send lane's own backstop). Every inventory
was wrong by exactly the rungs its author had not read.

This module finds them by SHAPE and DATA FLOW, from the AST of every file
under jarvis/. Nothing here imports the modules it reads.

  ANSWER STATE. The codebase has its own name for "a question Jarvis put
    and is waiting on": the ``_pending_*`` slots and the ``*_offer`` dicts
    ``Commander.question_open`` enumerates, ``pending_event`` on the
    calendar, the router's ``.pending()``; and for "the turn he is
    replying to": ``_last_turn`` and ``_last_undo``. ANSWER_STATE_RX.

  A RUNG is any function that reads answer state. It is where his words
    are judged as a reply rather than as a fresh command.

  A PARSER is any function a rung hands those words to (``parse_yes_no``,
    ``pick_from_answer``, ``leavetime.answer_minutes``, a session's
    ``settle`` reached duck-typed), followed transitively to FOLLOW_DEPTH.
    Not followed: a DISPATCHER (``_handle_inner`` -- a callee that calls
    DISPATCHER_RUNGS or more rungs, or hands the utterance to one), and a
    ROUTER on the command path (``_handle_known``, ``_match_assistant``:
    something a dispatcher hands every utterance to that applies
    ROUTER_SITES grammars of its own or fans the words out to
    ROUTER_FANOUT callees). Through those every command grammar in the
    file would be "in answer position", which is the opposite of the
    point. A LEAF on the command path IS followed: ``quiet_kind`` is a
    barge-in grammar for the dispatcher and a stop word for the quiz
    rung, and it is the second use this census is about.

  A SITE is a start-anchored regex applied, inside a rung or a parser, to
    a string that flows from his words: ``R.match(x)`` / ``R.fullmatch(x)``
    (start-anchored by definition), or ``R.search(x)`` / ``R.sub(_, x)``
    whose pattern -- reassembled from the tree, so a regex built from
    module constants is read as Python reads it -- opens with ``^``.

  STRIPPED means the string reaching the site passed through a STRIPPER on
    its way. The canonical stripper is ``jarvis.endpoint.strip_fillers``
    (and ``strip_inline_fillers``, the undo lane's whole-phrase variant
    built FROM the same FILLER_WORDS). A DERIVED stripper is any function
    with a word parameter whose every tainted ``return`` is stripped --
    ``_file_answer`` after its fix, ``briefing_answer``, ``correction_kind``
    -- found by a fixpoint over the tree, so a helper that strips on its
    first line launders nothing by hand. A SINK is a function whose
    returns never carry his words (``_person_from_answer`` returns one of
    the CANDIDATES); its result is clean. A KNOWN stripper is one the test
    names because the derivation cannot: ``_send_clean``, the send lane's
    own pre-existing filler lists, whose ``else t`` return makes it a
    passthrough on paper. (The other pre-existing list,
    ``leavetime._ANSWER_FILLER``, is a startswith loop the flow cannot see
    and ``answer_minutes`` returns an int -- a sink -- so the site behind it
    is named in the test's KNOWN table instead.) The flow is traced through
    assignments in source order and across the call into a parser.

Round 2 of the adversary (09-06) planted eight answer grammars and six
walked past this census green. What it sees now, and what it still does
not -- the BLIND SPOTS, so nobody reads its green as a proof:

  SEEN since round 2 (tests/test_answer_census.py plants each one):
  * A MODE that owns the next words the way a parked question does --
    the ringing alarm, lecture notes, dictation (MODE_STATE): the rung
    that reads it is a rung. Two of those were invisible.
  * A grammar that is not a regex: ``text.startswith(("yes", "yeah"))``,
    ``t.endswith(...)``, ``t in ("yes", "y")`` / ``t in _YES_WORDS``.
  * A rung's words under ANY parameter name, annotated or not
    (``utterance``): every positional parameter of a rung is tainted on
    entry unless the body only ever uses it as an object (``c.services``,
    ``m.group(1)``) or it is the channel flag (FLAG_PARAMS).
  * A regex held on the class or the instance (``self._RX``) and a local
    alias (``rx = _NO_RX; rx.match(t)``), an ``a if x else b`` between two,
    and a ``for rx in (_A_RX, _B_RX)`` loop.
  * A registry HANDLER is derived from the dispatch table (a module-level
    list of calls naming it) and is never a rung: it runs on words the
    table already matched as a command.

  BLIND, still:
  * A rung that reads its parked question through a name outside the
    convention -- neither ANSWER_STATE_RX nor MODE_STATE (the adversary's
    P3, ``_awaiting_confirm``). Deriving state from a read-and-clear
    shape was tried on 09-06 and measured at 218 rungs and 39 false
    sites; the convention stays, and tests/test_answer_census.py pins a
    floor of rungs and sites so a drift in it fails loudly.
  * The flow is optimistic about branches: a strip on one branch that
    comes LAST in source order counts for everything below it (P8). A
    strip that runs only sometimes passes.
  * A piece a search picked out (``m.group(1)``) is not the reply, and a
    regex over it is not a site: a leading filler never lands inside a
    capture that a mid-sentence search chose. A grammar that captures
    from ``^`` is caught at the capturing search instead.
  * It says nothing about whether a grammar SHOULD strip. The test's KNOWN
    table names each unstripped site that stays, with the kind and the
    reason; a stale row fails the test too.
  * A duck-typed call (``session.settle(t)``) is followed into every
    function of that name when there are DUCK_MAX or fewer, and never
    launders taint: only a precisely resolved callee can be a stripper or
    a sink.
  * A dispatch table built by calls (``registry.add(rx, handler)``) rather
    than a module-level list is not read as a table; its handlers would
    count as rungs if they read answer state.
"""
from __future__ import annotations

import ast
import pathlib
import re
from dataclasses import dataclass, field
from typing import Optional

ANSWER_STATE_RX = re.compile(
    r"^(?:_pending_\w+|\w+_offer|pending_event|_last_turn|_last_undo)$")
# A sticky MODE owns the next words exactly as a parked question does, and
# the rung that reads it judges his words as a steer rather than a command:
# the ringing alarm ("stop" / "snooze"), lecture notes ("end notes"),
# dictation ("end dictation"). Two of these rungs were invisible until
# 09-06 -- "uh, stop" did not silence a six o'clock alarm, "uh, end notes"
# was filed as a note -- because their state has no _pending_ in its name.
MODE_STATE = frozenset({"ringing", "lecture_course", "dictation"})
ANSWER_STATE_CALLS = frozenset({"pending"})       # router.pending()
CANONICAL = frozenset({"strip_fillers", "strip_inline_fillers"})
WORD_PARAMS = frozenset({"text", "said"})
# The channel flag. A ``str`` like the words, and never the words: it is
# compared to "voice" and never parsed. The one parameter name the taint
# skips, and it fails LOUD if a rung ever carried words under that name
# (a false site), never silent.
FLAG_PARAMS = frozenset({"source"})
DISPATCHER_RUNGS = 3          # a callee calling this many rungs is not followed
ROUTER_SITES = 4              # ...nor a command-path callee with this many own grammars
ROUTER_FANOUT = 3             # ...nor one handing the words to this many callees
FOLLOW_DEPTH = 4
DUCK_MAX = 6                  # a name with more homonyms than this is generic
FIXPOINT_ROUNDS = 8
CAPTURE_METHODS = frozenset({"group", "groups", "groupdict"})
# str/list/dict/re/log methods a duck-typed lookup must never follow
_NOT_DUCK = frozenset({
    "append", "extend", "get", "pop", "setdefault", "update", "keys", "items",
    "values", "join", "split", "strip", "lstrip", "rstrip", "lower", "upper",
    "startswith", "endswith", "replace", "format", "sub", "subn", "match",
    "search", "fullmatch", "finditer", "findall", "info", "debug", "warning",
    "error", "exception", "add", "remove", "discard", "count", "index",
    "encode", "decode", "write", "read", "put", "send", "emit", "log",
})
_FLAGS_RX = re.compile(r"^(?:\(\?[aiLmsux]+\))+")
_OPENERS_RX = re.compile(r"^(?:\(\?:|\()+")
_SITE_METHODS = ("match", "fullmatch", "search", "sub", "subn")
# A start-anchored grammar that is not a regex at all: "yes" in
# ``text.startswith(("yes", "yeah"))``, or the whole reply in
# ``t in ("yes", "y")``. The adversary's P2 (09-06) walked straight past
# the census with one of these.
_EDGE_METHODS = ("startswith", "endswith")


def is_answer_state(name: str) -> bool:
    return bool(ANSWER_STATE_RX.match(name)) or name in MODE_STATE


# The str methods: a parameter these are called on is being read as WORDS.
_STR_METHODS = frozenset({
    "lower", "upper", "casefold", "strip", "lstrip", "rstrip", "split",
    "rsplit", "splitlines", "startswith", "endswith", "replace", "title",
    "capitalize", "partition", "rpartition", "find", "rfind", "isdigit",
    "isalpha", "encode", "removeprefix", "removesuffix", "count", "index",
})


@dataclass(frozen=True)
class Site:
    module: str            # path relative to the root, posix
    function: str          # qualname: "f" or "Class.m"
    regex: str             # the compiled regex's name, or "re.<method>(<pattern>)"
    method: str            # match | fullmatch | search | sub | subn
    arg: str               # the argument expression, as written
    stripped: bool
    via: str               # how the string got here
    lineno: int
    chain: str = ""        # rung > parser > ... that led here

    @property
    def key(self) -> str:
        return f"{self.module}:{self.function}:{self.regex}"

    def __str__(self) -> str:
        state = "stripped" if self.stripped else "UNSTRIPPED"
        return (f"{self.key}:{self.method}({self.arg}) [{state}; {self.via}] "
                f"L{self.lineno} <{self.chain}>")


@dataclass
class _Module:
    path: str
    tree: ast.Module
    consts: dict = field(default_factory=dict)
    regexes: dict = field(default_factory=dict)      # name | "Class.attr" -> pattern | None
    collections: dict = field(default_factory=dict)  # name -> [pattern | None]
    wordsets: dict = field(default_factory=dict)     # name -> [str, ...] (a tuple/set of words)
    tabled: set = field(default_factory=set)         # functions a module-level TABLE dispatches to
    imports: dict = field(default_factory=dict)      # local -> (module path, name | None)
    functions: dict = field(default_factory=dict)    # qualname -> FunctionDef


@dataclass
class Census:
    root: pathlib.Path
    sites: list = field(default_factory=list)
    rungs: set = field(default_factory=set)          # (module, qualname)
    parsers: set = field(default_factory=set)        # (module, qualname)
    dispatchers: set = field(default_factory=set)
    routers: set = field(default_factory=set)        # command-path callees not followed
    handlers: set = field(default_factory=set)       # table-dispatched: on the command path by construction
    command_path: set = field(default_factory=set)
    kinds: dict = field(default_factory=dict)        # (module, qualname) -> strip|sink|pass
    known: dict = field(default_factory=dict)        # (module, qualname) -> the name given
    why: dict = field(default_factory=dict)          # rung -> the token that made it one
    tokens: dict = field(default_factory=dict)       # rung -> every token it reads

    def unstripped(self) -> list:
        return [s for s in self.sites if not s.stripped]

    def keys(self) -> tuple:
        return tuple(sorted({s.key for s in self.sites}))

    def strippers(self) -> set:
        return {k for k, v in self.kinds.items() if v == "strip"}

    def sinks(self) -> set:
        return {k for k, v in self.kinds.items() if v == "sink"}

    def report(self) -> str:
        lines = [f"rungs: {len(self.rungs)}  parsers: {len(self.parsers)}  "
                 f"derived strippers: {len(self.strippers())}  sinks: {len(self.sinks())}  "
                 f"known strippers: {sorted(q for _, q in self.known)}",
                 f"dispatchers: {sorted(q for _, q in self.dispatchers)}",
                 f"table-dispatched handlers (never rungs): {len(self.handlers)}",
                 f"routers not followed: {sorted(q for _, q in self.routers)}",
                 f"sites: {len(self.sites)}  unstripped: {len(self.unstripped())}"]
        lines += ["  " + str(s) for s in sorted(self.sites, key=lambda s: (s.module, s.lineno))]
        lines.append("rungs, and the token that made each one:")
        lines += [f"  {p}:{q}  {self.why[(p, q)]}" for p, q in sorted(self.rungs)]
        return "\n".join(lines)


# ---------------------------------------------------------------- patterns
def _resolve(node, consts) -> Optional[str]:
    """The string a module-level expression evaluates to, or None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return consts.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _resolve(node.left, consts), _resolve(node.right, consts)
        if left is not None and right is not None:
            return left + right
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        return _resolve(node.left, consts)          # "%s" % x: the anchor is on the left
    if isinstance(node, ast.JoinedStr):
        out = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                out.append(str(v.value))
            elif isinstance(v, ast.FormattedValue):
                s = _resolve(v.value, consts)
                if s is None:
                    return None
                out.append(s)
        return "".join(out)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("format", "join")):
        if node.func.attr == "join":
            return None                              # "|".join(...) has no anchor
        return _resolve(node.func.value, consts)
    if isinstance(node, ast.Subscript):
        base = _resolve(node.value, consts)
        return None if base is None else ""         # a slice of a constant: no anchor
    return None


def anchored(pattern: Optional[str]) -> bool:
    """Does the pattern open with ^ or \\A, after flag groups and openers?"""
    if pattern is None:
        return False
    p = _OPENERS_RX.sub("", _FLAGS_RX.sub("", pattern.lstrip()))
    return p.startswith("^") or p.startswith("\\A")


def _is_re_compile(node) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "compile" and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "re")


# ------------------------------------------------------------------ modules
def _load(root: pathlib.Path) -> dict:
    mods = {}
    for f in sorted(root.rglob("*.py")):
        rel = f.relative_to(root).as_posix()
        m = _Module(rel, ast.parse(f.read_text()))
        pkg_dir = f.parent
        for node in m.tree.body:
            bound = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name):
                bound = (node.targets[0].id, node.value)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                    and node.value is not None:      # REGISTRY: list[Command] = [...]
                bound = (node.target.id, node.value)
            if bound is not None:
                name, val = bound
                if _is_re_compile(val):
                    m.regexes[name] = _resolve(val.args[0], m.consts) if val.args else None
                elif isinstance(val, (ast.Tuple, ast.List)) and val.elts \
                        and all(_is_re_compile(e) for e in val.elts):
                    m.collections[name] = [
                        _resolve(e.args[0], m.consts) if e.args else None for e in val.elts]
                elif _words_of(val) is not None:
                    m.wordsets[name] = _words_of(val)
                elif isinstance(val, (ast.List, ast.Tuple)) and val.elts \
                        and all(isinstance(e, ast.Call) for e in val.elts):
                    # A dispatch TABLE: ``REGISTRY = [Command("send file",
                    # _SEND_FILE_RX.match, _h_send_file), ...]``. A function
                    # named in one runs on words the table already matched
                    # as a COMMAND; it is never judging a reply, however
                    # many _pending_ slots it reads while doing its job.
                    for e in val.elts:
                        for a in list(e.args) + [k.value for k in e.keywords]:
                            if isinstance(a, ast.Name):
                                m.tabled.add(a.id)
                else:
                    s = _resolve(val, m.consts)
                    if s is not None:
                        m.consts[name] = s
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = pkg_dir
                    for _ in range(node.level - 1):
                        base = base.parent
                    target = base / (node.module.replace(".", "/") if node.module else "")
                elif node.module and node.module.split(".")[0] == root.name:
                    target = root / "/".join(node.module.split(".")[1:])
                else:
                    continue
                for alias in node.names:
                    local = alias.asname or alias.name
                    as_mod = target / (alias.name + ".py")
                    as_pkg = target / alias.name / "__init__.py"
                    if as_mod.exists():
                        m.imports[local] = (as_mod.relative_to(root).as_posix(), None)
                    elif as_pkg.exists():
                        m.imports[local] = (as_pkg.relative_to(root).as_posix(), None)
                    else:
                        modfile = pathlib.Path(str(target) + ".py")
                        if not modfile.exists():
                            modfile = target / "__init__.py"
                        if modfile.exists():
                            m.imports[local] = (modfile.relative_to(root).as_posix(), alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    if parts[0] != root.name or len(parts) < 2:
                        continue
                    modfile = root / ("/".join(parts[1:]) + ".py")
                    if modfile.exists():
                        m.imports[alias.asname or alias.name] = (
                            modfile.relative_to(root).as_posix(), None)
        for node in m.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                m.functions[node.name] = node
            elif isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        m.functions[f"{node.name}.{sub.name}"] = sub
                    # A regex held on the CLASS (``_RX = re.compile(...)`` in
                    # the body, read as ``self._RX``): the adversary's P5.
                    elif isinstance(sub, ast.Assign) and len(sub.targets) == 1 \
                            and isinstance(sub.targets[0], ast.Name) \
                            and _is_re_compile(sub.value):
                        m.regexes[f"{node.name}.{sub.targets[0].id}"] = (
                            _resolve(sub.value.args[0], m.consts) if sub.value.args else None)
                # ...or bound on the instance in __init__, from a compile or
                # from a module regex.
                init = next((s for s in node.body if isinstance(s, ast.FunctionDef)
                             and s.name == "__init__"), None)
                for st in (ast.walk(init) if init is not None else ()):
                    if isinstance(st, ast.Assign) and len(st.targets) == 1 \
                            and isinstance(st.targets[0], ast.Attribute) \
                            and isinstance(st.targets[0].value, ast.Name) \
                            and st.targets[0].value.id == "self":
                        if _is_re_compile(st.value):
                            m.regexes[f"{node.name}.{st.targets[0].attr}"] = (
                                _resolve(st.value.args[0], m.consts) if st.value.args else None)
                        elif isinstance(st.value, ast.Name) and st.value.id in m.regexes:
                            m.regexes[f"{node.name}.{st.targets[0].attr}"] = m.regexes[st.value.id]
        mods[rel] = m
    return mods


def _words_of(node):
    """The strings of a literal tuple / list / set of words, or of a
    ``frozenset({...})`` / ``set([...])`` / ``tuple(...)`` over one; else
    None."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in ("frozenset", "set", "tuple", "list") \
            and len(node.args) == 1:
        node = node.args[0]
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)) and node.elts \
            and all(isinstance(e, ast.Constant) and isinstance(e.value, str)
                    for e in node.elts):
        return [e.value for e in node.elts]
    return None


# ------------------------------------------------------------- answer state
def answer_state_tokens(fn) -> list:
    """Every answer-state token the function reads, in source order.

    A method NAME is not state: ``self._try_teach_offer(text)`` ends in
    ``_offer`` and reads nothing, so the func of a Call never counts."""
    called = {n.func for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    # A MODE name as a bare string is a VALUE elsewhere ("ringing" is an
    # alarm item's state in timekeeper.py), so for modes only the
    # ``getattr(x, "ringing", None)`` spelling counts as a read.
    getattr_keys = {n.args[1] for n in ast.walk(fn)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == "getattr" and len(n.args) >= 2}
    found = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load) \
                and n not in called and is_answer_state(n.attr):
            found.append("." + n.attr)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and (
                ANSWER_STATE_RX.match(n.value)
                or (n.value in MODE_STATE and n in getattr_keys)):
            found.append(repr(n.value))
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and n.func.attr in ANSWER_STATE_CALLS and not n.args:
            found.append("." + n.func.attr + "()")
    return found


def reads_answer_state(fn) -> Optional[str]:
    """The first answer-state token the function reads, or None."""
    tokens = answer_state_tokens(fn)
    return tokens[0] if tokens else None


# ---------------------------------------------------------------- data flow
def _params(fn) -> list:
    names = [a.arg for a in fn.args.posonlyargs + fn.args.args]
    if names and names[0] in ("self", "cls"):
        names = names[1:]
    return names


def _annotated_str(fn, name) -> bool:
    for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs:
        if a.arg == name and a.annotation is not None:
            return ast.unparse(a.annotation) in ("str", "Optional[str]", "str | None")
    return False


def _word_param(fn, name) -> bool:
    if name in FLAG_PARAMS:
        return False
    return name in WORD_PARAMS or _annotated_str(fn, name)


def _object_only(fn, name) -> bool:
    """Is every use of the parameter an OBJECT use -- an attribute read or
    a non-string method (``c.services``, ``m.group(1)``, ``item.label``)?
    Then it is not his words, whatever the taint would otherwise say."""
    parents = {child: parent for parent in ast.walk(fn)
               for child in ast.iter_child_nodes(parent)}
    uses = [n for n in ast.walk(fn)
            if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)]
    if not uses:
        return False                     # unused: nothing says it is not words
    return all(isinstance(parents.get(u), ast.Attribute)
               and parents[u].value is u
               and parents[u].attr not in _STR_METHODS for u in uses)


def _rung_params(fn) -> dict:
    """EVERY positional parameter of a rung carries his words on entry,
    whatever it is called and however it is annotated -- a rung is where
    the utterance arrives, and the adversary's P4 (09-06) hid it under
    ``utterance`` with no annotation. Skipped: the channel flag, and a
    parameter the body only ever treats as an OBJECT (the commander and
    the match a registry handler receives as ``c`` and ``m``) -- tainting
    those was measured to plant false sites in the send-file handler."""
    return {p: False for p in _params(fn)
            if p not in FLAG_PARAMS
            and (p in WORD_PARAMS or _annotated_str(fn, p) or not _object_only(fn, p))}


def _utterance(fn) -> Optional[str]:
    """The parameter that carries his words: the first word parameter.
    ``source: str`` is a str too, and is never the utterance."""
    for p in _params(fn):
        if _word_param(fn, p):
            return p
    return None


def _word_params(fn) -> dict:
    return {p: False for p in _params(fn) if _word_param(fn, p)}


class _Flow:
    """Taint (his words) and strip state, walked in source order.

    ``classify(call)`` says what a Call does to the words it is handed:
    "strip" (its result is stripped), "sink" (its result is not his words
    at all), or None (it passes them through as they were)."""

    def __init__(self, tainted: dict, classify):
        self.state = dict(tainted)            # name -> stripped?
        self.classify = classify

    def judge(self, expr) -> tuple:
        """(tainted, stripped) for an expression: stripped when every
        tainted name in it is stripped or sits inside a stripper call."""
        found = []

        def walk(n, inside):
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Attribute) and n.func.attr in CAPTURE_METHODS:
                    return                    # a piece a search picked out
                kind = self.classify(n)
                if kind == "sink":
                    return                    # its result is not his words
                inner = inside or kind == "strip"
                for c in ast.iter_child_nodes(n):
                    walk(c, inner)
                return
            if isinstance(n, ast.Name) and n.id in self.state:
                found.append(inside or self.state[n.id])
            for c in ast.iter_child_nodes(n):
                walk(c, inside)
        walk(expr, False)
        return bool(found), bool(found) and all(found)

    def tainted(self, expr) -> bool:
        return self.judge(expr)[0]

    def stripped(self, expr) -> bool:
        return self.judge(expr)[1]

    def assign(self, targets, value):
        tainted, stripped = self.judge(value)
        for t in targets:
            names = [t] if isinstance(t, ast.Name) else (
                [e for e in t.elts if isinstance(e, ast.Name)]
                if isinstance(t, (ast.Tuple, ast.List)) else [])
            for n in names:
                if tainted:
                    self.state[n.id] = stripped
                else:
                    self.state.pop(n.id, None)   # rebound to something clean


def _regex_value(v, mod: _Module, mods: dict, local: dict, cls=None):
    """(label, pattern) for an EXPRESSION that evaluates to a compiled
    regex -- a module regex by name, a loop variable or LOCAL ALIAS over
    one (``rx = _NO_RX``; the adversary's P6), one held on the class or
    the instance (``self._RX``; P5), one imported, or a ``re.compile``
    in place -- else None."""
    if isinstance(v, ast.Name):
        if v.id in local:
            return local[v.id]
        if v.id in mod.regexes:
            return v.id, mod.regexes[v.id]
        if v.id in mod.imports:
            path, name = mod.imports[v.id]
            other = mods.get(path)
            if other and name in other.regexes:
                return v.id, other.regexes[name]
        return None
    if isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name):
        if v.value.id in ("self", "cls") and cls and f"{cls}.{v.attr}" in mod.regexes:
            return f"self.{v.attr}", mod.regexes[f"{cls}.{v.attr}"]
        if v.value.id in mod.imports:
            path, _ = mod.imports[v.value.id]
            other = mods.get(path)
            if other and v.attr in other.regexes:
                return f"{v.value.id}.{v.attr}", other.regexes[v.attr]
        return None
    if _is_re_compile(v):
        pat = _resolve(v.args[0], mod.consts) if v.args else None
        return f"re.compile({(pat or '?')[:32]!r})", pat
    if isinstance(v, ast.IfExp):                 # rx = _A_RX if x else _B_RX
        return _regex_value(v.body, mod, mods, local, cls) \
            or _regex_value(v.orelse, mod, mods, local, cls)
    return None


def _regex_of(node, mod: _Module, mods: dict, local: dict, cls=None):
    """(regex name, pattern) for the object a .match/.search/... is called
    on, or None. ``local`` maps a loop variable or alias to its regex."""
    v = node.func.value
    if isinstance(v, ast.Name) and v.id == "re" and node.args:
        pat = _resolve(node.args[0], mod.consts)
        label = (pat or ast.unparse(node.args[0]))[:32]
        return f"re.{node.func.attr}({label!r})", pat
    return _regex_value(v, mod, mods, local, cls)


def _is_wordset(node, mod: _Module) -> bool:
    return _words_of(node) is not None or (
        isinstance(node, ast.Name) and node.id in mod.wordsets)


def _wordset_label(node, mod: _Module) -> str:
    if isinstance(node, ast.Name):
        return node.id
    words = _words_of(node) or []
    return ",".join(repr(w) for w in words[:2]) + (",…" if len(words) > 2 else "")


def _ordered(fn):
    """Every node in the function body in source order."""
    return sorted((n for n in ast.walk(fn) if hasattr(n, "lineno")),
                  key=lambda n: (n.lineno, n.col_offset))


def _callee(call, mod: _Module, mods: dict, qualname: str):
    """((module path, qualname) candidates, precise) for a call.

    Precise: a module-level function by name, an imported name, a method
    on self, or ``alias.f`` through an import. Otherwise duck-typed: every
    function in the tree with that method name, when there are few enough
    of them for the name to mean something (``session.settle``).
    """
    f = call.func
    cls = qualname.split(".")[0] if "." in qualname else None
    if isinstance(f, ast.Name):
        if f.id in mod.functions:
            return [(mod.path, f.id)], True
        if f.id in mod.imports:
            path, name = mod.imports[f.id]
            other = mods.get(path)
            if other and name and name in other.functions:
                return [(path, name)], True
        return [], True
    if isinstance(f, ast.Attribute):
        if isinstance(f.value, ast.Name) and f.value.id in ("self", "cls") and cls:
            q = f"{cls}.{f.attr}"
            if q in mod.functions:
                return [(mod.path, q)], True
        if isinstance(f.value, ast.Name) and f.value.id in mod.imports:
            path, _ = mod.imports[f.value.id]
            other = mods.get(path)
            if other and f.attr in other.functions:
                return [(path, f.attr)], True
        if f.attr in _NOT_DUCK:
            return [], False
        ducks = [(p, q) for p, m in mods.items() for q in m.functions
                 if q.split(".")[-1] == f.attr]
        return (ducks if 0 < len(ducks) <= DUCK_MAX else []), False
    return [], True


def _walk_flow(fn, mod: _Module, mods: dict, flow: _Flow, on_site=None, on_return=None,
               cls=None):
    """Run the flow over a function in source order. ``on_site(node,
    label, method, arg)`` sees each grammar applied to tainted words -- a
    regex site, a ``startswith`` / ``endswith``, or ``x in (...)``;
    ``on_return`` sees each returned expression."""
    local = {}                    # loop variable or alias -> (label, pattern)
    for n in _ordered(fn):
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name):
            if isinstance(n.iter, ast.Name) and n.iter.id in mod.collections:
                pats = mod.collections[n.iter.id]
                local[n.target.id] = (f"{n.iter.id}[*]", pats[0] if pats else None)
            elif isinstance(n.iter, (ast.Tuple, ast.List)):
                rxs = [r for r in (_regex_value(e, mod, mods, local, cls)
                                   for e in n.iter.elts) if r]
                if rxs:
                    local[n.target.id] = ("(" + "|".join(r[0] for r in rxs) + ")[*]", rxs[0][1])
        elif isinstance(n, ast.Assign):
            flow.assign(n.targets, n.value)
            if len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                rx = _regex_value(n.value, mod, mods, local, cls)
                if rx:
                    local[n.targets[0].id] = rx
                else:
                    local.pop(n.targets[0].id, None)
        elif isinstance(n, (ast.AnnAssign, ast.AugAssign)) and n.value is not None:
            flow.assign([n.target], n.value)
        elif isinstance(n, ast.NamedExpr):
            flow.assign([n.target], n.value)
        elif isinstance(n, ast.Return) and n.value is not None and on_return:
            on_return(n)
        if not on_site:
            continue
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            if n.func.attr in _SITE_METHODS:
                rx = _regex_of(n, mod, mods, local, cls)
                if rx is None:
                    continue
                name, pat = rx
                idx = 1 if n.func.attr in ("sub", "subn") else 0
                if name.startswith("re."):
                    idx += 1                       # re.match(pattern, string)
                if len(n.args) <= idx or not flow.tainted(n.args[idx]):
                    continue
                if n.func.attr not in ("match", "fullmatch") and not anchored(pat):
                    continue
                on_site(n, name, n.func.attr, n.args[idx])
            elif n.func.attr in _EDGE_METHODS and n.args and flow.tainted(n.func.value):
                label = f"{n.func.attr}({ast.unparse(n.args[0])[:32]})"
                on_site(n, label, n.func.attr, n.func.value)
        elif isinstance(n, ast.Compare) and len(n.ops) == 1 \
                and isinstance(n.ops[0], (ast.In, ast.NotIn)) \
                and _is_wordset(n.comparators[0], mod) and flow.tainted(n.left):
            on_site(n, f"in({_wordset_label(n.comparators[0], mod)})", "in", n.left)


# ------------------------------------------------------------------- census
def census(root, known_strippers: dict = None) -> Census:
    """``known_strippers``: {(module path, qualname): reason} -- the
    pre-existing second filler lists the test names, whose output counts
    as stripped although the derivation cannot prove it."""
    root = pathlib.Path(root)
    mods = _load(root)
    out = Census(root=root)
    out.known = dict(known_strippers or {})

    # -- handlers: what a module-level dispatch table names (the registry)
    for path, m in mods.items():
        for name in m.tabled:
            if name in m.functions:
                out.handlers.add((path, name))

    # -- rungs: anything that reads answer state -- and is not a handler
    for path, m in mods.items():
        for q, fn in m.functions.items():
            if (path, q) in out.handlers:
                continue
            tokens = answer_state_tokens(fn)
            if tokens:
                out.rungs.add((path, q))
                out.why[(path, q)] = tokens[0]
                out.tokens[(path, q)] = frozenset(tokens)

    def classify_for(mod: _Module, qualname: str):
        def classify(call) -> Optional[str]:
            f = call.func
            fname = f.id if isinstance(f, ast.Name) else (
                f.attr if isinstance(f, ast.Attribute) else "")
            if fname in CANONICAL:
                return "strip"
            cands, precise = _callee(call, mod, mods, qualname)
            if not precise or len(cands) != 1:
                return None                    # a guess never launders taint
            if cands[0] in out.known:
                return "strip"
            kind = out.kinds.get(cands[0])
            return kind if kind in ("strip", "sink") else None
        return classify

    # -- derived strippers and sinks: a fixpoint over every word-taking function
    def classify_function(path, q) -> str:
        m, fn = mods[path], mods[path].functions[q]
        flow = _Flow(_word_params(fn), classify_for(m, q))
        seen = []

        def on_return(n):
            seen.append(flow.judge(n.value))
        _walk_flow(fn, m, mods, flow, on_return=on_return)
        tainted = [s for t, s in seen if t]
        if not tainted:
            return "sink"
        return "strip" if all(tainted) else "pass"

    for _ in range(FIXPOINT_ROUNDS):
        changed = False
        for path, m in mods.items():
            for q, fn in m.functions.items():
                if not _word_params(fn):
                    continue
                kind = classify_function(path, q)
                if out.kinds.get((path, q)) != kind:
                    out.kinds[(path, q)] = kind
                    changed = True
        if not changed:
            break

    # -- dispatchers: a callee calling DISPATCHER_RUNGS rungs...
    def rungs_called(path, q) -> int:
        m, fn = mods[path], mods[path].functions[q]
        seen = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Call):
                for c in _callee(n, m, mods, q)[0]:
                    if c in out.rungs and c != (path, q):
                        seen.add(c)
        return len(seen)

    for path, m in mods.items():
        for q in m.functions:
            if rungs_called(path, q) >= DISPATCHER_RUNGS:
                out.dispatchers.add((path, q))

    def own_sites(path, q) -> int:
        """Start-anchored grammars a function applies to its own word
        parameters -- the shape of a command router."""
        m, fn = mods[path], mods[path].functions[q]
        words = set(_word_params(fn))
        if not words:
            return 0
        cls = q.split(".")[0] if "." in q else None
        n_sites = 0
        for n in ast.walk(fn):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                    and n.func.attr in ("match", "fullmatch") and n.args \
                    and _regex_of(n, m, mods, {}, cls) is not None \
                    and {x.id for x in ast.walk(n.args[0]) if isinstance(x, ast.Name)} & words:
                n_sites += 1
        return n_sites

    def handed_to(path, q, derived: bool = False) -> set:
        """The callees a function hands his words to. By default the OWN
        utterance parameter, by name -- the way IN to the command path is
        a function that routes the words unchanged, and a rung that
        re-dispatches the sentence it RESOLVED (_try_day_shift ->
        _handle_inner(meant)) is a rung finishing its job, not a
        dispatcher. With ``derived`` any tainted argument counts: that is
        how the command path is closed, so a router that lowercases first
        (_match_assistant -> _send_file_pieces(t)) is still a router."""
        m, fn = mods[path], mods[path].functions[q]
        found = set()
        if not derived:
            utt = _utterance(fn)
            if utt is None:
                return found
            for n in ast.walk(fn):
                if isinstance(n, ast.Call) and any(
                        isinstance(a, ast.Name) and a.id == utt for a in n.args):
                    found.update(_callee(n, m, mods, q)[0])
            return found
        words = _word_params(fn)
        if not words:
            return found
        flow = _Flow(words, classify_for(m, q))
        for n in _ordered(fn):
            if isinstance(n, ast.Assign):
                flow.assign(n.targets, n.value)
            elif isinstance(n, (ast.AnnAssign, ast.AugAssign)) and n.value is not None:
                flow.assign([n.target], n.value)
            elif isinstance(n, ast.NamedExpr):
                flow.assign([n.target], n.value)
            if isinstance(n, ast.Call) and (
                    any(flow.tainted(a) for a in n.args)
                    or any(flow.tainted(k.value) for k in n.keywords)):
                found.update(_callee(n, m, mods, q)[0])
        return found

    # ...or a non-rung with ROUTER_SITES grammars of its own...
    for path, m in mods.items():
        for q in m.functions:
            if (path, q) not in out.rungs and own_sites(path, q) >= ROUTER_SITES:
                out.dispatchers.add((path, q))

    # ...or anything that hands its own utterance to a dispatcher (the way
    # IN: Commander.handle -> _handle_inner, resolve_uncertain -> _route_text).
    grew = True
    while grew:
        grew = False
        for path, m in mods.items():
            for q in m.functions:
                if (path, q) not in out.dispatchers and \
                        handed_to(path, q) & out.dispatchers:
                    out.dispatchers.add((path, q))
                    grew = True

    # -- the command path: whatever a dispatcher hands the utterance to
    # that is not a rung, closed under the same handing-on.
    frontier = set()
    for d in out.dispatchers:
        frontier |= {c for c in handed_to(*d, derived=True)
                     if c not in out.rungs and c not in out.dispatchers}
    while frontier:
        out.command_path |= frontier
        nxt = set()
        for c in frontier:
            nxt |= {x for x in handed_to(*c, derived=True) if x not in out.rungs
                    and x not in out.dispatchers and x not in out.command_path}
        frontier = nxt

    # A ROUTER on it is not followed; a leaf is. A router hands the words
    # on to another command-path callee that CONSUMES them
    # (_match_assistant -> _send_file_pieces), or applies ROUTER_SITES
    # grammars of its own, or fans them out to ROUTER_FANOUT callees.
    # quiet_kind hands them to nobody: a leaf, followed. Handing the words
    # to a string HELPER -- one derived as a stripper or a passthrough,
    # which gives the words back (_lecture_end -> strip_address) -- is not
    # routing either: under the old rule it was, and the lecture end
    # grammar sat behind that door unwalked (09-06).
    helpers = {c for c in out.command_path if out.kinds.get(c) in ("strip", "pass")}
    for c in out.command_path:
        hands = handed_to(*c, derived=True)
        if own_sites(*c) >= ROUTER_SITES or len(hands) >= ROUTER_FANOUT \
                or (hands & out.command_path) - helpers:
            out.routers.add(c)

    # -- the walk: from every rung, his words into every parser
    seen_sites = {}
    visited = set()

    def visit(path, q, tainted: dict, depth: int, chain: str = ""):
        chain = f"{chain} > {q}" if chain else q
        key = (path, q, tuple(sorted(tainted.items())))
        if key in visited or depth > FOLLOW_DEPTH:
            return
        visited.add(key)
        m, fn = mods[path], mods[path].functions[q]
        cls = q.split(".")[0] if "." in q else None
        flow = _Flow(tainted, classify_for(m, q))
        entry_stripped = bool(tainted) and all(tainted.values())

        def on_site(n, name, method, arg):
            stripped = flow.stripped(arg)
            text = ast.unparse(arg)
            via = ("stripped on entry" if entry_stripped and stripped
                   and not any(s in text for s in CANONICAL)
                   else ("here" if stripped else "raw"))
            site = Site(path, q, name, method, text, stripped, via, n.lineno, chain)
            # One record per site PER RUNG: strip_address is reached raw
            # from the lecture rung (the filed line, by design) and
            # stripped from the correction rung, and a regression on the
            # second must not hide behind the first.
            head = chain.split(" > ")[0]
            prev = seen_sites.get((site.key, site.lineno, head))
            if prev is None or (prev.stripped and not site.stripped):
                seen_sites[(site.key, site.lineno, head)] = site

        _walk_flow(fn, m, mods, flow, on_site=on_site, cls=cls)

        # the same walk again for the calls, with the flow now settled
        flow2 = _Flow(tainted, classify_for(m, q))

        def follow(n):
            cands, precise = _callee(n, m, mods, q)
            for cpath, cq in cands:
                if (cpath, cq) in out.dispatchers or (cpath, cq) == (path, q) \
                        or (cpath, cq) in out.routers:
                    continue
                cfn = mods[cpath].functions[cq]
                params = _params(cfn)
                passed = {}
                for i, a in enumerate(n.args):
                    if i < len(params):
                        t, s = flow2.judge(a)
                        if t:
                            passed[params[i]] = s
                for kw in n.keywords:
                    if kw.arg in params:
                        t, s = flow2.judge(kw.value)
                        if t:
                            passed[kw.arg] = s
                if not passed:
                    continue
                if not precise and not any(_word_param(cfn, p) for p in passed):
                    continue                       # a duck must at least take words
                if (cpath, cq) not in out.rungs:
                    out.parsers.add((cpath, cq))
                visit(cpath, cq, passed, depth + 1, chain)

        loops = {}
        for n in _ordered(fn):
            if isinstance(n, ast.For) and isinstance(n.target, ast.Name) \
                    and isinstance(n.iter, ast.Name) and n.iter.id in m.collections:
                loops[n.target.id] = n.iter.id
            elif isinstance(n, ast.Assign):
                flow2.assign(n.targets, n.value)
            elif isinstance(n, (ast.AnnAssign, ast.AugAssign)) and n.value is not None:
                flow2.assign([n.target], n.value)
            elif isinstance(n, ast.NamedExpr):
                flow2.assign([n.target], n.value)
            if isinstance(n, ast.Call):
                follow(n)

    # Commander.handle reads _pending_send AND is the dispatcher: a rung
    # that is also a dispatcher is walked for nothing -- its own body is
    # the command path.
    for path, q in sorted(out.rungs - out.dispatchers):
        words = _rung_params(mods[path].functions[q])
        if words:
            visit(path, q, words, 0)

    out.sites = sorted(seen_sites.values(), key=lambda s: (s.module, s.lineno, s.regex))
    return out


if __name__ == "__main__":               # numbers on stdout, nothing else
    import sys
    known = {("commander.py", "_send_clean"): "named"}
    print(census(sys.argv[1] if len(sys.argv) > 1 else "jarvis", known).report())
