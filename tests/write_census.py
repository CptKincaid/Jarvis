"""Every call in this lane that is not PROVEN harmless, derived from source.

READ THIS BEFORE YOU ADD AN EIGHTH PATTERN.  Rounds 2 and 3 hand-wrote a
table of dangerous writes and both were wrong.  Round 4 computed one and an
adversary stepped over it three ways.  Round 5 rebuilt it -- per call site,
guard KIND pinned -- and an adversary stepped over the rebuild SEVEN more
ways, every one leaving the row count frozen at 75:

    raw.replace(dest)              a DEAD entry in a spelling-matched
                                   ignore list handing a free pass to any
                                   variable named `raw`.  Path.replace IS
                                   os.replace.  MEASURED destroying his
                                   quarterly.xlsx: 100000 bytes -> 4, with
                                   11 census tests green.
    replace(a, b) bare             ignored; the entry meant dataclasses.
    self._my_stages.discard(p)     ignored by dotted name.
    _rm(p), _rm = os.remove        the alias is not in WRITES.
    getattr(os, "remove")(p)       no dotted name at all -> skipped.
    OPS["rm"](p)                   subscript callee -> skipped.
    (a or b).replace(c)            walks to a BoolOp.
  and structurally: the walk covered TWO modules, so a write moved into a
  third censused as 75 rows.  jarvis/tools/filepick.py was already imported
  AND CALLED by both walked modules and was censused by neither.

THREE CONSECUTIVE ROUNDS BLOCKED ON THE SAME INSTRUMENT IS NOT A FIXER WHO
KEEPS MISSING PATTERNS.  It is a design that cannot win.  Rounds 4 and 5
both asked, of each call: "is this one of the writes I know?" -- and SKIPPED
everything else.  Python has unboundedly many ways to name a callee, so the
skip branch is an unbounded hole and enumerating it is a race you lose one
round at a time.  Every single one of the ten defeats above is a SKIP.

SO THE DEFAULT IS INVERTED.  This census does not classify writes.  It
reports every call it cannot PROVE is harmless.  A call earns silence only
by resolving, THROUGH A REAL BINDING IN THE SOURCE, to something vouched
for.  Everything else is a row that a person has to write a line about.

WHY NOT THE CHOKEPOINT, since that was the question asked.  "Nothing may
touch his folders except one audited function" is the right INSTINCT and it
is what this file now enforces -- but building it as a rewrite of
foldersync.py would not have closed one of the ten defeats.  To check "no
mutating call outside the chokepoint" you must still decide, of every call,
whether it mutates; a chokepoint census that skips what it cannot resolve is
walked past by `raw.replace(dest)` exactly as this one was.  The chokepoint
buys a SHORTER TABLE, not a safer check.  And it would mean rewriting all 75
write sites in 2631 lines of file-moving code that has just survived five
adversarial rounds -- putting the product back at risk to fix the
instrument.  What was broken is the instrument.  So: keep the product, keep
the row-per-call-site table and the pinned guard kinds (both of those earned
their place in round 5), and change the one thing that was actually wrong --
the default, from SKIP to REPORT.

AND ROUND 7 INVERTED IT AGAIN, ONE LEVEL DOWN.  Round 6 fixed the default for
the CALLEE and left the same branch wearing a new sign: `_classify` still
answered "not a row" for `<method>.X` -- an attribute call on a receiver the
resolver cannot type -- whenever X was not one of the sixty words in WRITES.
The fixer named that hole itself ("p.shred()") and argued it was narrow
because os, shutil, subprocess, pathlib and tempfile "are not vouched
wholesale".  That is true only of the MODULE-LEVEL spelling of the import,
and an adversary then destroyed a 100000-byte file of his TWO independent
ways with the row count frozen at 90/90 and the lane suite at its baseline:

    logging.FileHandler(p, "w")     `logging` was vouched WHOLESALE in
                                    HARMLESS_MODULES.  A logger is a thing
                                    that writes to a file; that is what a
                                    logger IS.  No indirection at all.
    import subprocess (LAZY)        the resolver walked imports at module
    subprocess.getoutput("del ..")  level only, so a name imported inside a
                                    function was untypable and "getoutput"
                                    is not a word in WRITES.
    p.hardlink_to / symlink_to      three real pathlib methods that create
    p.lchmod                        or change a name.  No trick needed at
                                    all: this route already lived in that
                                    branch.

Enumerating the METHOD vocabulary is the same losing race round 6 correctly
refused for the callee, run again on the receiver -- the author of the attack
picks the name.  So the method-name free pass is DELETED.  Silence now
requires a RESOLVED, VOUCHED, FULLY-QUALIFIED name.  MEASURED PRICE: 394 more
rows over 65 distinct method names, and two names out of HARMLESS_MODULES
(`logging`, `argparse`) after re-reading that list against the question its
own comment asks.

THE FIVE RULES.  They are structural, not a list of tricks, and between them
they are total over `ast.Call`:

  U  UNRESOLVED.  The callee does not resolve to a name through a binding
     this file can see.  getattr, a subscript, a boolean expression, a
     string-concatenated attribute, operator.methodcaller, a local variable
     holding a callable -- all of these land here.  ROW.
  M  MUTATOR.  The resolved tail is in WRITES, whatever it is attached to.
     `raw.replace`, `self._my_stages.discard`, `p.unlink()` in a
     comprehension, an alias resolving to os.remove.  ROW.
  R  REACH.  The call leaves the walked set: a first-party module that is
     not censused, or a third-party/stdlib module nobody has vouched for.
     ROW.
  T  UNTYPED RECEIVER.  An attribute call whose receiver this file cannot
     follow to a type: a parameter, a local, an injected seam, the result of
     another call, a lazily imported module before round 7's fix.  ROW,
     WHATEVER THE METHOD IS CALLED.  This branch used to be silence when the
     name was not in WRITES, and that silence is what routes A, B and C
     walked through.
  S  SAFE.  It resolves to a BARE def or class in a censused module (its
     body is a scope of its own, walked separately), or to a
     fully-qualified name in PROVEN_SAFE, or to a safe builtin.  Note BARE:
     `<lane>.helper` is safe, `<lane>._Ops.rm` is NOT -- a class ATTRIBUTE
     is not a scope and can hold anything, which is how `_Ops.rm = os.remove`
     stayed silent through the first nine routes of round 7.

There is no per-call-site ignore list any more.  IGNORE_CALLEES is gone: it
was matched by SPELLING, so a dead entry in it was a live hole.  Its honest
replacement is PROVEN_SAFE, which is matched on a name RESOLVED THROUGH THE
MODULE'S REAL IMPORTS -- `raw.replace` cannot resolve to `dataclasses.replace`
because `raw` is a local, so it stays a row and gets a line in KNOWN.

WHAT THIS STILL CANNOT SEE, said out loud rather than discovered next round.
MEASURED, not assumed -- each line below was planted and the result recorded.
The round-6 version of this list led with `p.shred()` and called it narrow;
it was neither narrow nor still theoretical by the time the round ended, and
it is CLOSED now by rule T.  What is left:

  * A SECOND CALL SITE OF A METHOD NAME ALREADY VOUCHED FOR.  `p.shred()` is
    a row and needs a sentence; a 72nd `out.append(x)` is a row that needs
    only a line in UNTYPED_SITES.  So changing what an EXISTING receiver IS
    -- `out = []` becoming `out = EvilThing()` -- keeps the key, the name
    and the row, and the prose about `append` goes quietly false.  This is
    the same blind spot a sentence-per-call-site table would have had; the
    bound on it is unchanged: such an object comes either from a censused
    module, whose class methods are walked as scopes so the write surfaces
    there, or from an uncensused one, whose constructing call is already a
    rule-R row.
  * THE MODULE SET IS STILL A LIST.  Rule R makes a call into an unwalked
    first-party module a row, so a write cannot be hidden by moving house --
    but the census only walks what MODULES names, and a FOURTH module of
    this lane appears as one `?uncensused` row rather than as its own rows.
    tests/test_census_fails_closed.py pins the escape set at exactly two
    names for that reason.
  * the ordinal is positional: swapping two calls of one primitive inside
    one scope keeps the key set (round 5's stated blind spot, unchanged).
    Two lambdas in one scope used to COLLIDE outright; round 7 numbers them
    by line, which rule T is what made visible.
  * this reads source, not runtime.  A module rewritten on disk after import
    is out of scope here and always was.

Run it to print today's table:

    ~/vss_env/bin/python -m tests.write_census
"""
import ast
import builtins
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# EVERY module of this lane that moves a file or talks to his machine.  The
# set must be CLOSED: a call that resolves into a first-party module absent
# from here is a ROW (rule R), so moving a write into a fourth module cannot
# buy silence -- it buys a row saying the census does not walk that file.
# filepick was imported AND called by both of the other two through five
# rounds and censused by neither.
MODULES = ("jarvis/foldersync.py", "jarvis/tools/remote.py",
           "jarvis/tools/filepick.py")

# ---------------------------------------------------------------- the shapes
# A name that can DESTROY or CREATE something.  Matched on the resolved tail,
# so os.replace, Path.replace, raw.replace and a module-level alias for
# os.remove all reduce to one word -- and matched WHATEVER the receiver is,
# which is the whole correction: round 5 let a full dotted spelling buy an
# exemption.  Over-broad on purpose: a name here that turns out to be
# harmless costs one line in KNOWN; a name MISSING here costs a file of his.
WRITES = frozenset({
    # THE ARBITRARY ONE, FIRST.  run_ssh runs a command line of our choosing
    # on his Windows machine, so it is every write there is -- del, rmdir /s,
    # move, a redirect.
    "run_ssh",
    # local
    "replace", "rename", "renames", "link", "symlink", "unlink", "remove",
    "rmdir", "removedirs", "mkdir", "makedirs", "write_text", "write_bytes",
    "open", "fdopen", "truncate", "ftruncate", "move", "copy", "copy2",
    "copyfile", "copytree", "rmtree", "touch", "utime", "write",
    "writelines", "mkstemp", "mkdtemp", "chmod", "chown",
    # process: the seam every remote write actually goes through
    "Popen", "run", "call", "check_call", "check_output", "system",
    "popen", "execv", "execvp", "spawn",
    # arbitrary evaluation is arbitrary writing
    "exec", "eval", "compile", "__import__",
    # remote
    "run_copy", "sftp_rename", "sftp_remove", "sftp_mkdir", "sftp_rmdir",
    "run_sftp", "push", "pull", "send", "claim", "discard", "fetch",
    "stage_open", "stage_close", "remove_landed", "land_beside", "save",
})

# One operation that CREATES a name and refuses if it is taken.  A write
# behind one of these has a window of ZERO, not a small one.
CLAIMS = frozenset({
    "link", "sftp_mkdir", "sftp_rename", "stage_open", "claim",
    "mark_claiming", "land_beside", "_open_stage", "_claim_part",
})

# A question about a name.  A write that follows one of these is the shape
# that has bitten this lane three times.
READS = frozenset({
    "exists", "is_file", "is_dir", "stat", "lstat", "fstat", "listdir",
    "iterdir", "glob", "scandir", "read_text", "read_bytes", "read",
    "listing", "sftp_listing", "list_remote", "parse_sftp_entries",
    "stat_key", "dedupe_name", "claim_candidates", "name_series",
    "_list_sftp", "staged_file_missing", "landed_row", "has", "blocked",
    "load", "access",
})

# Modules whose ENTIRE surface is vouched for: they cannot name a file.  os,
# os.path, pathlib, shutil, subprocess, tempfile, glob and operator are
# deliberately NOT here -- reaching into any of those needs a line in
# PROVEN_SAFE naming the exact function.
#
# TWO NAMES CAME OUT OF THIS LIST IN ROUND 7, both found by re-reading it
# against the question this comment asks rather than against a memory of what
# a module is "for":
#   logging   -- logging.FileHandler(path, mode="w") OPENS A PATH AND
#                TRUNCATES IT.  MEASURED by the fifth adversary destroying a
#                100000-byte file of his, fully qualified, no indirection of
#                any kind, with the census green at 90/90.  A logger is a
#                thing that writes to a file; that is what a logger IS.  The
#                cost of taking it out is that every `log.warning(...)` in
#                the lane is a row now -- and those rows are honest, because
#                `log` is bound to the RESULT OF A CALL and this file cannot
#                prove what it is.
#   argparse  -- argparse.FileType("w") is a callable that opens a path for
#                writing.  Same shape, found by the same question.
# Everything left below was re-read one at a time.  The test for it is that
# each survives the sentence "this module has no function that takes a name
# and creates, truncates or destroys the thing at it".  json.dump, textwrap,
# traceback.print_exc and contextlib.redirect_stdout all take an ALREADY OPEN
# file object -- the open is the row, and it is somebody else's line.
HARMLESS_MODULES = frozenset({
    "json", "re", "time", "errno", "stat", "shlex", "dataclasses",
    "threading", "itertools", "difflib", "typing", "sys",
    "contextlib", "fcntl", "collections", "math", "textwrap", "string",
    "unicodedata", "hashlib", "base64", "uuid", "warnings", "types",
    "functools", "enum", "abc", "copy", "traceback",
})

# Fully-qualified names from a module that is NOT wholly harmless, each one
# looked at.  RESOLVED THROUGH THE MODULE'S OWN IMPORTS, never matched by
# spelling: `raw.replace` cannot reach `dataclasses.replace` here, because
# `raw` is a local and a local resolves to nothing.  That is the difference
# between this list and the IGNORE_CALLEES it replaces.
PROVEN_SAFE = frozenset({
    # constructors and pure string maths on paths -- name a path, touch none
    "pathlib.Path", "os.path.join", "os.path.basename", "os.path.dirname",
    "os.path.expanduser", "os.path.isabs", "os.path.isfile",
    "os.path.abspath", "os.path.normpath", "os.path.splitext",
    "os.path.exists", "os.path.getsize",
    # read-only interrogation of the process, never of a name of his
    "os.getpid", "os.environ.get", "os.fspath", "os.strerror",
    "os.scandir", "os.access",
    # re.compile builds a pattern.  `compile` is in WRITES because the
    # BUILTIN of that name is the front door to exec; this is the one
    # fully-qualified spelling that is not.
    "re.compile",
    # already-open descriptors: these move bytes we own, they create and
    # destroy no name.  os.write/os.truncate are NOT here: they are in
    # WRITES and every one of them has a line in KNOWN.
    "os.fsync", "os.lseek", "os.close", "os.fstat", "os.read",
    # dataclasses.replace(conf, ...).  The round-5 ignore list spelled this
    # "replace" and "raw.replace"; the first was blind to a bare local and
    # the second was DEAD and let a 100000-byte file be destroyed.
    "dataclasses.replace",
})

SAFE_BUILTINS = frozenset({
    "len", "str", "int", "float", "bool", "list", "dict", "set", "tuple",
    "frozenset", "sorted", "reversed", "max", "min", "sum", "any", "all",
    "zip", "enumerate", "range", "isinstance", "issubclass", "callable",
    "getattr", "hasattr", "repr", "abs", "round", "next", "iter", "print",
    "type", "id", "hash", "format", "bytes", "divmod", "ord", "chr",
    "SystemExit", "ValueError", "OSError", "RuntimeError", "TypeError",
    "KeyError", "StopIteration", "Exception",
})


# ------------------------------------------------------------ the resolver
class _Unresolved(str):
    """A callee this file could not follow to a name.  It is a ROW."""


def _module_bindings(tree: ast.AST) -> dict:
    """Module-level name -> the dotted thing it is bound to.

    Imports, defs, classes, and simple module-level aliases.  A name bound
    to anything ELSE at module level -- a call, a subscript, a conditional
    -- maps to None, which means UNRESOLVED, which means every call through
    it is a row.  That is how `_rm = os.remove` and
    `_rm = getattr(os, "remove")` both stop being free passes.
    """
    out = {}
    pending = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out[a.asname or a.name.split(".")[0]] = (
                    a.name if a.asname else a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            for a in node.names:
                out[a.asname or a.name] = f"{base}.{a.name}" if base else a.name
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            out[node.name] = "<lane>." + node.name
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target])
            for t in targets:
                if isinstance(t, ast.Name):
                    if isinstance(node.value, (ast.Name, ast.Attribute)):
                        pending.append((t.id, node.value))
                    else:
                        out.setdefault(t.id, None)
    # aliases, to a fixpoint: `_rm = os.remove` then `_r2 = _rm`.
    for _ in range(4):
        for name, value in pending:
            got = _resolve(value, out, set(), set(), set())
            out[name] = None if isinstance(got, _Unresolved) else got
    return out


def _scope_imports(node) -> dict:
    """Names THIS scope imports, as ``_module_bindings`` would spell them.

    ROUTE B.  ``_module_bindings`` walked ``ast.Import`` at module level only,
    so ``import subprocess`` written inside a function bound nothing this file
    could see: `subprocess` became an untypable receiver, the method name was
    all rule M had, and ``subprocess.getoutput("del ...")`` was silent.  An
    import is an import wherever it is written, so a lazy one resolves like a
    module-level one now.

    Imports nested in a ``try``/``if``/``with`` inside the scope count -- the
    ``try: import x except ImportError:`` shape is the ordinary spelling of a
    lazy import.  A nested def or class does NOT: that is its own scope and
    gets its own dict, so a closure over an outer function's import stays
    untypable, which is fail-closed and the direction to be wrong in.
    """
    out = {}
    body = getattr(node, "body", [])
    stack = list(body) if isinstance(body, list) else [body]
    while stack:
        child = stack.pop()
        if isinstance(child, SCOPED):
            continue
        if isinstance(child, ast.Import):
            for a in child.names:
                out[a.asname or a.name.split(".")[0]] = (
                    a.name if a.asname else a.name.split(".")[0])
        elif isinstance(child, ast.ImportFrom):
            base = child.module or ""
            for a in child.names:
                out[a.asname or a.name] = (f"{base}.{a.name}" if base
                                           else a.name)
        else:
            stack.extend(ast.iter_child_nodes(child))
    return out


def _lane_names(tree: ast.AST) -> set:
    """Every name defined as a def or a class ANYWHERE in this module.

    A call to one of these is safe *here* because the thing it calls is a
    scope of its own, walked separately -- the write inside it shows up as
    a row there.  Nesting matters: `read_config` defines `get`, `num` and
    `folder` inside itself, and a census that only read module-level
    statements called all fourteen of those unresolved.
    """
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            out.add(node.name)
    return out


def _assigned_in(node) -> set:
    """Names this scope binds itself: assignments, loops, with, except,
    walrus, comprehension targets.

    A locally bound name is OPAQUE even when a def of the same name exists
    in the module -- otherwise ``_rm = os.remove`` inside a function would
    be waved through by a ``def _rm`` somewhere else in the file.  Shadowing
    is the alias attack wearing a hat.
    """
    out = set()
    body = getattr(node, "body", [])
    stack = list(body) if isinstance(body, list) else [body]
    while stack:
        child = stack.pop()
        if isinstance(child, SCOPED):
            continue
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            out.add(child.id)
        elif isinstance(child, ast.ExceptHandler) and child.name:
            out.add(child.name)
        stack.extend(ast.iter_child_nodes(child))
    return out


def _resolve(node: ast.AST, bindings: dict, methods: set, lane: set,
             local: set):
    """The callee as a resolved name, or ``_Unresolved``.

    Returns one of:
      "os.replace"        a dotted name reached through a real import
      "<lane>.helper"     a def or class of this module
      "<method>.unlink"   an attribute on a receiver we cannot type
      "<builtin>.len"     a builtin
      _Unresolved(...)    rule U: nothing above applies
    """
    attrs = []
    while isinstance(node, ast.Attribute):
        attrs.append(node.attr)
        node = node.value
    attrs.reverse()

    if not isinstance(node, ast.Name):
        # The root is a call, a subscript, a boolean expression, a literal.
        # We cannot type the receiver, so the ATTRIBUTE NAME is all we have;
        # with no attribute at all there is nothing to go on: rule U.  This
        # is where getattr(...)(p), OPS["rm"](p), (a or b).replace(c) and
        # operator.methodcaller("unlink")(p) land.
        if attrs:
            return "<method>." + attrs[-1]
        return _Unresolved(type(node).__name__)

    root = node.id
    if root in ("self", "cls"):
        if not attrs:
            return _Unresolved("self")
        if len(attrs) == 1 and attrs[0] in methods:
            return "<lane>." + attrs[0]
        return "<method>." + attrs[-1]

    if root in local:
        # Bound in this very scope.  Whatever the module says about the
        # name, here it is a variable, and a variable is rule U.
        return ("<method>." + attrs[-1]) if attrs else _Unresolved(
            "local-binding:" + root)

    if root in bindings:
        base = bindings[root]
        if base is None:
            # Bound at module level to something we cannot follow -- a call,
            # a comprehension.  `log = get_logger(...)` is this, and so is
            # `_rm = getattr(os, "remove")`.  With an attribute after it the
            # receiver is untypable, which is rule T and a ROW since round 7
            # -- `log.warning` included; called bare, rule U.
            return ("<method>." + attrs[-1]) if attrs else _Unresolved(
                "opaque-alias:" + root)
        if attrs and base.startswith("<lane>."):
            # A DEF OR CLASS OF THIS MODULE WITH AN ATTRIBUTE AFTER IT.
            # `<lane>.helper` is safe because helper's BODY is a scope of its
            # own, walked separately -- but `<lane>._Ops.rm` is not a scope,
            # it is an ATTRIBUTE, and a class attribute can hold anything:
            #
            #     class _Ops:
            #         rm = os.remove
            #     ...
            #     _Ops.rm(p)
            #
            # is a plain dotted name in the source with no getattr, no
            # subscript and no import trick, and joining the attrs blindly
            # made it "<lane>._Ops.rm", which rule S waved through on the
            # `<lane>.` prefix.  Found by planting it in round 7 after the
            # other nine routes were already caught.  The attribute cannot be
            # typed, so it is rule T like any other untypable receiver.
            return "<method>." + attrs[-1]
        return ".".join([base] + attrs) if attrs else base

    if root in lane:
        return ("<method>." + attrs[-1]) if attrs else "<lane>." + root

    if not attrs and hasattr(builtins, root):
        return "<builtin>." + root

    if attrs:
        # A parameter, a comprehension target, an import inside a function.
        # Untypable, so the attribute name decides -- `raw.replace` is here.
        return "<method>." + attrs[-1]

    # A bare name that is not bound anywhere we can see: a parameter holding
    # a callable, an injected seam.  Rule U -- an alias built at run time is
    # exactly this shape, and so is every one of this lane's `run=` seams.
    return _Unresolved("unbound:" + root)


def _classify(resolved, walked: set):
    """(is_row, primitive).  Rules U, S, M, T, R applied in that order."""
    if isinstance(resolved, _Unresolved):
        return True, "?unresolved"                              # rule U

    tail = resolved.rsplit(".", 1)[-1]
    if resolved in PROVEN_SAFE:                                 # rule S
        return False, tail
    if tail in WRITES:                                          # rule M
        return True, tail
    if resolved.startswith("<lane>."):                          # rule S
        return False, tail
    if resolved.startswith("<method>."):                        # rule T
        # ROUND 7, AND IT IS THE WHOLE ROUND.  This branch used to return
        # "not a row".  A call on a receiver this file cannot type was
        # judged BY ITS METHOD NAME: in WRITES it was a row, and otherwise
        # it was silence.  Round 6 named the hole itself ("p.shred()") and
        # called it narrow on the grounds that os, shutil, subprocess,
        # pathlib and tempfile "are not vouched wholesale" -- true only of
        # the MODULE-LEVEL spelling of the import.  The fifth adversary then
        # destroyed a 100000-byte file of his TWO independent ways with the
        # count frozen at 90/90: logging.FileHandler(p, mode="w") through a
        # module vouched wholesale, and subprocess.getoutput("del ...")
        # through a lazy import.  A third route -- hardlink_to, symlink_to,
        # lchmod -- needed no trick at all and lived in this branch already.
        #
        # Enumerating the method vocabulary is the same losing race round 6
        # correctly refused for the CALLEE, run again one level down: Python
        # has unboundedly many method names and the author of the attack
        # picks his.  So the default is inverted here too.  SILENCE NOW
        # REQUIRES A RESOLVED, VOUCHED, FULLY-QUALIFIED NAME.  An untypable
        # receiver is a row, whatever the method is called.
        #
        # MEASURED PRICE: 396 additional rows, 67 distinct method names.
        # That is the price of the design, not an argument against it -- the
        # same trade round 6 took when KNOWN went 75 to 90.  The name is
        # carried in the primitive so the pin fails on a NEW NAME rather
        # than only on a new call site.
        return True, "?untyped:" + tail
    if resolved.startswith("<builtin>."):
        return (False, tail) if tail in SAFE_BUILTINS else (True, "?builtin")
    root = resolved.split(".")[0]
    if root in HARMLESS_MODULES:                                # rule S
        return False, tail
    if any(resolved == w or resolved.startswith(w + ".") for w in walked):
        return False, tail       # rule S: a module this census DOES walk
    if root == "jarvis":                                        # rule R
        return True, "?uncensused"
    return True, "?reach"                                       # rule R


# --------------------------------------------------------------- the walk
SCOPED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _calls_in_scope(node) -> list:
    """Every ``ast.Call`` belonging to THIS scope and no deeper one.

    A scope's own calls are the calls in its BODY.  A nested def is its own
    scope (so a closure's write is not counted twice, which is round 5's
    fix and still right) -- but that def's DECORATOR LIST and its default
    arguments are collected HERE, because they run where the def is
    written, not when it is called.  They are deliberately not collected
    again by the nested scope itself: charging one call site to two scopes
    is the same double vision that made an occurrence count meaningless.

    A comprehension is walked inline.  It has a scope at run time, but a
    write inside one is WRITTEN here and must be READ here.
    """
    out = []
    stack = []
    body = getattr(node, "body", [])
    for stmt in (body if isinstance(body, list) else [body]):
        stack.append(stmt)
    for child in ast.iter_child_nodes(node):
        if isinstance(child, SCOPED):
            stack.extend(getattr(child, "decorator_list", []))
            args = getattr(child, "args", None)
            if args is not None:
                stack.extend([d for d in (list(args.defaults) +
                                          list(args.kw_defaults)) if d])
            # ROUTE D.  A ClassDef's BASES and KEYWORDS run where the class
            # is WRITTEN, in this scope, exactly as its decorators do -- and
            # they were collected by nobody: not here (only decorators and
            # defaults were), and not by the ClassDef's own scope (which gets
            # its body).  `class _C(self._registry.obliterate(p)): pass` had
            # no owner at all.
            stack.extend(getattr(child, "bases", []))
            stack.extend([k.value for k in getattr(child, "keywords", [])])
    while stack:
        child = stack.pop()
        if isinstance(child, SCOPED):
            continue
        if isinstance(child, ast.Call):
            out.append(child)
        stack.extend(ast.iter_child_nodes(child))
    return out


def _scopes(tree: ast.AST):
    """(qualified name, [calls]) for every scope: the module body, every
    class body, every function, every lambda.  Both of the last two run
    code the module body does not."""
    out = [("<module>", _calls_in_scope(tree), tree)]

    def walk(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                out.append((f"{prefix}{child.name}.<class body>",
                            _calls_in_scope(child), child))
                walk(child, f"{prefix}{child.name}.")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append((f"{prefix}{child.name}", _calls_in_scope(child),
                            child))
                walk(child, f"{prefix}{child.name}.")
            elif isinstance(child, ast.Lambda):
                out.append((f"{prefix}<lambda>", _calls_in_scope(child),
                            child))
                # ROUTE E.  A lambda used to be appended and not RECURSED
                # into, and `_calls_in_scope` skips every SCOPED child -- so
                # a lambda inside a lambda was walked by nobody and every
                # call in it vanished.  One line, and it was a whole scope.
                walk(child, f"{prefix}<lambda>.")
            else:
                walk(child, prefix)
    walk(tree)

    # TWO LAMBDAS IN ONE FUNCTION SHARE A NAME, and a shared name is a
    # COLLIDED KEY: `Ledger._trim` has two of them, both called
    # "Ledger._trim.<lambda>", so their rows landed on the same
    # (module, scope, primitive, ordinal) and one silently covered the
    # other.  Round 6 could not see this because a lambda in this lane
    # holds no MUTATOR -- rule T is what made its calls into rows and the
    # collision visible.  Disambiguate by line so the second one gets its
    # own key, and number only where there IS a clash, so the ordinary
    # single-lambda name does not churn.
    clashing = {n for n in {e[0] for e in out}
                if sum(1 for e in out if e[0] == n) > 1}
    order = {}
    for name in clashing:
        lines = sorted(getattr(o, "lineno", 0) for n, _c, o in out
                       if n == name)
        order[name] = lines
    ranked = []
    for name, calls, owner in out:
        if name in clashing:
            rank = order[name].index(getattr(owner, "lineno", 0)) + 1
            name = f"{name} #{rank}"
        ranked.append((name, calls, owner))
    return [(n, c, o) for n, c, o in ranked if c]


def _method_names(tree: ast.AST) -> set:
    """Every name defined as a method of a class in this module.  `self.X()`
    is safe only when X is one of these -- its body is a scope of its own.
    A mutator name is still a row: rule M runs before rule S."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.add(item.name)
    return out


def _first_party(paths) -> set:
    """The dotted first-party module names the census walks, so a call into
    a jarvis module that is NOT walked can be told apart from one that is."""
    return {p[:-3].replace("/", ".") for p in paths}


def census(paths=MODULES, sources=None) -> list:
    """One row per call this file cannot prove harmless:
    ``(module, scope, primitive, n, guard)``.

    ``primitive`` is the mutator's name for a rule-M row, and otherwise says
    WHY there is a row at all: ``?unresolved`` (rule U -- the callee could
    not be followed to a name), ``?reach`` / ``?uncensused`` (rule R -- the
    call leaves the walked set), ``?builtin``.  An unresolvable call is not
    skipped any more; it has to be EXPLAINED.

    ``n`` is which occurrence of that primitive this is within that scope,
    from 1 in line order -- a call site, not a function, because round 4
    keyed rows per function and a SECOND write in a listed function was
    invisible (measured: a 100000-byte file of his, 100000 -> 0, all tests
    green).  An ordinal rather than a line number so that unrelated edits
    above do not churn the table into being edited to green.

    ``guard`` is "claim:<what>" when an atomic claim precedes the write in
    that scope, "check:<what>" when only a question does, and "" when
    NOTHING does -- the row that cost this lane two files.

    ``sources`` maps a module name to source TEXT, so a test can census a
    mutated copy without writing to the working tree.
    """
    walked = _first_party(paths)
    found = []
    for rel in paths:
        text = (sources or {}).get(rel)
        if text is None:
            text = (REPO / rel).read_text()
        tree = ast.parse(text)
        bindings = _module_bindings(tree)
        methods = _method_names(tree)
        lane = _lane_names(tree)
        for qual, calls, owner in _scopes(tree):
            local = _assigned_in(owner)
            # ROUTE B: an import written INSIDE this scope binds here too.
            scoped = dict(bindings, **_scope_imports(owner))
            checks, claims, writes = [], [], []
            for node in calls:
                resolved = _resolve(node.func, scoped, methods, lane, local)
                is_row, prim = _classify(resolved, walked)
                line = getattr(node, "lineno", 0)
                if not isinstance(resolved, _Unresolved):
                    tail = resolved.rsplit(".", 1)[-1]
                    if tail in CLAIMS or _excl_open(node, tail):
                        claims.append((line, "excl-open" if tail == "open"
                                       else tail))
                    elif tail in READS:
                        checks.append((line, tail))
                if is_row:
                    writes.append((line, prim))
            seen = {}
            for wline, wtail in sorted(writes):
                seen[wtail] = seen.get(wtail, 0) + 1
                before_claim = sorted({t for ln, t in claims if ln <= wline})
                before_check = sorted({t for ln, t in checks if ln <= wline})
                if before_claim:
                    guard = "claim:" + "+".join(before_claim)
                elif before_check:
                    guard = "check:" + "+".join(before_check)
                else:
                    guard = ""
                found.append((rel, qual, wtail, seen[wtail], guard))
    return sorted(found)


def kind(guard: str) -> str:
    """A guard reduced to the only three answers that matter."""
    if guard.startswith("claim:"):
        return "claim"
    if guard.startswith("check:"):
        return "check"
    return "none"


def _excl_open(node: ast.Call, tail: str) -> bool:
    """``os.open(..., O_CREAT | O_EXCL | ...)`` is a CLAIM, not a check --
    the local half of the same idea as sftp mkdir."""
    if tail != "open":
        return False
    return "O_EXCL" in ast.dump(node)


def table(rows=None, known=None) -> str:
    """The census as the table that used to be hand-written.

    Two halves since round 7, because the prose is split two ways: a named
    write gets its own sentence per CALL SITE, and an untypable receiver
    gets one per METHOD NAME.  Both are printed here so the reader sees one
    table -- the split is in where the sentence is kept, not in what is
    reported.
    """
    from tests.test_write_census import (KNOWN, UNTYPED,  # noqa: PLC0415
                                         UNTYPED_SITES)
    rows = census() if rows is None else rows
    known = KNOWN if known is None else known

    def sentence(m, f, w, n):
        row = known.get((m, f, w, n))
        if row:
            return row[1]
        if w.startswith("?untyped:"):
            name = w.split(":", 1)[1]
            if (m, f, w, n) in UNTYPED_SITES and name in UNTYPED:
                return UNTYPED[name]
            if name in UNTYPED:
                return "*** THE SITE IS NOT PINNED ***"
        return "*** NOT IN THE TABLE ***"

    def where(m, f, w, n):
        return f"{m}:{f}" + (f" #{n}" if n > 1 else "")
    width = max(len(where(m, f, w, n)) for m, f, w, n, _g in rows)
    gwidth = max(len(f"{g} -> {w}") for _m, _f, w, _n, g in rows)
    out = ["WHERE".ljust(width) + "  GUARD -> WRITE".ljust(gwidth + 2) +
           "  WINDOW"]
    out.append("-" * (width + gwidth + 40))
    for mod, fn, write, n, guard in rows:
        note = sentence(mod, fn, write, n)
        out.append(where(mod, fn, write, n).ljust(width) +
                   f"  {guard or 'NOTHING'} -> {write}".ljust(gwidth + 2) +
                   f"  {note}")
    return "\n".join(out)


if __name__ == "__main__":                # pragma: no cover - a printout
    import sys
    sys.path.insert(0, str(REPO))
    print(table())
    print(f"\n{len(census())} calls in "
          f"{', '.join(MODULES)} that are not proven harmless, derived by "
          f"AST from the source at "
          f"{os.environ.get('GIT_COMMIT', 'this working tree')}.")
