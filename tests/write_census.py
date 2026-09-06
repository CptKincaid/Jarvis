"""Every call in this lane that is not PROVEN harmless, derived from source.

=============================== THE THREAT MODEL ===============================

READ THIS FIRST.  Nine rounds in, this is what the census is for, what it
catches, what it does not and never will, and what the safety of his files
actually rests on.  It is written down because six adversaries in a row beat
this instrument -- and NOT the same way each time.  Rounds 4 through 7
beat it with HONEST shapes (a second write in a function that already had
one, a dead ignore entry, a fully-qualified logging.FileHandler, three
undisguised pathlib methods) and were measuring the right thing; the
census catches each of those shapes today BECAUSE those rounds found
them.  What the defeats had in common was the instrument's DEFAULT --
skip what you cannot resolve -- and that default was the wrong question,
not the rounds.

WHAT IT IS FOR.  One threat, and one only: an HONEST author -- us, next
week, tired -- shipping a read-then-write pair without noticing.

    os.replace(raw, dest) with no claim in front of it
    shutil.rmtree in a new helper
    a second write_text in a function that already has one
    a run_ssh whose command line happens to be `del`

Round 4 lost files of his to exactly those shapes, and since round 6 every
one of them is a ROW here: a line in a table that a person has to write a
sentence about, and a test that fails until they do.  That is the whole
job.  THE CENSUS IS A REGRESSION PIN OVER THE WRITES WE KNOW ABOUT.  It is
not a proof that no write exists, and nothing below should be read as one.

WHAT IT CATCHES.  Each of these was planted and the row recorded:

  * a new call to a name in WRITES, on ANY receiver, spelled through a real
    import or a real alias: os.replace, raw.replace, `_rm = os.remove`,
    p.unlink() inside a comprehension;
  * a second write in a function that already has one -- rows are per
    CALL SITE, not per function;
  * a write at module level, in a class body, in a decorator expression,
    in a lambda, in a lambda inside a lambda, in a class's base list;
  * a write moved into a first-party module the census does not walk
    (rule R: a ?uncensused row, and the walked set itself is pinned);
  * a call reached THROUGH a vouched name into one it merely re-exports --
    remote.subprocess.getoutput, shlex.os.posix_spawn (rule S stops at one
    segment since round 8);
  * a call on a receiver it cannot type -- a parameter, an injected seam,
    the result of another call -- whatever the method is called (rule T);
  * a callee it cannot resolve at all -- getattr, a subscript, a dict of
    operations, operator.methodcaller (rule U);
  * a guard that got weaker: claim -> check -> nothing, on any row;
  * and, from round 9, a class whose ANCESTRY it cannot vouch for.
    Constructing `class _Sub(logging.FileHandler)` runs an inherited
    __init__ the walk never reads, so such a class is a row at every call
    site (?inherits:) unless every base resolves to something proven.

WHAT IT DOES NOT CATCH, AND NEVER WILL.  This is a static walk over source
text.  Python will always have one more way to name a thing at run time,
and an author who WANTS a write to be silent here has these, none of which
is a mistake anyone makes by accident:

  * building the callee at run time -- getattr(os, "re" + "move"), a dict
    of operations, exec of a string.  These are ROWS, but the row says
    ?unresolved, and it is the SENTENCE a person writes for it that
    decides; the census cannot tell `sleep` from `unlink` behind a seam;
  * changing what an existing receiver IS, so that a vouched method name
    -- append, get, warning -- lands on a different object;
  * swapping a class after it is written: a decorator, a metaclass reached
    through a base, __init_subclass__, or a def that RETURNS a writer
    class without a single call in its body.  Round 9 refuses a class
    keyword and an unproven base; it does not, and cannot, read what a
    decorator returns;
  * rewriting the module on disk after import, or monkeypatching os.

Of the routes above the adversaries actually found two -- round 5's
getattr(os, "remove") and OPS["rm"], and round 9's subclass of a writer
-- and each was closed as a rule rather than an entry; there will be
another.  The rest of what rounds 4 through 7 found was not dynamism at
all: honest shapes, measured the right way, and the census catches them
today BECAUSE those rounds did.  That game cannot be won by a static walk, and
the product is not held hostage to it: a bypass that needs a deliberate
subclass of logging.FileHandler is an ATTACK, and the author of this lane
is us.  The dynamism game measures the wrong thing; the rounds that planted
honest mistakes measured the right one.

WHAT HIS FILES ACTUALLY REST ON is not this file.  It is three properties
of the product itself, confirmed by seven adversaries and reopened by none:

  ORDERING   the ledger records which name we are taking, and is fsynced,
             BEFORE the claim is made; the landing is recorded BEFORE the
             original moves to Sent.  A crash at any point re-derives the
             same answer, so a file cannot be sent twice.
  IDENTITY   nothing of his is ever deleted by its NAME.  A note goes only
             if it carries our marker, read through a descriptor first
             (the sub-millisecond window between that read and the unlink
             is stated in _unlink_if_ours, not hidden); a staging folder
             goes only if THIS process took it in THIS run.
  CLAIM      every write that could land where a file already is takes
             one operation that creates the name or refuses -- an sftp
             `rename -l` and an exclusive mkdir on his machine (a property
             HE confirmed of his own box), O_EXCL here -- so the window is
             zero, not small.  Absence is never inferred from a listing.

Those live in jarvis/foldersync.py and jarvis/tools/remote.py and are
pinned by tests/test_foldersync_hand_and_identity.py,
tests/test_foldersync_round5.py, tests/test_write_chokepoint.py and
tests/test_provenance.py.  What THIS file adds is a fence around the calls
those properties are made of, so that an honest edit cannot quietly take
one apart -- a downgraded guard, a write added beside a claimed one, a new
helper that deletes.

TODAY'S TABLE IS 510 ROWS.  502 on the round-9 tip (1039ce8); round 10
added eight and moved none -- seven call sites of the lane's own
destroyers (_replace_ours, _drop, _unlink_after_landing), which are in
WRITES now, and the fdopen over _replace_ours's claimed temp.  Pinned by
a test that DERIVES the number from the two tables in
tests/test_write_census.py rather than typing it; the round-8 report said
490, and that number was stale the day it was written.

============================== HOW IT GOT HERE ==============================

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
  S  SAFE.  It resolves to a BARE def in a censused module (its body is a
     scope of its own, walked separately), or to a BARE class of one whose
     ANCESTRY is proven (round 9: every base resolves to object, an
     exception, a harmless module's own name, PROVEN_SAFE, or another
     proven class -- see _ancestry), or to a fully-qualified name in
     PROVEN_SAFE, or to a safe builtin.  Note BARE: `<lane>.helper` is
     safe, `<lane>._Ops.rm` is NOT -- a class ATTRIBUTE is not a scope and
     can hold anything, which is how `_Ops.rm = os.remove` stayed silent
     through the first nine routes of round 7.  And note PROVEN: a class
     is not its body.  `class _Sub(logging.FileHandler): pass` has an
     empty body and a constructor that truncates; it is a ?inherits: row.

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
    # ROUND 8 WAS ASKED TO ADD getoutput, getstatusoutput AND THE os.exec* /
    # posix_spawn FAMILY HERE, as defence in depth behind the namespace fix
    # below.  REFUSED, and the refusal is measured rather than argued.
    # Round 7 left a test whose whole point is this exact move -- no table
    # the instrument consults may name a trick an adversary used -- and it
    # failed the moment the names went in.  Two things broke, both real:
    #
    #   * `remote.subprocess.getoutput(...)` stopped being a ?through: row
    #     and became a plain `getoutput` row.  The table then SAYS "a known
    #     write" when what it KNOWS is "a call reached through a namespace
    #     this census does not walk".  The second sentence is the true one
    #     and it is the one that generalises.
    #   * a lazy `import subprocess; subprocess.getoutput(...)` stopped
    #     resolving as ?reach -- round 7 pinned that structural answer on
    #     purpose (test_a_lazy_import_resolves_exactly_like_a_module_level_one)
    #     and the vocabulary entry silently took it away.
    #
    # A name here that catches an attack is a name that stops the STRUCTURE
    # being tested against it, and the structure is the only part that
    # holds against the seventeenth spelling.  Three rounds of this lane
    # have now been lost to enumerating vocabulary; the list above stays
    # the list of things this lane's OWN code does, not a list of tricks.
    # arbitrary evaluation is arbitrary writing
    "exec", "eval", "compile", "__import__",
    # remote
    "run_copy", "sftp_rename", "sftp_remove", "sftp_mkdir", "sftp_rmdir",
    "run_sftp", "push", "pull", "send", "claim", "discard", "fetch",
    "stage_open", "stage_close", "remove_landed", "land_beside", "save",
    # THE LANE'S OWN DESTROYERS, round 10.  land_beside has been here since
    # round 5; these three were not, so a NEW caller of self._drop(p) or of
    # _unlink_after_landing(p, ...) on a verify-failed path deleted his
    # 100000-byte Outbox file, and a new _replace_ours(...) at the note
    # name overwrote it to a 300-byte note -- census 92/92 green all three
    # times.  The PRODUCT suite caught each (5/5/1 tests), so this is a
    # tightening, not a hole: every call site is a row with a sentence now,
    # and a new caller fails HERE, before a product test has to.
    "_replace_ours", "_drop", "_unlink_after_landing",
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

# Modules whose OWN surface is vouched for: they cannot name a file.  os,
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
#
# ROUND 8: THAT SENTENCE IS ABOUT THE MODULE'S OWN FUNCTIONS, and until now
# the code asked it of everything the module's NAMESPACE could reach.  No
# module on earth passes that stronger reading, because a module re-exports
# every module it imports as an attribute of itself.  MEASURED: six of the
# names below re-export `os` outright --
#
#     shlex.os        contextlib.os      uuid.os
#     dataclasses.os? no -- dataclasses.inspect.os, one hop further
#     typing.contextlib.os               traceback.linecache.os
#
# -- and remote.py already writes `import shlex` at line 96, so
# `shlex.os.posix_spawn(...)` was one line, no new import, no alias, no
# lazy import, no getattr, and NOT A ROW.  So the match is no longer on the
# ROOT of the resolved name.  It is the root plus EXACTLY ONE segment:
# `json.dumps` is vouched for, `json.codecs.EncodedFile` is a row.  That
# makes the sentence above a checkable claim about this list instead of an
# unmeetable one about the standard library's import graph.
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


def _module_bindings(tree: ast.AST, verdict: dict) -> dict:
    """Module-level name -> the dotted thing it is bound to.

    Imports, defs, classes, and simple module-level aliases.  A name bound
    to anything ELSE at module level -- a call, a subscript, a conditional
    -- maps to None, which means UNRESOLVED, which means every call through
    it is a row.  That is how `_rm = os.remove` and
    `_rm = getattr(os, "remove")` both stop being free passes.

    ``verdict`` is this module's slice of :func:`_ancestry`: class name ->
    "" when the class may be blessed, else why not.  A class with a reason,
    AT ANY DEPTH, is bound here to ``<inherits>.<name>`` rather than to
    ``<lane>.<name>`` -- and bound here, at module level, on purpose: the
    lane set (:func:`_lane_names`) blesses a nested class module-wide, so
    the refusal has to be found first, and ``_resolve`` reads this dict
    before it reads that set.  It is required, not defaulted, for the same
    reason ``_classify`` requires ``exports``: a caller that forgets it gets
    a census that blesses every class, silently.
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
    # ROUND 9.  A class the ancestry check could not vouch for is spelled
    # <inherits>.<name>, whatever else the name is bound to, so that every
    # call site of it -- and every alias to it, below -- is a row.
    for name, why in verdict.items():
        if why:
            out[name] = "<inherits>." + name
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

    TRUE OF A DEF, AND ONLY OF A DEF.  A class's body is walked, but what
    runs when the class is CALLED is its constructor, which may be
    inherited from a module nobody walked.  That is why this set is
    consulted AFTER ``_module_bindings``, where every class with an
    unproven ancestry is already bound to ``<inherits>.<name>``.
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
    # ROUND 10: a PARAMETER is a name this scope binds, and the args live
    # on the def, not in its body.  Without this a parameter named like a
    # def of the module -- `def f(dedupe_name, p): dedupe_name(p)` -- was
    # blessed as <lane>.dedupe_name, which is the shadowing this docstring
    # says is opaque.  Defaults and annotations are Load-context and add
    # nothing; a lambda in a default is SCOPED and skipped like any other.
    args = getattr(node, "args", None)
    if isinstance(args, ast.arguments):
        stack.append(args)
    while stack:
        child = stack.pop()
        if isinstance(child, SCOPED):
            continue
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            out.add(child.id)
        elif isinstance(child, ast.arg):
            out.add(child.arg)
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
        if attrs and base.startswith(("<lane>.", "<inherits>.")):
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


def _classify(resolved, walked: set, exports: dict):
    """(is_row, primitive).  Rules U, S, M, T, R applied in that order.

    ``exports`` maps each walked module's dotted name to what its
    module-level ``def`` and ``class`` STATEMENTS bind: name -> "def",
    "class" (ancestry proven) or "inherits" (ancestry NOT proven; a row at
    every call site).  It is required, not defaulted: a caller that forgets
    it would silently get a different answer from the census, and this file
    already has one test whose whole point is that a hand-rebuilt resolver
    must not drift from the real one.
    """
    if isinstance(resolved, _Unresolved):
        return True, "?unresolved"                              # rule U

    tail = resolved.rsplit(".", 1)[-1]
    if resolved in PROVEN_SAFE:                                 # rule S
        return False, tail
    if tail in WRITES:                                          # rule M
        return True, tail
    if resolved.startswith("<inherits>."):                      # rule S, no
        # ROUND 9.  A class of this module whose ancestry _ancestry could
        # not prove.  Its body was walked and is clean; its CONSTRUCTOR is
        # inherited from somewhere the walk never read.  MEASURED on
        # 0cfad85: `class _Sub(logging.FileHandler): pass` then
        # `_Sub(p, "w")` -- 100000 bytes to 0, census delta ZERO.
        return True, "?inherits:" + tail
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
    # ROUND 8, AND IT IS THE WHOLE ROUND.  The two rules below are the last
    # two that hand out silence, and BOTH of them used to match a PREFIX of
    # the resolved name and then stop looking.  A prefix is not a promise
    # about what is behind it.  Each was written as a statement about a
    # MODULE and read by the code as a statement about every name that
    # module's namespace can reach -- and those are not the same statement,
    # BECAUSE A MODULE RE-EXPORTS EVERY MODULE IT IMPORTS AS AN ATTRIBUTE
    # OF ITSELF.  remote.py says `import subprocess`, so
    #
    #     remote.subprocess.getoutput("rm -f .../quarterly.xlsx")
    #
    # resolved to jarvis.tools.remote.subprocess.getoutput, matched the
    # walked prefix, and was NOT A ROW.  MEASURED on the round-7 tip in a
    # git-archive copy: quarterly.xlsx 100000 bytes -> GONE, census frozen
    # at 483 rows identical to the clean baseline, 42 census tests green.
    # Sixteen further spellings of the one idea were planted against that
    # same instrument and every one was silent; they are all in
    # tests/test_census_reexport.py.
    #
    # THE RULE IS ONE SENTENCE: a vouched namespace vouches for its OWN
    # members, never for what it re-exports.  So both prefixes stop at one
    # segment.
    root = resolved.split(".")[0]
    if root in HARMLESS_MODULES:                                # rule S
        # `json.dumps` yes; `json.codecs.EncodedFile` no.  See the comment
        # on HARMLESS_MODULES: six of the names on that list re-export os,
        # and remote.py already imports one of them.
        if resolved.count(".") == 1:
            return False, tail
        return True, "?through:" + tail
    for mod in sorted(walked):
        if resolved != mod and not resolved.startswith(mod + "."):
            continue
        rest = resolved[len(mod) + 1:]
        # A DEF OR A CLASS OF THAT MODULE, and nothing else.  Not "a name
        # bound there": that reading blesses `remote.subprocess` and
        # `remote.FileHandler` alike, and FileHandler(p, "w") TRUNCATES --
        # it is route A of round 7 handed back through a walked module.
        # Not a module-level variable either: `_RM = os.remove` over there
        # is round 5's alias attack with one module between us and it.
        # A def or a class is the one case where silence is EARNED, because
        # its body is a scope of its own and is censused in its own module;
        # everything else behind that prefix is a namespace this file has
        # not walked and cannot type.
        what = (exports.get(mod, {}).get(rest)
                if rest and "." not in rest else None)
        if what in ("def", "class"):
            return False, tail   # rule S: a real def, or a PROVEN class
        if what == "inherits":
            # ROUND 9, the cross-module half.  It IS a class statement of
            # that module, and rule S used to stop reading there.  The
            # class is a row because its ancestry is not proven; the
            # reason is in _ancestry's verdict for it.
            return True, "?inherits:" + tail
        return True, "?through:" + tail
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


def _exports(tree: ast.AST, verdict: dict) -> dict:
    """What each MODULE-LEVEL ``def`` or ``class`` STATEMENT binds here:
    name -> "def", "class" or "inherits".

    This is what one walked module may reach for in another and still be
    silent -- a def, or a class whose ancestry ``verdict`` (this module's
    slice of :func:`_ancestry`) has proven.  A class it has NOT proven is
    "inherits", which ``_classify`` turns into a row: round 9's adversary
    put `class _Sub(logging.FileHandler): pass` in remote.py and called
    `remote._Sub(p, "w")` from foldersync, and "a class statement of a
    walked module" was the whole of the old test.

    Deliberately NOT ``_lane_names``, which walks to any depth: a def
    nested inside a function is not reachable as ``<module>.<name>`` at
    all, and blessing it would be blessing a name that does not exist.
    Deliberately NOT ``_module_bindings`` either, which is every name bound
    at module level -- that includes ``import subprocess`` and
    ``from logging import FileHandler``, which are the two shapes this
    whole rule exists to stop.
    """
    out = {}
    for n in ast.iter_child_nodes(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[n.name] = "inherits" if verdict.get(n.name) else "def"
        elif isinstance(n, ast.ClassDef):
            out[n.name] = "inherits" if verdict.get(n.name) else "class"
    return out


def _ancestry(trees: dict) -> dict:
    """rel -> {class name -> "" when the class may be blessed, else why not}.

    ROUND 9.  Rule S blessed a def and a class alike, on one sentence: "its
    body is a scope of its own and is censused in its own module".  TRUE
    FOR A DEF.  FALSE FOR A CLASS THAT SUBCLASSES A WRITER: what runs when
    you call ``_Sub(path, "w")`` is the INHERITED ``__init__``, in a module
    the walk never reads.  MEASURED on 0cfad85, git-archive copy, nothing
    of his touched:

        class _Sub(logging.FileHandler): pass     planted in remote.py
        class _Kill(subprocess.Popen): pass       no new import at all
        remote._Sub(p, "w")                       100000 bytes -> 0
        remote._Kill(["rm", "-f", p])             GONE
        the same class, local to foldersync       100000 bytes -> 0
        census                                    502 rows, delta ZERO
        census suite                              63 passed

    So a class earns silence only when EVERYTHING that runs to construct
    it is already vouched for: every base, to the root, and no class
    keyword at all (``metaclass=`` is exactly "who constructs this").  A
    base is proven when it resolves, THROUGH A REAL BINDING, to one of:

      nothing                      an implicit object
      <builtin>.object, or a builtin that is an exception class
      <harmless>.<one segment>     enum.Enum, typing.NamedTuple, abc.ABC
      a name in PROVEN_SAFE        pathlib.Path
      a class STATEMENT of a walked module that is itself proven, by this
      same rule, to a fixpoint -- in this module or across one

    Everything else is a REASON: an unresolved name, a walked class whose
    own base is not proven, a def, an attribute, a module the census has
    not walked, a keyword, a cycle.  A reason makes the class a row at
    every call site, spelled ``?inherits:<name>``, and the reason is the
    sentence somebody has to answer in KNOWN.  This is structural, not a
    list: a base reached through an alias, a mixin two levels up, and a
    walked class inheriting a walked class inheriting a writer are all the
    same one rule and none of them is named anywhere in this file.

    A def stays blessed: its body IS the behaviour.  A DECORATOR IS NOT
    CHECKED, and honestly so -- ``@swap`` where ``def swap(cls): return
    logging.FileHandler`` makes a writer out of an innocent class without
    a single call in its body, and no static walk sees that.  It is the
    dynamism game, named at the top of this file as not played.
    """
    binds = {rel: _module_bindings(tree, {}) for rel, tree in trees.items()}
    mod_of = {rel: rel[:-3].replace("/", ".") for rel in trees}
    rel_of = {m: r for r, m in mod_of.items()}
    groups = {rel: {} for rel in trees}          # rel -> name -> [ClassDef]
    for rel, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                groups[rel].setdefault(node.name, []).append(node)
    top = {rel: {n.name for n in ast.iter_child_nodes(tree)
                 if isinstance(n, ast.ClassDef)}
           for rel, tree in trees.items()}
    verdict = {rel: {} for rel in trees}         # decided names only

    def lineage(rel, name):
        """A class STATEMENT `name` at the top of `rel`: "" proven, a reason,
        or None while undecided."""
        if name not in top[rel]:
            return (f"has a base, {name}, that is not a class statement at "
                    f"the top of {mod_of[rel]}")
        if name not in verdict[rel]:
            return None
        why = verdict[rel][name]
        return f"inherits {name}, which {why}" if why else ""

    def judge_base(rel, base):
        r = _resolve(base, binds[rel], set(), set(), set())
        if isinstance(r, _Unresolved):
            return f"has a base this census cannot resolve ({r})"
        if r.startswith("<lane>."):
            return lineage(rel, r[len("<lane>."):])
        if r.startswith("<method>."):
            return ("has a base reached through an attribute this census "
                    f"cannot type ({r})")
        if r.startswith("<builtin>."):
            obj = getattr(builtins, r[len("<builtin>."):], None)
            if obj is object or (isinstance(obj, type)
                                 and issubclass(obj, BaseException)):
                return ""
            return f"has a builtin base this census does not vouch for ({r})"
        if r in PROVEN_SAFE:
            return ""
        if r.split(".")[0] in HARMLESS_MODULES and r.count(".") == 1:
            return ""
        for mod in sorted(rel_of):
            if r.startswith(mod + "."):
                rest = r[len(mod) + 1:]
                if "." in rest:
                    return f"has a base reached through {mod}'s namespace ({r})"
                return lineage(rel_of[mod], rest)
        return f"has a base this census has not walked ({r})"

    def judge(rel, node):
        """"" proven, a reason, or None while a base is still undecided."""
        for kw in node.keywords:
            return (f"is built with a class keyword ({kw.arg or '**'}=...), "
                    "which runs code of its own at every construction")
        waiting = False
        for base in node.bases:
            got = judge_base(rel, base)
            if got:
                return got
            if got is None:
                waiting = True
        return None if waiting else ""

    pending = {(rel, name) for rel in trees for name in groups[rel]}
    for _ in range(len(pending) + 1):
        moved = False
        for rel, name in sorted(pending):
            results = [judge(rel, n) for n in groups[rel][name]]
            reason = next((r for r in results if r), None)
            if reason is not None:
                verdict[rel][name] = reason           # dirty wins, at once
            elif all(r == "" for r in results):
                verdict[rel][name] = ""
            else:
                continue                              # a base is undecided
            pending.discard((rel, name))
            moved = True
        if not pending or not moved:
            break
    for rel, name in pending:
        verdict[rel][name] = ("is part of a cycle of bases this census "
                              "cannot order")
    return verdict


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
    # EVERY module is parsed before ANY is classified.  Rule S now asks a
    # question about a module OTHER than the one it is reading -- "is this
    # name a def or a class over there?" -- so the answer has to exist
    # before the walk starts.  It is also why ``sources`` maps the whole
    # set: censusing a mutated copy of one module must see the real other
    # two, and it does.
    trees = {}
    for rel in paths:
        text = (sources or {}).get(rel)
        if text is None:
            text = (REPO / rel).read_text()
        trees[rel] = ast.parse(text)
    # ROUND 9: and every class's ANCESTRY is judged across the whole set
    # before any module is classified, for the same reason -- a class in
    # foldersync may inherit a class in remote that inherits a writer.
    verdicts = _ancestry(trees)
    exports = {rel[:-3].replace("/", "."): _exports(t, verdicts[rel])
               for rel, t in trees.items()}
    found = []
    for rel in paths:
        tree = trees[rel]
        bindings = _module_bindings(tree, verdicts[rel])
        methods = _method_names(tree)
        lane = _lane_names(tree)
        for qual, calls, owner in _scopes(tree):
            local = _assigned_in(owner)
            # ROUTE B: an import written INSIDE this scope binds here too.
            scoped = dict(bindings, **_scope_imports(owner))
            checks, claims, writes = [], [], []
            for node in calls:
                resolved = _resolve(node.func, scoped, methods, lane, local)
                is_row, prim = _classify(resolved, walked, exports)
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
