"""Screen Q&A with the LOCAL vision model ("what's on my screen?").

One tool, ``screen_qa(question)``: grab the active display, shrink it to
``screen.max_width`` pixels wide, JPEG + base64, and ask Ollama's
``screen.model`` for a short spoken answer, handed back as an authored
``speak=`` line.  WHAT BECOMES OF THAT LINE DEPENDS ON WHAT ELSE THE TURN
OWES, and the unqualified form of this sentence has now been wrong in this
docstring twice, in both directions.  ALONE, the line ends the turn as it
stands and no model round rephrases it -- which is the point, because the
vision call can take 25 s against an 8 s tool-loop budget.  HELD BESIDE A
READ HE IS STILL OWED, it is appended to the render round's reply, and that
round may phrase the screen content in its own words or drop this line
altogether.  That is brain.py's rule, stated there beside ``answer_owed``
("an authored line ends the turn alone, and held beside an owed read it is
appended to the render round's reply"), and driven by
tests/test_reply_coverage.py::test_a_read_that_authors_its_own_line_still_lets_the_rest_run.
The ``if result.speak: break`` loop that would have made the short form true
was REMOVED (brain.py says so where it used to be), so any comment resting
on it is describing a deleted code path.  The active window title
(xdotool) rides along in the prompt as context, because a screenshot of a
terminal says nothing about WHICH terminal.

The model is the CHAT model by default (``screen.model: ""``), because
``OLLAMA_MAX_LOADED_MODELS=1``: any other vision model evicts the resident
chat model and costs ~7 s on the next spoken turn.  gemma4:26b carries a
clip projector and reports the ``vision`` capability, so the eyes are free.
Two live findings from 2026-08-31/09-01 shaped the rest of this module:

* llama3.2-vision:latest is pulled (7.8 GiB) but ollama 0.33.1 cannot load
  ``mllama`` AT ALL -- /api/chat answers 500 "unknown model architecture:
  'mllama'".  A configured model that this build refuses is remembered in
  ``_UNUSABLE`` and the next candidate (the chat model) is used, so the
  answer arrives anyway and the log names the config key to fix.
* gemma4 is a THINKING model, and with ``think`` unset it spent all 200
  num_predict tokens reasoning: ``message.content`` came back EMPTY and the
  tool said "My vision model isn't answering, sir." while the model had in
  fact described the screen perfectly inside ``message.thinking``.  So the
  payload sends ``think: false`` for any model whose capabilities include
  thinking -- and for any model ollama will not describe -- much as brain.py
  does on the chat path, which sends the field unconditionally.

The payload also mirrors the brain's ``keep_alive: -1`` and ``num_ctx``
when it is talking to the chat model: a request with a different num_ctx
makes ollama restart the runner (measured: 9.4 s, and it drops the
brain's keep_alive pin with it), which is the very stall this tool is
supposed to avoid.

Three seams, all module-level so tests replace them: ``_grab_screen(display)``
-> PIL image (PIL.ImageGrab first, then ImageMagick ``import`` when the XCB
grab fails), ``_ask_vision(payload, timeout)`` -> the decoded /api/chat
reply, and ``_ollama_show(model)`` -> the decoded /api/show body (manifest
only -- it never loads a model, so it cannot evict anything).

Privacy: the screenshot lives only in memory; it is written to disk ONLY
with JARVIS_DEBUG_SCREEN=1 (``~/.cache/jarvis/screen_last.jpg``, 0600),
and the image bytes are never logged — sizes and timings only.  Nothing
else reaches disk: the ImageMagick fallback is asked for ``png:-`` and the
PNG comes back on stdout, so there is no temp file to clean up and no
window for one to leak from.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from jarvis.logs import get_logger
from jarvis.tools.location import cache_dir, cfg_get
from jarvis.tools.registry import ToolResult, ToolSpec

log = get_logger("tools.screen")

OLLAMA_URL = "http://localhost:11434"
# "" = "whatever model the brain keeps resident"; see chat_model().  Naming a
# second model here is legal but costs the chat model its slot.
DEFAULT_MODEL = ""
FALLBACK_CHAT_MODEL = "gemma4:26b"   # only if jarvis.brain will not import
DEFAULT_MAX_WIDTH = 1280
DEFAULT_DISPLAY = ":1"             # the Spark's desktop; DISPLAY env wins
DEFAULT_QUESTION = "what's on my screen"
VISION_TIMEOUT_S = 25.0            # a cold projector load is ~9 s
SHOW_TIMEOUT_S = 5.0               # /api/show reads a manifest; it is instant
GRAB_TIMEOUT_S = 8.0               # the ImageMagick CLI grab (the only one)
WINDOW_TIMEOUT_S = 2.0             # xdotool
JPEG_QUALITY = 80
MAX_SENTENCES = 3
ANSWER_WORD_CAP = 80               # a rambling vision model, trimmed
MIN_WIDTH, MAX_WIDTH = 320, 4096   # sane bounds for screen.max_width
DEBUG_ENV = "JARVIS_DEBUG_SCREEN"
DEBUG_FILE = "screen_last.jpg"

NO_SCREEN_LINE = "I couldn't get a look at the screen, sir."
# The generic line is for a model that is present and simply did not answer
# (ollama down, a timeout).  A SETUP failure gets a line that names the model
# and the reason -- "My vision model isn't answering, sir." sent Hunter
# looking at the wrong thing twice on 2026-08-31.
NO_VISION_LINE = "My vision model isn't answering, sir."
NOT_INSTALLED_LINE = "I can't see, sir. The vision model {model} isn't installed."
NOT_VISION_LINE = "I can't see, sir. {model} isn't a vision model."
CANNOT_LOAD_LINE = "I can't see, sir. This Ollama build can't load {model}."
SETUP_HINT = ("set screen.model in ~/.config/jarvis/assistant.json to a "
              "vision-capable model this ollama can load (\"\" = the chat model)")
SYSTEM_PROMPT = (
    "You are the eyes of JARVIS, a voice assistant. You are shown a "
    "screenshot of the user's desktop. Answer the question about it in "
    "plain spoken prose: at most three short sentences, no lists, no "
    "markdown, no preamble, no mention of the word screenshot. Read any "
    "text that matters to the question exactly as written.")

_MARKDOWN = re.compile(r"(^|\n)\s*([-*•]\s+|\d+[.)]\s+|#{1,6}\s+)|\*\*|`|__", re.M)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class ScreenUnavailable(RuntimeError):
    """No display, or every screenshot method failed."""


class VisionUnavailable(RuntimeError):
    """Ollama unreachable, timed out, or the vision model errored."""


# ------------------------------------------------------------- display
def display_name() -> str:
    return os.environ.get("DISPLAY") or DEFAULT_DISPLAY


def _display_env(display: str) -> dict:
    return {**os.environ, "DISPLAY": display}


def _grab_pil(display: str):
    from PIL import ImageGrab
    # xdisplay= goes straight to the XCB grabber, so a service started
    # without DISPLAY in its environment still reaches the desktop on :1.
    return ImageGrab.grab(xdisplay=display)


def _grab_cli(display: str):
    """ImageMagick ``import`` on the X root window (``png:-``, straight to
    stdout -- nothing is written to disk); None when ``import`` is missing
    or the grab failed.  There is deliberately no gnome-screenshot path;
    the comment below says why."""
    from PIL import Image
    env = _display_env(display)
    # No gnome-screenshot: the project spec forbids it (it drives the Shell's
    # screenshot UI). ImageMagick `import` reads the X root window quietly.
    if shutil.which("import"):
        try:
            proc = subprocess.run(["import", "-display", display, "-window", "root",
                                   "png:-"], capture_output=True,
                                  timeout=GRAB_TIMEOUT_S, env=env)
            if proc.returncode == 0 and proc.stdout:
                with Image.open(io.BytesIO(proc.stdout)) as img:
                    img.load()
                    return img.copy()
        except (OSError, subprocess.SubprocessError) as exc:
            log.debug("import failed: %s", type(exc).__name__)
    return None


def _grab_screen(display: Optional[str] = None):
    """Seam 1: the whole display as a PIL image; raises ScreenUnavailable."""
    display = display or display_name()
    try:
        return _grab_pil(display)
    except Exception as exc:  # noqa: BLE001 - XCB / no display / no PIL
        log.info("screen: PIL grab on %s failed (%s); trying the CLI",
                 display, type(exc).__name__)
    img = _grab_cli(display)
    if img is None:
        raise ScreenUnavailable(f"no screenshot method worked on {display}")
    return img


def _window_title(display: Optional[str] = None) -> str:
    """The active window's title, or "" when xdotool cannot say."""
    display = display or display_name()
    try:
        proc = subprocess.run(["xdotool", "getactivewindow", "getwindowname"],
                              capture_output=True, text=True, errors="replace",
                              timeout=WINDOW_TIMEOUT_S, env=_display_env(display))
    except Exception:  # noqa: BLE001 - a title is optional (a Latin-1 WM_NAME raised)
        return ""
    if proc.returncode != 0:
        return ""
    return " ".join(proc.stdout.split())[:120]


# --------------------------------------------------------------- image
def coerce_width(value) -> int:
    try:
        width = int(float(str(value)))
    except (TypeError, ValueError):
        return DEFAULT_MAX_WIDTH
    return max(MIN_WIDTH, min(MAX_WIDTH, width))


def downscale(img, max_width: int = DEFAULT_MAX_WIDTH):
    """RGB, at most ``max_width`` wide, aspect kept.  Never upscales."""
    from PIL import Image
    if img.mode != "RGB":
        img = img.convert("RGB")
    width, height = img.size
    if width <= max_width:
        return img
    new_h = max(1, round(height * max_width / width))
    return img.resize((max_width, new_h), Image.LANCZOS)


def encode_jpeg(img, quality: int = JPEG_QUALITY) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def _debug_dump(jpeg: bytes) -> Optional[Path]:
    """JARVIS_DEBUG_SCREEN=1 keeps the last capture; otherwise no disk."""
    if os.environ.get(DEBUG_ENV) != "1":
        return None
    try:
        path = cache_dir() / DEBUG_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(jpeg)
        os.chmod(path, 0o600)
        return path
    except OSError as exc:
        log.debug("screen: debug dump failed: %s", exc)
        return None


def capture(display: Optional[str] = None, max_width: int = DEFAULT_MAX_WIDTH) -> tuple:
    """(base64 jpeg, (orig_w, orig_h), (w, h)); raises ScreenUnavailable."""
    img = _grab_screen(display)
    orig = tuple(img.size)
    small = downscale(img, max_width)
    jpeg = encode_jpeg(small)
    _debug_dump(jpeg)
    return base64.b64encode(jpeg).decode("ascii"), orig, tuple(small.size)


# --------------------------------------------------------------- model
def normalise_model(name) -> str:
    """``gemma4`` and ``gemma4:latest`` are the same model to ollama."""
    name = str(name or "").strip()
    return name if (not name or ":" in name) else f"{name}:latest"


def same_model(a, b) -> bool:
    return bool(a) and normalise_model(a) == normalise_model(b)


def chat_model() -> str:
    """The model the brain keeps resident.  Imported lazily: tools/ is built
    before brain in some entry points, and a missing brain must not cost the
    tool its default."""
    try:
        from jarvis.brain import OLLAMA_MODEL
        return str(OLLAMA_MODEL or "") or FALLBACK_CHAT_MODEL
    except Exception:                            # noqa: BLE001 - import order
        return FALLBACK_CHAT_MODEL


def chat_num_ctx() -> Optional[int]:
    """The brain's num_ctx.  Sending a DIFFERENT one restarts the runner --
    measured 2026-09-01: a screen question with ollama's default 262144
    reloaded gemma4 (9.4 s) and dropped its keep_alive pin to five minutes."""
    try:
        from jarvis.brain import NUM_CTX
        return int(NUM_CTX)
    except Exception:                            # noqa: BLE001 - import order
        return None


def _ollama_show(model: str, timeout: float = SHOW_TIMEOUT_S) -> dict:
    """Seam 3: POST /api/show -> decoded JSON.  Manifest only: this never
    loads a model, so it cannot evict the chat model."""
    data = json.dumps({"model": model}).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/show", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body or b"{}")


# Both caches expire. A verdict that outlived the process would be right
# for the wrong reason: `ollama pull`, an ollama upgrade, or a restarted
# server all change the answer, and Jarvis can run for days.
CACHE_TTL_S = 600.0

_CAPS_CACHE: dict = {}          # model -> ((capabilities|None, reason), when)


def _cached(store: dict, key: str):
    hit = store.get(key)
    if hit is None or time.monotonic() - hit[1] > CACHE_TTL_S:
        store.pop(key, None)
        return None
    return hit[0]


def model_caps(model: str):
    """``(frozenset(capabilities), "")`` or ``(None, reason)`` when ollama
    could not say -- a model that is not pulled answers 404 with
    ``model "x" not found``, which is the difference between "install it"
    and "it is broken".  Cached: this rides on every spoken screen question.
    """
    key = normalise_model(model)
    hit = _cached(_CAPS_CACHE, key)
    if hit is not None:
        return hit
    try:
        info = _ollama_show(model)
        caps = info.get("capabilities") if isinstance(info, dict) else None
        result = (frozenset(str(c) for c in caps), "") if isinstance(caps, list) \
            else (None, "no capabilities in /api/show")
    except urllib.error.HTTPError as exc:
        result = (None, f"HTTP {exc.code}: {http_error_detail(exc)}")
    except Exception as exc:                     # noqa: BLE001 - ollama down
        result = (None, type(exc).__name__)
    _CAPS_CACHE[key] = (result, time.monotonic())
    return result


_UNUSABLE: dict = {}            # model -> (why, when)


def mark_unusable(model: str, why: str) -> None:
    _UNUSABLE[normalise_model(model)] = (why, time.monotonic())


def unusable_reason(model: str) -> str:
    return _cached(_UNUSABLE, normalise_model(model)) or ""


def forget_models() -> None:
    """Drop both verdict caches (tests, and anything that re-pulls a model)."""
    _CAPS_CACHE.clear()
    _UNUSABLE.clear()


def candidate_models(cfg) -> list:
    """The configured model first, then the chat model as the fallback.

    The fallback is not politeness: llama3.2-vision:latest is what
    assistant.json still says on this box and ollama 0.33.1 cannot load it,
    so without a second candidate every "what's on my screen?" is a spoken
    apology until someone hand-edits the config and restarts."""
    configured = str(cfg_get(cfg, "screen.model", DEFAULT_MODEL) or "").strip()
    chat = chat_model()
    out: list = []
    for name in (configured or chat, chat):
        if name and not any(same_model(name, seen) for seen in out):
            out.append(name)
    return out


def setup_line(model: str, reason: str) -> str:
    """The spoken excuse.  A SETUP failure names the model and what is wrong
    with it; everything else falls through to NO_VISION_LINE, which names
    neither -- see the comment on that constant."""
    spoken = normalise_model(model).replace(":latest", "")
    low = (reason or "").lower()
    if "not found" in low or "404" in low:
        return NOT_INSTALLED_LINE.format(model=spoken)
    if "not a vision model" in low:
        return NOT_VISION_LINE.format(model=spoken)
    if low.startswith("http"):
        return CANNOT_LOAD_LINE.format(model=spoken)
    return NO_VISION_LINE


# -------------------------------------------------------------- vision
def _guard_payload(payload: dict) -> int:
    """brain.fit_material over the payload's messages, in place; returns
    the calibrated estimate (0 when the brain is unavailable)."""
    try:
        from jarvis import brain
        _, estimate = brain.fit_material(
            payload.get("messages") or [], label="screen",
            num_predict=(payload.get("options") or {}).get("num_predict"))
        return int(estimate)
    except Exception:                            # noqa: BLE001 - never block
        log.debug("screen: window guard unavailable", exc_info=True)
        return 0


def _log_reply_tokens(reply: dict, estimate: int, payload=None) -> None:
    """The ctx: line for the vision request. The image count rides along
    so the round is logged but NOT folded into the brain's calibration:
    the round-3 review measured one screen question (image costed at zero
    then, and Ollama's count including whatever it charges for the image)
    pinning the process-wide factor at its 2.0 clamp."""
    try:
        from jarvis import brain
        images = brain.image_count((payload or {}).get("messages"))
        brain._log_round_tokens(reply, estimate, label="screen",
                                images=images)
    except Exception:                            # noqa: BLE001 - log only
        log.debug("screen: ctx log unavailable", exc_info=True)


def _ask_vision(payload: dict, timeout: float = VISION_TIMEOUT_S) -> dict:
    """Seam 2: POST /api/chat -> decoded JSON.  Raises on transport errors,
    timeouts and non-JSON bodies."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body or b"{}")


def vision_payload(model: str, b64: str, question: str, title: str,
                   suppress_thinking: bool = True, resident: bool = False,
                   num_ctx: Optional[int] = None) -> dict:
    """The /api/chat body.

    ``suppress_thinking`` sends ``think: false``.  It is not cosmetic: with
    it unset gemma4 spent all 200 num_predict tokens in ``message.thinking``
    and returned an EMPTY ``content`` (measured 2026-09-01: done_reason
    "length", eval_count 200, a perfect description of the desktop stuck in
    the reasoning block) -- which surfaced as "My vision model isn't
    answering, sir."  ``ask_screen`` sends it for a model whose capabilities
    include thinking AND for one whose capabilities ollama would not report;
    it is withheld only from a model known NOT to think, since a build that
    rejects the field on a plain model would turn a working model into a 400.

    ``resident`` means "this IS the brain's model": keep_alive -1 and the
    brain's num_ctx keep the SAME runner, so the screen question costs no
    reload and does not silently drop the brain's pin.  For any other model
    keep_alive 0 unloads it the moment it has answered, so the chat model
    can come straight back (OLLAMA_MAX_LOADED_MODELS=1)."""
    user = f"Question: {question}"
    if title:
        user = f"Active window: {title}\n{user}"
    payload = {"model": model, "stream": False,
               "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user, "images": [b64]}],
               # Low temperature: reading text off a screen wants no invention.
               # num_predict bounds the wait; the answer is word-capped anyway.
               "options": {"temperature": 0.2, "num_predict": 200},
               "keep_alive": -1 if resident else 0}
    if suppress_thinking:
        payload["think"] = False
    if resident and num_ctx:
        payload["options"]["num_ctx"] = int(num_ctx)
    return payload


def tidy_answer(text, cap: int = ANSWER_WORD_CAP) -> str:
    """Plain prose, whitespace collapsed, at most ``cap`` words, cut at the
    last sentence end that fits.  The only caller is ``ask_screen``, whose
    answer becomes screen_qa's ``speak=`` line.  It is spoken AS IT STANDS
    only when the turn owes nothing else; held beside an owed read it goes
    into the render round with the rest, where the model may re-word it or
    leave it out.  See the module docstring -- and do not shorten either
    sentence back to an unqualified one."""
    text = _MARKDOWN.sub(lambda m: m.group(1) or " ", str(text or ""))
    text = " ".join(text.split()).strip()
    if not text:
        return ""
    if len(text.split()) <= cap:
        return text
    kept, used = [], 0
    for sent in _SENTENCE_END.split(text):
        n = len(sent.split())
        if kept and used + n > cap:
            break
        kept.append(sent)
        used += n
        if used >= cap:
            break
    out = " ".join(kept)
    words = out.split()
    if len(words) > cap:
        out = " ".join(words[:cap]).rstrip(",;:") + "."
    return out


def http_error_detail(exc) -> str:
    """Ollama's own words out of an HTTPError body, else the HTTP reason.

    ``{"error": "... unknown model architecture: 'mllama'"}`` is the entire
    difference between "the box is broken" and "pull a model this build can
    load", and it is only ever in the body."""
    try:
        body = exc.read()
    except Exception:                            # noqa: BLE001 - body gone
        body = b""
    try:
        parsed = json.loads(body or b"{}")
        detail = parsed.get("error") if isinstance(parsed, dict) else ""
    except Exception:                            # noqa: BLE001 - not JSON
        detail = ""
    if not detail:
        detail = (body or b"").decode("utf-8", "replace")
    detail = " ".join(str(detail or getattr(exc, "reason", "") or "").split())
    return detail[:160]


def ask_screen(question: str, model: str, b64: str, title: str,
               timeout: float = VISION_TIMEOUT_S,
               caps: Optional[frozenset] = None) -> str:
    """The vision model's answer; raises VisionUnavailable on any failure."""
    # caps unknown (ollama would not say) -> still send think:false: a
    # thinking model that eats its whole budget reasoning is the failure we
    # have actually seen, and every model here that takes the field is one.
    payload = vision_payload(model, b64, question, title,
                             suppress_thinking=caps is None or "thinking" in caps,
                             resident=same_model(model, chat_model()),
                             num_ctx=chat_num_ctx())
    # The same window guard and the same ctx: log line as every request
    # brain.py makes: fit_material() trims the tail of the user text (the
    # question -- the image is costed as a fixed allowance, never as its
    # base64, and never at zero: measured 2026-09-04, the walk put this
    # prompt at 1122 tokens and the guard at 116 until fit_material was
    # taught to look at the last message's images) before the post,
    # _log_round_tokens() records what Ollama counted after it, tagged
    # [screen] and left OUT of the calibration factor (an image round
    # says nothing about the text rate). Tolerant of a missing brain the
    # way chat_model() is; a tiny prompt is the norm here, and the point
    # is that "every request" in the docs is true, not that this one is
    # at risk.
    estimate = _guard_payload(payload)
    try:
        reply = _ask_vision(payload, timeout=timeout)
    except urllib.error.HTTPError as exc:
        # HTTPError is a URLError subclass, so this clause MUST come first.
        # LIVE 2026-08-31 21:22 and 21:27: "What's on my screen?" twice, and
        # both times the whole record of it was ``screen:
        # llama3.2-vision:latest did not answer: HTTPError`` -- the class
        # name, nothing else -- while Hunter heard "My vision model isn't
        # answering, sir." Ollama had returned 500 with "unknown model
        # architecture: 'mllama'": the model is pulled, this build's
        # llama-server simply cannot load it. That is a one-line fix to
        # screen.model, and it was invisible.
        raise VisionUnavailable(f"HTTP {exc.code}: {http_error_detail(exc)}") from exc
    except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
        # socket.timeout is an OSError; a proxy's HTML page is the ValueError
        raise VisionUnavailable(type(exc).__name__) from exc
    if not isinstance(reply, dict):
        raise VisionUnavailable("reply is not an object")
    _log_reply_tokens(reply, estimate, payload)
    if reply.get("error"):
        raise VisionUnavailable(str(reply["error"])[:80])
    message = reply.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    answer = tidy_answer(content)
    if not answer:
        # Name the shape of the emptiness. An answer that is all reasoning
        # means think:false did not take (an old ollama, or a model that
        # ignores the field), and that is a one-word difference in the log
        # between "the model is mute" and "the model is thinking at me".
        thinking = isinstance(message, dict) and message.get("thinking")
        raise VisionUnavailable("thinking-only reply (think:false ignored)"
                                if thinking else "empty answer")
    return answer


def look(question: str, b64: str, title: str, cfg) -> tuple:
    """Ask the first candidate model that can actually answer.

    Returns ``(answer, model)``; raises VisionUnavailable carrying the last
    reason when none could.  A model that ollama REFUSES (404, or the 500
    "unknown model architecture: \'mllama\'" that llama3.2-vision gives on
    this build) is remembered in _UNUSABLE, so the wasted attempt is paid
    once per CACHE_TTL_S -- the verdict EXPIRES, deliberately (see the
    comment on that constant), so a model fixed by an ``ollama pull`` is
    tried again -- and every question until then goes straight to the
    fallback.
    """
    reason = "no vision model configured"
    model = ""
    for model in candidate_models(cfg):
        reason = unusable_reason(model)
        if reason:
            continue
        caps, why = model_caps(model)
        if caps is not None and "vision" not in caps:
            reason = f"{model} is not a vision model (capabilities: " \
                     f"{', '.join(sorted(caps)) or 'none'})"
            mark_unusable(model, reason)
            continue
        if caps is None and ("not found" in why.lower() or "404" in why):
            reason = why
            mark_unusable(model, reason)
            continue
        try:
            return ask_screen(question, model, b64, title, caps=caps), model
        except VisionUnavailable as exc:
            reason = str(exc)
            # An HTTP status is ollama refusing THIS model (it cannot load
            # the architecture, or it is gone); a timeout or a socket error
            # is the server, and trying a second model would only stall
            # Hunter twice.
            if not reason.startswith("HTTP "):
                raise
            mark_unusable(model, reason)
            log.warning("screen: ollama will not load %s (%s); %s", model,
                        reason, SETUP_HINT)
    raise VisionUnavailable(reason or f"{model} unusable")


# --------------------------------------------------------------- tools
def make_tools(cfg, services) -> list[ToolSpec]:
    def screen_qa(question=DEFAULT_QUESTION, **_) -> ToolResult:
        question = " ".join(str(question or "").split()) or DEFAULT_QUESTION
        max_width = coerce_width(cfg_get(cfg, "screen.max_width", DEFAULT_MAX_WIDTH))
        display = display_name()
        t0 = time.monotonic()
        try:
            b64, orig, small = capture(display, max_width)
        except Exception as exc:  # noqa: BLE001 - one spoken excuse, never a raise
            log.warning("screen: capture on %s failed: %s", display,
                        str(exc)[:80] or type(exc).__name__)
            return ToolResult(text="could not capture the screen", ok=False,
                              speak=NO_SCREEN_LINE)
        title = _window_title(display)
        wanted = (candidate_models(cfg) or [""])[0]
        try:
            answer, model = look(question, b64, title, cfg)
        except VisionUnavailable as exc:
            log.warning("screen: %s did not answer: %s; %s", wanted, exc,
                        SETUP_HINT)
            # The reason rides in the tool text too: ok=False + speak= means
            # the spoken line is fixed, so without this the recorded answer
            # to "why can't he see?" is nowhere at all.
            return ToolResult(text=f"vision model {wanted} unavailable: {exc}"
                                   f" -- {SETUP_HINT}",
                              ok=False, speak=setup_line(wanted, str(exc)))
        except Exception as exc:  # noqa: BLE001 - tool boundary
            log.exception("screen: unexpected failure")
            return ToolResult(text=f"vision failed: {str(exc)[:60]}", ok=False,
                              speak=NO_VISION_LINE)
        if not same_model(model, wanted):
            # Loud on EVERY question, not once: the wasted ATTEMPT is paid
            # once per verdict TTL (_UNUSABLE, above), but nothing dedupes
            # this line.  The answer arrived and the config still points at
            # something this ollama cannot use, which is worth saying again.
            log.warning("screen: %s is unusable here, answered with %s "
                        "instead; %s", wanted, model, SETUP_HINT)
        # Sizes and timing only -- never the image, never the answer text.
        log.info("screen: %dx%d -> %dx%d, %d KB, %s answered in %.1fs",
                 orig[0], orig[1], small[0], small[1], len(b64) * 3 // 4096,
                 model, time.monotonic() - t0)
        text = f"Active window: {title}. {answer}" if title else answer
        # speak=answer: WITHIN THE TOOL LOOP nothing rephrases this. The
        # vision call can take 25 s against an 8 s loop budget, so asking
        # for a second model turn to phrase it would be refused and "I have
        # the result but..." spoken instead; the answer was asked for in
        # spoken form, so it is authored here: it goes straight to the speaker.
        # THAT IS AS FAR AS THIS MODULE'S SAY GOES. If the same
        # turn also owes him a read, the render round takes this line with
        # the rest and may re-word it or drop it -- brain.py's rule beside
        # answer_owed, not this module's. Do not restate it as "no model
        # turn ever rephrases it": that has been wrong here twice.
        return ToolResult(text=text, max_sentences=MAX_SENTENCES, speak=answer)

    return [ToolSpec(
        name="screen_qa",
        # <= 20 words: it rides in every prompt (registry word cap).
        description="Look at the user's screen and answer a question about what is showing.",
        parameters={"type": "object", "properties": {
            "question": {"type": "string",
                         "description": "What to answer about the screen "
                                        "(default: describe what is showing)."}}},
        handler=screen_qa)]
