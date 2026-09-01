"""Screen Q&A with the LOCAL vision model ("what's on my screen?").

One tool, ``screen_qa(question)``: grab the active display, shrink it to
``screen.max_width`` pixels wide, JPEG + base64, and ask Ollama's
``screen.model`` (llama3.2-vision by default) for a short spoken answer;
the brain's model turn then phrases the reply.  The active window title
(xdotool) rides along in the prompt as context, because a screenshot of a
terminal says nothing about WHICH terminal.

``screen.model`` must be a model THIS ollama can load, not merely one that
is pulled.  LIVE 2026-08-31: llama3.2-vision:latest is on disk (7.27 GiB)
and ollama 0.33.1 answers /api/chat with 500 "unknown model architecture:
'mllama'" for it.  gemma4:26b ships a projector layer and is already the
resident chat model, so pointing screen.model at it also dodges the
OLLAMA_MAX_LOADED_MODELS=1 eviction a separate vision model causes.

Two seams, both module-level so tests replace them:
``_grab_screen(display)`` -> PIL image (PIL.ImageGrab first, then
``gnome-screenshot -f`` / ImageMagick ``import`` when the XCB grab fails)
and ``_ask_vision(payload, timeout)`` -> the decoded /api/chat reply.

Privacy: the screenshot lives only in memory; it is written to disk ONLY
with JARVIS_DEBUG_SCREEN=1 (``~/.cache/jarvis/screen_last.jpg``, 0600),
and the image bytes are never logged — sizes and timings only.  The
gnome-screenshot fallback has no stdout mode, so it round-trips through a
0700 temp dir that is removed before the function returns.
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
DEFAULT_MODEL = "llama3.2-vision:latest"
DEFAULT_MAX_WIDTH = 1280
DEFAULT_DISPLAY = ":1"             # the Spark's desktop; DISPLAY env wins
DEFAULT_QUESTION = "what's on my screen"
VISION_TIMEOUT_S = 25.0            # a cold llama3.2-vision load is ~10 s
GRAB_TIMEOUT_S = 8.0               # the CLI screenshot fallbacks
WINDOW_TIMEOUT_S = 2.0             # xdotool
JPEG_QUALITY = 80
MAX_SENTENCES = 3
ANSWER_WORD_CAP = 80               # a rambling vision model, trimmed
MIN_WIDTH, MAX_WIDTH = 320, 4096   # sane bounds for screen.max_width
DEBUG_ENV = "JARVIS_DEBUG_SCREEN"
DEBUG_FILE = "screen_last.jpg"

NO_SCREEN_LINE = "I couldn't get a look at the screen, sir."
NO_VISION_LINE = "My vision model isn't answering, sir."
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
    """gnome-screenshot, then ImageMagick ``import``; None when neither is
    installed or both failed."""
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


# -------------------------------------------------------------- vision
def _ask_vision(payload: dict, timeout: float = VISION_TIMEOUT_S) -> dict:
    """Seam 2: POST /api/chat -> decoded JSON.  Raises on transport errors,
    timeouts and non-JSON bodies."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body or b"{}")


def vision_payload(model: str, b64: str, question: str, title: str) -> dict:
    user = f"Question: {question}"
    if title:
        user = f"Active window: {title}\n{user}"
    return {"model": model, "stream": False,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": user, "images": [b64]}],
            # Low temperature: reading text off a screen wants no invention.
            # num_predict bounds the wait; the answer is capped in words anyway.
            "options": {"temperature": 0.2, "num_predict": 200}}


def tidy_answer(text, cap: int = ANSWER_WORD_CAP) -> str:
    """Plain prose, whitespace collapsed, at most ``cap`` words, cut at the
    last sentence end that fits (the brain rephrases what is left)."""
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
               timeout: float = VISION_TIMEOUT_S) -> str:
    """The vision model's answer; raises VisionUnavailable on any failure."""
    payload = vision_payload(model, b64, question, title)
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
    if reply.get("error"):
        raise VisionUnavailable(str(reply["error"])[:80])
    message = reply.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    answer = tidy_answer(content)
    if not answer:
        raise VisionUnavailable("empty answer")
    return answer


# --------------------------------------------------------------- tools
def make_tools(cfg, services) -> list[ToolSpec]:
    def screen_qa(question=DEFAULT_QUESTION, **_) -> ToolResult:
        question = " ".join(str(question or "").split()) or DEFAULT_QUESTION
        model = str(cfg_get(cfg, "screen.model", DEFAULT_MODEL) or DEFAULT_MODEL)
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
        try:
            answer = ask_screen(question, model, b64, title)
        except VisionUnavailable as exc:
            log.warning("screen: %s did not answer: %s", model, exc)
            # The reason rides in the tool text too: ok=False + speak= means
            # the spoken line is fixed, so without this the recorded answer
            # to "why can't he see?" is nowhere at all.
            return ToolResult(text=f"vision model {model} unavailable: {exc}",
                              ok=False, speak=NO_VISION_LINE)
        except Exception as exc:  # noqa: BLE001 - tool boundary
            log.exception("screen: unexpected failure")
            return ToolResult(text=f"vision failed: {str(exc)[:60]}", ok=False,
                              speak=NO_VISION_LINE)
        # Sizes and timing only -- never the image, never the answer text.
        log.info("screen: %dx%d -> %dx%d, %d KB, %s answered in %.1fs",
                 orig[0], orig[1], small[0], small[1], len(b64) * 3 // 4096,
                 model, time.monotonic() - t0)
        text = f"Active window: {title}. {answer}" if title else answer
        # speak=answer: the vision call can take 25 s and the brain's tool
        # loop budget is 8 s, so a second model turn to phrase this would be
        # refused and "I have the result but..." spoken instead. The answer
        # was asked for in spoken form; it goes straight to the speaker.
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
