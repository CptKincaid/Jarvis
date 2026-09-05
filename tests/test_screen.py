"""Screen Q&A tool (jarvis.tools.screen): a generated PIL image through the
``_grab_screen`` seam and a canned /api/chat reply through ``_ask_vision``
-> the payload the vision model sees (downscaled JPEG, window title,
question), the spoken excuses for a missing display / a silent model /
a timeout, the answer word cap, the tool-description word cap, and the
privacy rules (no file on disk unless JARVIS_DEBUG_SCREEN=1, the image
never in the log).

Firewall: no network (urlopen is poisoned), no xdotool / gnome-screenshot
(the subprocess paths are faked), tmp cache dir from conftest.
"""
import base64
import io
import json
import logging
import stat
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from PIL import Image

from jarvis.tools import screen as scr
from jarvis.tools.registry import DESCRIPTION_WORD_CAP, ToolRegistry, ToolResult


# ------------------------------------------------------------ fixtures
@pytest.fixture(autouse=True)
def _firewall(monkeypatch):
    def _no_network(*a, **k):
        raise AssertionError("unit test tried to reach the network")
    monkeypatch.setattr(urllib.request, "urlopen", _no_network)
    monkeypatch.delenv(scr.DEBUG_ENV, raising=False)
    monkeypatch.setenv("DISPLAY", ":9")           # never the live :1
    # model_caps() and the "this model will not load here" verdict are cached
    # for the process; without this a test that poisons gemma4:26b decides
    # the NEXT test's outcome (it did, on the first run of this fix).
    scr.forget_models()
    yield
    scr.forget_models()


def make_image(width=2560, height=1440, mode="RGB"):
    """A tiny synthetic 'desktop': two colour bands so JPEG has something."""
    img = Image.new(mode, (width, height), (30, 60, 120) if mode == "RGB"
                    else (30, 60, 120, 255))
    band = Image.new(mode, (width, height // 4), (220, 220, 220) if mode == "RGB"
                     else (220, 220, 220, 255))
    img.paste(band, (0, 0))
    return img


class FakeVision:
    """Records the payload; answers with ``content`` or raises ``fail``."""

    def __init__(self, content="A terminal running pytest, all green.", fail=None,
                 reply=None):
        self.content, self.fail, self.reply = content, fail, reply
        self.payloads, self.timeouts = [], []

    def __call__(self, payload, timeout=None):
        self.payloads.append(payload)
        self.timeouts.append(timeout)
        if self.fail is not None:
            raise self.fail
        if self.reply is not None:
            return self.reply
        return {"model": payload["model"], "done": True,
                "message": {"role": "assistant", "content": self.content}}


# gemma4:26b's real capabilities, live 2026-09-01 (`ollama show gemma4:26b`).
GEMMA4_CAPS = ["completion", "vision", "tools", "thinking"]


def show(caps=GEMMA4_CAPS, seen=None):
    """A fake /api/show: capabilities per model, recording who was asked."""
    def _show(model, timeout=None):
        if seen is not None:
            seen.append(model)
        by_model = caps if isinstance(caps, dict) else None
        got = by_model.get(model, GEMMA4_CAPS) if by_model else caps
        if isinstance(got, Exception):
            raise got
        return {"capabilities": list(got)}
    return _show


@pytest.fixture
def wired(monkeypatch):
    """All three seams faked; returns (vision, grabbed_displays)."""
    vision = FakeVision()
    grabbed = []

    def grab(display=None):
        grabbed.append(display)
        return make_image()
    monkeypatch.setattr(scr, "_grab_screen", grab)
    monkeypatch.setattr(scr, "_ask_vision", vision)
    monkeypatch.setattr(scr, "_ollama_show", show())
    monkeypatch.setattr(scr, "_window_title", lambda display=None: "hunter@spark: ~/Jarvis")
    return vision, grabbed


def tool(cfg=None):
    reg = ToolRegistry()
    reg.register_many(scr.make_tools(cfg, None))
    return reg


def decode_image(payload):
    images = payload["messages"][-1]["images"]
    assert isinstance(images, list) and len(images) == 1
    raw = base64.b64decode(images[0])
    img = Image.open(io.BytesIO(raw))
    img.load()
    return img, raw


# ------------------------------------------------------------ the spec
def test_tool_spec_within_budget():
    reg = tool()
    assert reg.names() == ["screen_qa"]
    spec = reg.get("screen_qa")
    assert spec.description_words() <= DESCRIPTION_WORD_CAP
    assert reg.budget()["ok"]
    schema = spec.schema()["function"]
    assert schema["parameters"]["properties"]["question"]["type"] == "string"
    # the handler must tolerate the model omitting every argument
    assert callable(spec.handler)


# ------------------------------------------------------------ happy path
def test_screen_qa_sends_downscaled_jpeg_title_and_question(wired):
    vision, grabbed = wired
    res = tool().call("screen_qa", {"question": "which test is failing?"})
    assert isinstance(res, ToolResult) and res.ok
    # spoken directly: the 25 s vision call exceeds the brain's 8 s tool-loop
    # budget, so a second model turn to phrase it would be refused
    assert res.speak == "A terminal running pytest, all green."
    assert res.max_sentences == scr.MAX_SENTENCES
    assert "A terminal running pytest, all green." in res.text
    assert "hunter@spark: ~/Jarvis" in res.text
    assert grabbed == [":9"], "the grab targets the DISPLAY in force"

    payload = vision.payloads[-1]
    # DEFAULT_MODEL is "" -- the eyes ride on the model the brain already has
    # resident, because OLLAMA_MAX_LOADED_MODELS=1 makes a second one an
    # eviction, not an addition.
    assert scr.DEFAULT_MODEL == ""
    assert payload["model"] == scr.chat_model()
    assert payload["stream"] is False
    assert payload["messages"][0]["role"] == "system"
    assert "three short sentences" in payload["messages"][0]["content"]
    user = payload["messages"][-1]
    assert user["role"] == "user"
    assert "which test is failing?" in user["content"]
    assert "hunter@spark: ~/Jarvis" in user["content"]
    assert "images" not in payload["messages"][0], "the image rides on the user turn"
    img, raw = decode_image(payload)
    assert img.format == "JPEG"
    assert img.size == (1280, 720), "2560x1440 -> 1280 wide, aspect kept"
    assert vision.timeouts[-1] == scr.VISION_TIMEOUT_S <= 25
    assert "```" not in res.text and "**" not in res.text


def test_default_question_when_the_model_sends_nothing(wired):
    vision, _ = wired
    res = tool().call("screen_qa", {})
    assert res.ok
    assert scr.DEFAULT_QUESTION in vision.payloads[-1]["messages"][-1]["content"]
    res = tool().call("screen_qa", {"question": "   "})
    assert res.ok
    assert scr.DEFAULT_QUESTION in vision.payloads[-1]["messages"][-1]["content"]


def test_no_window_title_leaves_prompt_and_text_clean(wired, monkeypatch):
    vision, _ = wired
    monkeypatch.setattr(scr, "_window_title", lambda display=None: "")
    res = tool().call("screen_qa", {"question": "what is this"})
    assert res.ok
    assert "Active window" not in res.text
    assert "Active window" not in vision.payloads[-1]["messages"][-1]["content"]


# ------------------------------------------------------------- config
def test_model_and_max_width_come_from_config(wired):
    vision, _ = wired
    cfg = {"screen": {"model": "qwen3-vl:8b", "max_width": 640}}
    res = tool(cfg).call("screen_qa", {"question": "hi"})
    assert res.ok
    assert vision.payloads[-1]["model"] == "qwen3-vl:8b"
    img, _ = decode_image(vision.payloads[-1])
    assert img.size == (640, 360)


def test_config_read_through_assistant_config_get(wired):
    vision, _ = wired

    class Cfg:                       # AssistantConfig-shaped stand-in
        def get(self, dotted, default=None):
            return {"screen.model": "moondream", "screen.max_width": "800"}.get(dotted, default)
    res = tool(Cfg()).call("screen_qa", {"question": "hi"})
    assert res.ok
    assert vision.payloads[-1]["model"] == "moondream"
    img, _ = decode_image(vision.payloads[-1])
    assert img.size == (800, 450)


def test_coerce_width_clamps_and_defaults():
    assert scr.coerce_width(None) == scr.DEFAULT_MAX_WIDTH
    assert scr.coerce_width("wide") == scr.DEFAULT_MAX_WIDTH
    assert scr.coerce_width(10) == scr.MIN_WIDTH
    assert scr.coerce_width(99999) == scr.MAX_WIDTH
    assert scr.coerce_width("1024.0") == 1024


# ---------------------------------------------------------- downscale
def test_downscale_keeps_aspect_never_upscales_and_forces_rgb():
    assert scr.downscale(make_image(3840, 2160), 1280).size == (1280, 720)
    assert scr.downscale(make_image(1920, 1200), 1280).size == (1280, 800)
    small = make_image(800, 600)
    assert scr.downscale(small, 1280) is small, "no resample when already small"
    rgba = scr.downscale(make_image(2000, 1000, mode="RGBA"), 1000)
    assert rgba.mode == "RGB" and rgba.size == (1000, 500)
    assert scr.downscale(make_image(1280, 720), 1280).size == (1280, 720)


def test_capture_returns_base64_jpeg_and_sizes(monkeypatch):
    monkeypatch.setattr(scr, "_grab_screen", lambda display=None: make_image(2560, 1440))
    b64, orig, small = scr.capture(":9", 1280)
    assert orig == (2560, 1440) and small == (1280, 720)
    raw = base64.b64decode(b64)
    assert raw[:3] == b"\xff\xd8\xff", "JPEG magic"
    assert len(raw) < 200_000


# ------------------------------------------------------------ failures
def test_missing_display_speaks_the_screen_excuse(monkeypatch, wired):
    vision, _ = wired

    def no_display(display=None):
        raise scr.ScreenUnavailable("no screenshot method worked on :9")
    monkeypatch.setattr(scr, "_grab_screen", no_display)
    res = tool().call("screen_qa", {"question": "what's up"})
    assert res.ok is False
    assert res.speak == scr.NO_SCREEN_LINE
    assert res.speak.endswith(", sir.")
    assert vision.payloads == [], "no capture -> the vision model is never asked"


def test_grab_screen_without_a_display_raises_after_both_methods(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    seen = []

    def pil_fails(display):
        seen.append(("pil", display))
        raise OSError("cannot open display")

    def cli_fails(display):
        seen.append(("cli", display))
        return None
    monkeypatch.setattr(scr, "_grab_pil", pil_fails)
    monkeypatch.setattr(scr, "_grab_cli", cli_fails)
    with pytest.raises(scr.ScreenUnavailable):
        scr._grab_screen()
    # DISPLAY unset -> the Spark's desktop, and the CLI fallback was tried
    assert seen == [("pil", scr.DEFAULT_DISPLAY), ("cli", scr.DEFAULT_DISPLAY)]


def test_grab_screen_falls_back_to_the_cli_when_pil_fails(monkeypatch):
    def pil_fails(display):
        raise OSError("Pillow was built without XCB support")
    monkeypatch.setattr(scr, "_grab_pil", pil_fails)
    monkeypatch.setattr(scr, "_grab_cli", lambda display: make_image(1000, 500))
    assert scr._grab_screen(":9").size == (1000, 500)


def test_grab_cli_skips_gnome_screenshot_and_uses_import(monkeypatch, tmp_path):
    calls = []

    def fake_which(name):
        return f"/usr/bin/{name}" if name in ("gnome-screenshot", "import") else None

    def fake_run(cmd, capture_output=False, timeout=None, env=None, **_):
        calls.append((cmd[0], timeout, env.get("DISPLAY")))
        assert timeout <= 8, "every subprocess is bounded"
        if cmd[0] == "gnome-screenshot":
            return subprocess.CompletedProcess(cmd, 1, b"", b"no screen")
        buf = io.BytesIO()
        make_image(640, 480).save(buf, format="PNG")
        return subprocess.CompletedProcess(cmd, 0, buf.getvalue(), b"")
    monkeypatch.setattr(scr.shutil, "which", fake_which)
    monkeypatch.setattr(scr.subprocess, "run", fake_run)
    img = scr._grab_cli(":9")
    assert img.size == (640, 480)
    assert [c[0] for c in calls] == ["import"]          # gnome-screenshot is forbidden by spec
    assert all(c[2] == ":9" for c in calls), "the fallbacks are told which display"


def test_the_docstrings_describe_the_grab_path_the_code_actually_takes():
    """The prose outlived the code it described.

    jarvis-v3 REMOVED the gnome-screenshot path -- screen.py's own comment
    in ``_grab_cli`` says "the project spec forbids it (it drives the
    Shell's screenshot UI)" and the test above pins that ``import`` is the
    only binary run.  But the prose that described the removed path stayed:
    the module docstring still promised a "gnome-screenshot fallback" that
    "round-trips through a 0700 temp dir that is removed before the
    function returns", and ``_grab_cli``'s own docstring still promised
    "gnome-screenshot, then ImageMagick ``import``".

    There is no temp dir in this module at ALL -- ``import`` is asked for
    ``png:-`` and the PNG comes back on stdout -- so a reader chasing the
    cleanup path was chasing code that does not exist, and a reader
    auditing what touches disk was given one extra thing to worry about
    that never happens.  Pin the docs to the code both ways: what the
    source does, and what the docstrings are allowed to claim."""
    src = Path(scr.__file__).read_text()
    # What the code actually does: one CLI grabber, PNG on stdout, no disk.
    assert "png:-" in src
    assert "tempfile" not in src and "mkdtemp" not in src
    assert "0700" not in src, "no 0700 directory exists in this module"
    doc = f"{scr.__doc__ or ''}\n{scr._grab_cli.__doc__ or ''}"
    # What the docstrings may not claim.
    assert "0700" not in doc, "the docstring invents a temp dir the code never makes"
    for gone in ("gnome-screenshot fallback", "gnome-screenshot, then"):
        assert gone not in doc, f"docstring still promises {gone!r}"
    # ...and what they must still say, so this is not just a deletion.
    assert "ImageMagick" in (scr._grab_cli.__doc__ or "")




def test_vision_error_object_and_empty_answer(wired):
    vision, _ = wired
    vision.reply = {"error": f"model '{scr.chat_model()}' not found"}
    res = tool().call("screen_qa", {})
    # A model that is not there is a SETUP failure, and the spoken line says
    # so: "isn't answering" sent Hunter looking at ollama, not at the pull.
    assert res.ok is False
    assert res.speak == scr.NOT_INSTALLED_LINE.format(model="gemma4:26b")

    vision.reply = {"message": {"role": "assistant", "content": "   "}}
    res = tool().call("screen_qa", {})
    assert res.ok is False and res.speak == scr.NO_VISION_LINE

    vision.reply = ["not", "an", "object"]
    res = tool().call("screen_qa", {})
    assert res.ok is False and res.speak == scr.NO_VISION_LINE


def test_handler_never_raises_even_when_everything_explodes(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("kaboom")
    monkeypatch.setattr(scr, "_grab_screen", boom)
    monkeypatch.setattr(scr, "_window_title", boom)
    monkeypatch.setattr(scr, "_ask_vision", boom)
    spec = scr.make_tools({}, None)[0]
    res = spec.handler(question="x")          # direct, without the registry guard
    assert res.ok is False and res.speak == scr.NO_SCREEN_LINE

    monkeypatch.setattr(scr, "_grab_screen", lambda display=None: make_image(100, 100))
    monkeypatch.setattr(scr, "_window_title", lambda display=None: "")
    res = spec.handler(question="x")          # vision raising something unexpected
    assert res.ok is False and res.speak == scr.NO_VISION_LINE


# ----------------------------------------------------------- word caps
def test_answer_is_capped_and_markdown_stripped(wired):
    vision, _ = wired
    long = " ".join(f"Sentence number {i} has exactly six words." for i in range(40))
    vision.content = "**Summary:**\n- " + long
    res = tool().call("screen_qa", {"question": "everything"})
    assert res.ok
    answer = res.text.split(". ", 1)[1]      # after the "Active window: ..." clause
    assert len(answer.split()) <= scr.ANSWER_WORD_CAP
    assert answer.endswith("."), "cut at a sentence boundary"
    assert "**" not in answer and "\n" not in answer and "- " not in answer


def test_tidy_answer_edge_cases():
    assert scr.tidy_answer(None) == ""
    assert scr.tidy_answer("  short   answer  ") == "short answer"
    # one run-on sentence longer than the cap is hard-cut, not dropped
    run_on = " ".join(["word"] * 120)
    out = scr.tidy_answer(run_on, cap=10)
    assert len(out.split()) == 10 and out.endswith(".")
    # sentences that fit are kept whole; the first is kept even if over cap
    assert scr.tidy_answer("One two three. Four five six. Seven eight.", cap=6) == \
        "One two three. Four five six."
    assert scr.tidy_answer("`code` and\n# heading", cap=80) == "code and heading"
    # a mid-line hash is not a heading (a window title like "#42 open")
    assert scr.tidy_answer("issue #42 is open") == "issue #42 is open"


# ------------------------------------------------------------ privacy
def test_no_screenshot_on_disk_unless_debug(wired, tmp_path, monkeypatch):
    monkeypatch.setattr(scr, "cache_dir", lambda: tmp_path / "cache")
    res = tool().call("screen_qa", {"question": "x"})
    assert res.ok
    assert not (tmp_path / "cache").exists()

    monkeypatch.setenv(scr.DEBUG_ENV, "1")
    res = tool().call("screen_qa", {"question": "x"})
    assert res.ok
    dump = tmp_path / "cache" / scr.DEBUG_FILE
    assert dump.exists()
    assert stat.S_IMODE(dump.stat().st_mode) == 0o600
    assert dump.read_bytes()[:3] == b"\xff\xd8\xff"


def test_image_and_answer_never_reach_the_log(wired, caplog):
    vision, _ = wired
    vision.content = "The secret document says PROJECT-ZEBRA."
    with caplog.at_level(logging.DEBUG, logger="jarvis"):
        res = tool().call("screen_qa", {"question": "read it"})
    assert res.ok
    b64 = vision.payloads[-1]["messages"][-1]["images"][0]
    text = caplog.text
    assert b64[:40] not in text
    assert "PROJECT-ZEBRA" not in text
    assert "screen:" in text, "sizes and timing are logged"


# ------------------------------------------------------------- seams
def test_ask_vision_posts_json_to_ollama_chat(monkeypatch):
    seen = {}

    class Resp:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self.body

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["timeout"] = timeout
        seen["body"] = json.loads(req.data)
        seen["ctype"] = req.get_header("Content-type")
        return Resp(b'{"message": {"role": "assistant", "content": "ok"}}')
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    reply = scr._ask_vision({"model": "m", "messages": []}, timeout=7)
    assert reply["message"]["content"] == "ok"
    assert seen["url"] == "http://localhost:11434/api/chat"
    assert seen["timeout"] == 7
    assert seen["body"]["model"] == "m"
    assert seen["ctype"] == "application/json"


def test_window_title_is_bounded_and_failure_tolerant(monkeypatch):
    def fake_run(cmd, capture_output=False, text=False, timeout=None, env=None, **_):
        assert cmd[:2] == ["xdotool", "getactivewindow"]
        assert timeout <= 8 and env["DISPLAY"] == ":9"
        return subprocess.CompletedProcess(cmd, 0, "  Jarvis   —  Terminal \n", "")
    monkeypatch.setattr(scr.subprocess, "run", fake_run)
    assert scr._window_title() == "Jarvis — Terminal"

    def missing(*a, **k):
        raise FileNotFoundError("xdotool")
    monkeypatch.setattr(scr.subprocess, "run", missing)
    assert scr._window_title() == ""

    def slow(*a, **k):
        raise subprocess.TimeoutExpired("xdotool", 2)
    monkeypatch.setattr(scr.subprocess, "run", slow)
    assert scr._window_title() == ""

    def failed(cmd, **k):
        return subprocess.CompletedProcess(cmd, 1, "", "no active window")
    monkeypatch.setattr(scr.subprocess, "run", failed)
    assert scr._window_title() == ""


def test_persona_lines_are_film_jarvis():
    for line in (scr.NO_SCREEN_LINE, scr.NO_VISION_LINE):
        assert line.endswith(", sir.")
        assert len(line.split()) <= 12


def test_an_http_error_keeps_ollamas_reason(wired, caplog):
    """LIVE 2026-08-31, 21:22 and 21:27: "What's on my screen?" twice, and
    both times the entire record was

        screen: llama3.2-vision:latest did not answer: HTTPError

    -- the exception CLASS NAME -- while Hunter heard "My vision model isn't
    answering, sir."  Ollama had answered 500 with
    {"error": "... unknown model architecture: 'mllama'"}: the model is
    pulled and 7.27 GiB of it is on disk, this build's llama-server simply
    cannot load that architecture.  That is a one-line change to
    screen.model and it was nowhere in the log, so the reason has to survive
    the HTTPError: urlopen raises it, and the body is where it lives."""
    vision, _grabbed = wired
    body = (b'{"error":"llama runner process has terminated: exit status 1: '
            b'error loading model: unknown model architecture: \'mllama\'"}')
    vision.fail = urllib.error.HTTPError(
        "http://localhost:11434/api/chat", 500, "Internal Server Error",
        {}, io.BytesIO(body))
    with caplog.at_level(logging.WARNING, logger="jarvis"):
        res = tool().call("screen_qa", {})
    assert res.ok is False
    # Aloud he is told WHICH model this build cannot load, so the fix is
    # findable without reading the log.
    assert res.speak == scr.CANNOT_LOAD_LINE.format(model="gemma4:26b")
    for where in (caplog.text, res.text):
        assert "HTTP 500" in where
        assert "unknown model architecture: 'mllama'" in where


def test_a_dead_ollama_is_still_just_the_class_name(wired):
    """The HTTPError clause must not swallow the plain-transport case: a
    refused connection has no body and no status to report."""
    vision, _grabbed = wired
    vision.fail = urllib.error.URLError("connection refused")
    res = tool().call("screen_qa", {})
    assert res.ok is False and res.speak == scr.NO_VISION_LINE
    assert "URLError" in res.text



# ask_screen names FOUR classes at the transport seam (screen.py, the clause
# under the HTTPError one): URLError, OSError, ValueError, TimeoutError.
# MEASURED on this interpreter (CPython 3.12.3): socket.timeout IS
# TimeoutError, TimeoutError is an OSError subclass, and URLError is one too
# -- so that tuple is really (OSError, ValueError), and the two tests above
# already inject HTTPError and URLError.  These are the classes nothing
# pinned: a timeout, a bare socket error, and a body that is not JSON.
@pytest.mark.parametrize("failure, name", [
    (TimeoutError("timed out"), "TimeoutError"),      # == socket.timeout here
    (OSError("connection reset by peer"), "OSError"),
    (ValueError("body is not JSON"), "ValueError"),
])
def test_a_transport_failure_speaks_out_of_the_named_vision_path(wired, caplog,
                                                                 failure, name):
    """Not "does he get an excuse" -- he gets one either way, because
    ``screen_qa`` has an ``except Exception`` at the tool boundary that also
    returns NO_VISION_LINE.  What is pinned here is WHICH path produced it.

    The named path (VisionUnavailable) puts the model and the setup hint in
    ``res.text`` and logs one WARNING.  The tool-boundary catch-all puts
    ``vision failed:`` there instead -- and for a bare ``TimeoutError()``,
    whose ``str()`` is empty, that text is the word "failed" and nothing
    else -- then logs a full traceback.  That difference is the whole of the
    LIVE 2026-08-31 lesson two tests up: the excuse was spoken correctly and
    the RECORD of why was a class name, so the one-line config fix was
    invisible.  A 25 s call to a local vision model fails by timing out more
    often than by any other route, and it was the one class in that tuple
    with nothing holding it on the diagnosable path."""
    vision, _grabbed = wired
    vision.fail = failure
    wanted = scr.candidate_models(None)[0]
    with caplog.at_level(logging.DEBUG, logger="jarvis"):
        res = tool().call("screen_qa", {})
    assert res.ok is False
    assert res.speak == scr.NO_VISION_LINE
    # The reason and the fix ride in the recorded text; the catch-all can
    # produce neither -- it never sees which model was asked.
    assert name in res.text
    assert wanted in res.text
    assert scr.SETUP_HINT in res.text
    assert "vision failed:" not in res.text
    # ...and a transport failure is a warning with a hint, not a traceback.
    assert any(r.levelno == logging.WARNING and scr.SETUP_HINT in r.getMessage()
               for r in caplog.records)
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

def test_http_error_detail_survives_a_body_that_is_not_json():
    """A proxy's HTML page, or a body already read: never a second failure
    inside the error path."""
    err = urllib.error.HTTPError("u", 502, "Bad Gateway", {},
                                 io.BytesIO(b"<html>nope</html>"))
    assert "nope" in scr.http_error_detail(err)
    empty = urllib.error.HTTPError("u", 503, "Service Unavailable", {}, None)
    assert scr.http_error_detail(empty) == "Service Unavailable"


# ------------------------------------------------- #88: the model choice
def test_a_thinking_model_is_told_not_to_think(wired):
    """LIVE 2026-09-01, the whole of bug #88 after the config was pointed at
    a model this ollama CAN load: gemma4:26b answered /api/chat with
    ``done_reason "length"``, ``eval_count 200`` and

        message.content  = ''
        message.thinking = 'The user wants to know what is on their screen...'

    -- the entire 200-token num_predict budget spent reasoning, an empty
    content, and Hunter heard "My vision model isn't answering, sir." while
    the model had described his desktop perfectly.  brain.py has sent
    ``think: false`` on the chat path since it was written; the eyes did
    not.  With the field set: 0.8 s and a correct three-sentence answer."""
    vision, _ = wired
    res = tool().call("screen_qa", {"question": "what is showing"})
    assert res.ok
    assert vision.payloads[-1]["think"] is False


def test_a_model_that_cannot_think_is_not_sent_the_field(wired, monkeypatch):
    """think:false is sent from capabilities, never blind: a build that
    rejects the field on a plain model would turn a WORKING vision model
    into a 400."""
    vision, _ = wired
    monkeypatch.setattr(scr, "_ollama_show", show(["completion", "vision"]))
    res = tool().call("screen_qa", {})
    assert res.ok
    assert "think" not in vision.payloads[-1]


def test_an_all_thinking_reply_names_itself_in_the_log(wired, caplog):
    """If think:false is ever ignored the log must say WHICH emptiness this
    is -- "empty answer" is what sent this bug to a second fix pass."""
    vision, _ = wired
    vision.reply = {"message": {"role": "assistant", "content": "",
                                "thinking": "The user wants to know..."}}
    with caplog.at_level(logging.WARNING, logger="jarvis"):
        res = tool().call("screen_qa", {})
    assert res.ok is False
    assert "thinking-only" in res.text and "thinking-only" in caplog.text


def test_the_chat_models_own_runner_is_reused(wired):
    """A screen question must not restart the runner the brain is using.

    Measured 2026-09-01: with no num_ctx the request took ollama's default
    262144, which spawned a SECOND runner for the same model -- 9.4 s of
    load_duration -- and left it with the default five-minute keep_alive, so
    the brain's ``keep_alive: -1`` pin was silently gone and the next spoken
    turn paid the reload again.  Same num_ctx + keep_alive -1: 0.8 s, and
    /api/ps unchanged."""
    from jarvis import brain
    vision, _ = wired
    res = tool().call("screen_qa", {})
    assert res.ok
    payload = vision.payloads[-1]
    assert payload["model"] == scr.chat_model() == brain.OLLAMA_MODEL
    assert payload["keep_alive"] == -1
    assert payload["options"]["num_ctx"] == brain.NUM_CTX


def test_a_second_model_is_unloaded_the_moment_it_has_answered(wired):
    """OLLAMA_MAX_LOADED_MODELS=1: a model that is not the chat model has
    just evicted it, so it may not also be PINNED there -- keep_alive 0 lets
    the brain's residency loop take the slot straight back."""
    vision, _ = wired
    res = tool({"screen": {"model": "mistral-small3.2:24b"}}).call("screen_qa", {})
    assert res.ok
    payload = vision.payloads[-1]
    assert payload["model"] == "mistral-small3.2:24b"
    assert payload["keep_alive"] == 0
    # not the brain's runner: sending its num_ctx would be a coincidence, not
    # a reuse, and a wrong context window for a different model.
    assert "num_ctx" not in payload["options"]


def test_an_unloadable_model_falls_back_to_the_chat_model(wired, caplog):
    """assistant.json on the live box still says llama3.2-vision:latest, and
    ollama 0.33.1 cannot load 'mllama' at all.  A hand edit + a restart is
    the real fix, but until then the answer must still arrive: the config
    model is tried once, refused, remembered, and the chat model answers."""
    vision, _ = wired
    broken = "llama3.2-vision:latest"

    def refuse_the_broken_one(payload, timeout=None):
        vision.payloads.append(payload)
        if payload["model"] == broken:
            body = (b'{"error":"error loading model: unknown model '
                    b"architecture: 'mllama'\"}")
            raise urllib.error.HTTPError("u", 500, "Internal Server Error",
                                         {}, io.BytesIO(body))
        return {"message": {"role": "assistant", "content": "A terminal."}}

    import jarvis.tools.screen as _scr
    _scr._ask_vision = refuse_the_broken_one          # restored by monkeypatch below
    try:
        reg = tool({"screen": {"model": broken}})
        with caplog.at_level(logging.WARNING, logger="jarvis"):
            res = reg.call("screen_qa", {})
        assert res.ok and res.speak == "A terminal."
        assert [p["model"] for p in vision.payloads] == [broken, scr.chat_model()]
        assert "mllama" in caplog.text and "screen.model" in caplog.text
        # ...and the wasted attempt is paid ONCE: the verdict is remembered.
        res = reg.call("screen_qa", {})
        assert res.ok
        assert [p["model"] for p in vision.payloads] == \
            [broken, scr.chat_model(), scr.chat_model()]
    finally:
        _scr._ask_vision = vision


def test_a_model_without_vision_is_never_even_asked(wired, monkeypatch):
    """/api/show reads the manifest and loads nothing, so the capability
    check is free -- and it keeps a text-only model from being handed an
    image and blamed for the silence."""
    vision, _ = wired
    asked = []
    monkeypatch.setattr(scr, "_ollama_show",
                        show({"qwen3:30b-a3b": ["completion", "tools"]}, asked))
    res = tool({"screen": {"model": "qwen3:30b-a3b"}}).call("screen_qa", {})
    assert "qwen3:30b-a3b" in asked
    assert [p["model"] for p in vision.payloads] == [scr.chat_model()], \
        "the text-only model is skipped, the chat model answers"
    assert res.ok
    # ...and with no fallback left, the spoken line names the reason.
    scr.forget_models()
    monkeypatch.setattr(scr, "_ollama_show", show(["completion", "tools"]))
    res = tool().call("screen_qa", {})
    assert res.ok is False
    assert res.speak == scr.NOT_VISION_LINE.format(model=scr.chat_model())


def test_the_verdict_cache_expires(wired, monkeypatch):
    """A remembered "this model will not load" must not outlive an `ollama
    pull` or an ollama upgrade; Jarvis runs for days at a time."""
    scr.mark_unusable("llama3.2-vision:latest", "HTTP 500: nope")
    assert scr.unusable_reason("llama3.2-vision") == "HTTP 500: nope"
    later = time.monotonic() + scr.CACHE_TTL_S + 1
    monkeypatch.setattr(scr.time, "monotonic", lambda: later)
    assert scr.unusable_reason("llama3.2-vision:latest") == ""


def test_model_caps_reads_api_show_and_reports_a_missing_model(monkeypatch):
    posted = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"capabilities": ["completion", "vision"]}).encode()

    def fake_urlopen(req, timeout=None):
        posted["url"] = req.full_url
        posted["body"] = json.loads(req.data)
        return Resp()
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    caps, why = scr.model_caps("gemma4:26b")
    assert posted["url"].endswith("/api/show"), "show, not generate: it loads nothing"
    assert posted["body"] == {"model": "gemma4:26b"}
    assert "vision" in caps and why == ""

    scr.forget_models()

    def missing(req, timeout=None):
        raise urllib.error.HTTPError(
            "u", 404, "Not Found", {},
            io.BytesIO(b'{"error":"model \'moondream\' not found"}'))
    monkeypatch.setattr(urllib.request, "urlopen", missing)
    caps, why = scr.model_caps("moondream")
    assert caps is None and "not found" in why
