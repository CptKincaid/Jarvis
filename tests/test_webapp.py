"""The phone client (jarvis/webapp.py).

Real modules throughout -- the real AssistantConfig, the real intercom
decode over a real wav, the real cmdsock reply collector, and the real
JarvisApp for the wiring section. The only concession to a test box is the
bind: every server here binds 127.0.0.1 on port 0 (an ephemeral loopback
port the kernel picks) and is closed in the fixture's teardown, so nothing
is ever reachable off this machine and no fixed port is claimed.

The QR encoder is checked by DECODING what it draws, with OpenCV -- an
encoder that is merely well-formed but unscannable is worse than no QR at
all, because the failure only shows up with a phone in his hand.
"""
from __future__ import annotations

import http.client
import io
import json
import os
import socket
import stat
from types import SimpleNamespace

import pytest

import jarvis.app as app_mod
import jarvis.intercom as intercom
import jarvis.webapp as wa
from jarvis.assistant_config import SECRET_KEYS, AssistantConfig
from jarvis.commander import CommandResult
from jarvis.events import JarvisReply, Status, bus
from tests.test_cmdsock import FakeApp
from tests.test_app_wiring import build, paths, seams  # noqa: F401 - fixtures


# ------------------------------------------------------------------ helpers
class PhoneApp(FakeApp):
    """cmdsock's fake app plus the two seams the phone adds: an assistant
    config (the intercom switches) and decode_clip (Whisper's stand-in)."""

    def __init__(self, cfg=None):
        super().__init__()
        self.assistant = cfg
        self.heard = "what's due this week"
        self.clips: list = []

    def decode_clip(self, audio, verify=True):
        self.clips.append((audio, verify))
        result = SimpleNamespace(text=self.heard, accepted=True)
        return audio, {}, False, result


def make_config(tmp_path, **phone):
    values = {"enabled": True, "bind": "127.0.0.1", "port": 0,
              "link_file": str(tmp_path / "link.txt"),
              "qr_file": str(tmp_path / "link.svg")}
    values.update(phone)
    return AssistantConfig({"phone": values}, path=tmp_path / "assistant.json")


@pytest.fixture
def server(tmp_path):
    cfg = make_config(tmp_path)
    app = PhoneApp(cfg)
    srv = wa.PhoneServer(app, cfg=cfg, timeout_s=5.0, idle_grace_s=0.4)
    assert srv.start(), "the loopback bind should have succeeded"
    yield SimpleNamespace(app=app, cfg=cfg, srv=srv, tmp=tmp_path)
    srv.stop()


def call(srv, method, path, body=None, token=None, ctype=None, headers=None):
    """One request; returns (status, parsed-json-or-bytes)."""
    conn = http.client.HTTPConnection(srv.host, srv.port, timeout=20)
    try:
        head = dict(headers or {})
        if token is not None:
            head["Authorization"] = "Bearer " + token
        if ctype:
            head["Content-Type"] = ctype
        conn.request(method, path, body=body, headers=head)
        resp = conn.getresponse()
        raw = resp.read()
        try:
            return resp.status, json.loads(raw.decode("utf-8"))
        except ValueError:
            return resp.status, raw
    finally:
        conn.close()


def ask(srv, text, token=None, **extra):
    payload = {"text": text}
    payload.update(extra)
    return call(srv, "POST", "/api/ask", json.dumps(payload),
                token=srv.token if token is None else token,
                ctype="application/json")


def wav_bytes(seconds=0.6, rate=16000):
    """A real container libsndfile reads, so to_mono_16k runs for real."""
    sf = pytest.importorskip("soundfile")
    np = pytest.importorskip("numpy")
    n = int(seconds * rate)
    tone = (0.2 * np.sin(2 * np.pi * 220 *
                         np.arange(n) / rate)).astype("float32")
    buf = io.BytesIO()
    sf.write(buf, tone, rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ---------------------------------------------------- 1. the bind is private
@pytest.mark.parametrize("host", [
    "192.168.50.11", "10.0.0.4", "172.16.9.9", "127.0.0.1", "169.254.3.3",
    # 100.64.0.0/10 is carrier-grade NAT, which some home routers hand out.
    # ipaddress.is_private says False for it -- which is why the check is
    # `not is_global` and not a list of ranges someone wrote from memory.
    "100.64.0.7", "fd00::1", "[fd00::1]", "::1",
])
def test_private_addresses_are_allowed(host):
    assert wa.is_private_host(host)


@pytest.mark.parametrize("host", [
    "0.0.0.0", "::", "*", "", "   ", "8.8.8.8", "1.1.1.1",
    "2606:4700::1111", "jarvis.local", "localhost", "example.com",
    "224.0.0.1", "192.168.1.1:8765",
])
def test_public_wildcard_and_name_binds_are_refused(host):
    """A hostname is refused rather than resolved: a name that resolves to
    a public record must not sneak past a check that resolves first."""
    assert not wa.is_private_host(host)


def test_a_public_bind_address_refuses_to_start_and_says_so(tmp_path):
    cfg = make_config(tmp_path, bind="0.0.0.0")
    seen: list = []
    bus.subscribe(Status, seen.append)
    try:
        srv = wa.PhoneServer(PhoneApp(cfg), cfg=cfg)
        assert srv.start() is False
        assert not srv.running
        bus.drain()
    finally:
        bus.unsubscribe(Status, seen.append)
    assert any("public bind" in e.text for e in seen), \
        "a refused bind must be visible, not merely absent"


def test_a_routable_bind_address_refuses_to_start(tmp_path):
    cfg = make_config(tmp_path, bind="8.8.8.8")
    srv = wa.PhoneServer(PhoneApp(cfg), cfg=cfg)
    assert srv.start() is False
    assert not srv.running


def test_lan_address_is_never_a_public_one():
    """Whatever this box's default route is, the guess is only returned
    when it is private -- otherwise "" and start() refuses."""
    addr = wa.lan_address()
    assert addr == "" or wa.is_private_host(addr)


# ------------------------------------------------------------ 2. off by default
def test_the_feature_is_off_in_the_shipped_defaults():
    assert AssistantConfig.DEFAULTS["phone"]["enabled"] is False
    assert AssistantConfig.DEFAULTS["phone"]["bind"] == ""
    assert AssistantConfig.DEFAULTS["phone"]["token"] == ""


def test_disabled_binds_nothing(tmp_path):
    cfg = make_config(tmp_path, enabled=False)
    srv = wa.PhoneServer(PhoneApp(cfg), cfg=cfg)
    assert srv.start() is False
    assert not srv.running
    assert srv.port == 0


def test_no_config_at_all_is_off_rather_than_open():
    srv = wa.PhoneServer(SimpleNamespace(), cfg=None)
    assert srv.start() is False
    assert not srv.running


# ---------------------------------------------------------------- 3. the key
def test_a_key_is_generated_on_first_enable_and_persisted(tmp_path):
    cfg = make_config(tmp_path)
    assert cfg.get("phone.token") == ""
    srv = wa.PhoneServer(PhoneApp(cfg), cfg=cfg)
    try:
        assert srv.start()
        assert len(srv.token) >= 32
        assert cfg.get("phone.token") == srv.token
        # and it survives a reload from the file, so the home-screen link
        # he saved this morning still works after a restart
        again = AssistantConfig.load(cfg.path)
        assert again.get("phone.token") == srv.token
    finally:
        srv.stop()


def test_the_key_is_a_secret_and_is_masked(tmp_path):
    assert "phone.token" in SECRET_KEYS
    cfg = make_config(tmp_path, token="s3cr3t-key-value")
    assert cfg.redacted()["phone"]["token"] != "s3cr3t-key-value"
    assert "s3cr3t-key-value" not in repr(cfg)
    assert "s3cr3t-key-value" not in cfg.scrub("the key is s3cr3t-key-value")


def test_an_existing_key_is_reused_not_rotated(tmp_path):
    cfg = make_config(tmp_path, token="kept-across-restarts-0123456789")
    srv = wa.PhoneServer(PhoneApp(cfg), cfg=cfg)
    try:
        assert srv.start()
        assert srv.token == "kept-across-restarts-0123456789"
    finally:
        srv.stop()


def test_authorized_rejects_everything_but_the_key(tmp_path):
    srv = wa.PhoneServer(PhoneApp(), cfg=None)
    srv.token = "abcdef"
    assert srv.authorized("abcdef")
    assert not srv.authorized("abcdeg")
    assert not srv.authorized("abcde")
    assert not srv.authorized("")
    assert not srv.authorized(None)
    srv.token = ""
    assert not srv.authorized("abcdef")      # no key -> nothing is authorised


# ------------------------------------------------------------- 4. every gate
def test_no_key_no_action(server):
    for method, path, body in (("POST", "/api/ask", '{"text": "hello"}'),
                               ("POST", "/api/voice", b"xx"),
                               ("GET", "/api/ping", None)):
        status, out = call(server.srv, method, path, body,
                           ctype="application/json")
        assert status == 401, (method, path)
        assert "key" in out["error"]
    assert server.app.calls == [], "nothing may dispatch without the key"


def test_a_wrong_key_no_action(server):
    status, _ = ask(server.srv, "what's due", token="not-the-key")
    assert status == 401
    assert server.app.calls == []


def test_the_key_in_the_query_string_works_too(server):
    """The home-screen link carries ?t=... ; the page uses a header, but a
    plain GET of /api/ping is how the link is checked."""
    status, out = call(server.srv, "GET",
                       "/api/ping?t=" + server.srv.token)
    assert status == 200 and out["ok"] is True


# ------------------------------------------------------------ 5. the text turn
def test_a_question_goes_through_dispatch_text_as_the_phone(server):
    status, out = ask(server.srv, "what time is it")
    assert status == 200
    assert out["messages"][0]["text"] == "It is ten, sir."
    assert server.app.calls == [("what time is it", "phone", True)]


def test_phone_is_a_socket_source_so_the_turn_is_quiet_by_default(server):
    ask(server.srv, "what time is it")
    assert server.app.calls[-1][2] is True, "silent unless he asks aloud"
    ask(server.srv, "what time is it", speak=True)
    assert server.app.calls[-1][2] is False
    assert wa.SOURCE in app_mod.SOCKET_SOURCES


def test_the_turn_is_stamped_so_two_phones_do_not_cross(server):
    ask(server.srv, "what time is it")
    assert server.app.turn_id and len(server.app.turn_id) == 12


def test_a_reply_stamped_for_another_turn_is_not_read_as_this_one(server):
    """The collector is cmdsock's, and this is the property it exists for."""
    app = server.app

    def foreign(text, source="typed", quiet=False, turn_id=""):
        app.calls.append((text, source, quiet))
        bus.publish(JarvisReply(text="somebody else's answer",
                                turn_id="a-different-turn"))
        return app.result

    app.dispatch_text = foreign
    status, out = ask(server.srv, "what time is it", timeout=3)
    assert status == 200
    texts = [m.get("text") for m in out["messages"]]
    assert "somebody else's answer" not in texts


def test_an_answer_that_arrives_later_still_reaches_the_phone(server):
    """A brain turn answers from a worker thread; the stream waits for the
    thinking -> idle edge, exactly as the CLI does."""
    server.app.result = CommandResult(handled=True, reply="Thinking, sir.",
                                      speak=False, done=False)
    server.app.plan = "brain"
    server.app.answer = "Two items on Tuesday, sir."
    status, out = ask(server.srv, "how heavy is my week", timeout=8)
    assert status == 200
    texts = [m.get("text") for m in out["messages"] if m["kind"] == "reply"]
    assert "Two items on Tuesday, sir." in texts
    assert out["reason"] == "answered"


def test_status_and_board_are_reads_not_turns(server):
    for text, expect in (("status", "All systems nominal, sir."),
                         ("diagnostics", "All systems nominal, sir."),
                         ("board", "VITALS")):
        status, out = ask(server.srv, text)
        assert status == 200
        assert expect in out["messages"][0]["text"]
    assert server.app.calls == [], "a read must never travel the dispatch path"


def test_an_empty_question_is_refused_without_dispatching(server):
    status, out = ask(server.srv, "   ")
    assert status == 400
    assert server.app.calls == []


def test_a_malformed_body_is_a_clear_400(server):
    status, out = call(server.srv, "POST", "/api/ask", "{not json",
                       token=server.srv.token, ctype="application/json")
    assert status == 400
    assert "malformed" in out["error"]


# ------------------------------------------------------------- 6. the clip
def test_a_clip_takes_the_intercom_path_and_comes_back_as_heard(server):
    raw = wav_bytes()
    status, out = call(server.srv, "POST", "/api/voice", raw,
                       token=server.srv.token, ctype="audio/wav")
    assert status == 200
    assert out["heard"] == "what's due this week"
    # intercom decoded it to the pipeline's own mono float32 @ 16 kHz
    audio, verify = server.app.clips[-1]
    assert audio.dtype.name == "float32" and audio.ndim == 1
    assert verify is False                     # intercom.verify_speaker default
    assert server.app.calls[-1][:2] == ("what's due this week", "phone")


def test_a_clip_jarvis_cannot_decode_says_which_formats_work(server):
    status, out = call(server.srv, "POST", "/api/voice", b"\x00" * 4096,
                       token=server.srv.token, ctype="audio/webm")
    assert status == 400
    assert out["error"] == intercom.BAD_FORMAT_LINE
    assert server.app.calls == []


def test_the_intercom_switch_turns_the_microphone_leg_off_too(tmp_path):
    """One switch, both transports: webapp reads intercom.settings."""
    cfg = make_config(tmp_path)
    cfg.set("intercom.enabled", False)
    app = PhoneApp(cfg)
    srv = wa.PhoneServer(app, cfg=cfg)
    try:
        assert srv.start()
        status, out = call(srv, "POST", "/api/voice", wav_bytes(),
                           token=srv.token, ctype="audio/wav")
        assert status == 400
        assert out["error"] == intercom.OFF_LINE
        assert app.calls == []
    finally:
        srv.stop()


# ------------------------------------------------------------- 7. the limits
def test_an_oversized_question_is_refused_before_it_is_read(server):
    body = json.dumps({"text": "x" * (wa.MAX_TEXT_BYTES + 100)})
    status, out = call(server.srv, "POST", "/api/ask", body,
                       token=server.srv.token, ctype="application/json")
    assert status == 413
    assert server.app.calls == []


def test_an_oversized_clip_is_told_so_rather_than_hung_up_on(server):
    """Just past the cap: drained and answered, so the phone can say why
    instead of showing a bare network error (see _Handler._refuse)."""
    cap = server.srv.max_audio_bytes
    status, out = call(server.srv, "POST", "/api/voice", b"\x00" * (cap + 1024),
                       token=server.srv.token, ctype="audio/wav")
    assert status == 413
    assert "too large" in out["error"]
    assert server.app.calls == []


def test_a_runaway_upload_is_cut_off_rather_than_swallowed(server):
    """Far past the cap: the body is NOT read to the end. Either the 413
    arrives or the pipe breaks first -- both are the upload being refused,
    and neither reads a gigabyte into this process."""
    cap = server.srv.max_audio_bytes
    huge = cap + wa.DRAIN_SLACK_BYTES * 4
    try:
        status, _ = call(server.srv, "POST", "/api/voice", b"\x00" * huge,
                         token=server.srv.token, ctype="audio/wav")
        assert status == 413
    except OSError:
        pass                       # the pipe broke: refused before the end
    assert server.app.calls == []


def test_the_audio_cap_never_exceeds_the_intercoms_own(tmp_path):
    cfg = make_config(tmp_path, max_audio_mb=9999)
    srv = wa.PhoneServer(PhoneApp(cfg), cfg=cfg)
    assert srv.max_audio_bytes == intercom.MAX_AUDIO_BYTES
    cfg.set("phone.max_audio_mb", "nonsense")
    assert srv.max_audio_bytes == wa.DEFAULT_MAX_AUDIO_MB * 1048576


def test_a_body_with_no_length_is_refused(server):
    """Refusing before the read is the whole point of requiring a length."""
    conn = http.client.HTTPConnection(server.srv.host, server.srv.port,
                                      timeout=10)
    try:
        conn.putrequest("POST", "/api/ask", skip_accept_encoding=True)
        conn.putheader("Authorization", "Bearer " + server.srv.token)
        conn.endheaders()
        resp = conn.getresponse()
        assert resp.status == 411
    finally:
        conn.close()
    assert server.app.calls == []


def test_the_rate_limiter_lets_a_burst_through_then_holds(tmp_path):
    lim = wa.RateLimiter(window_s=60.0, limit=3, audio_limit=1)
    assert [lim.allow("a") for _ in range(4)] == [True, True, True, False]
    assert lim.allow("b") is True              # per client, not global
    lim2 = wa.RateLimiter(window_s=60.0, limit=10, audio_limit=1)
    assert lim2.allow("a", audio=True) is True
    assert lim2.allow("a", audio=True) is False
    assert lim2.allow("a") is True             # the text bucket is separate


def test_the_rate_limit_answers_429_rather_than_dispatching(server):
    server.srv.limiter = wa.RateLimiter(window_s=60.0, limit=2)
    assert ask(server.srv, "one")[0] == 200
    assert ask(server.srv, "two")[0] == 200
    status, out = ask(server.srv, "three")
    assert status == 429
    assert len(server.app.calls) == 2


# ------------------------------------------------------- 8. the page itself
def test_the_page_is_served_and_carries_no_data(server):
    status, body = call(server.srv, "GET", "/")
    assert status == 200
    page = body.decode("utf-8")
    assert "<title>Jarvis</title>" in page
    # the shell is static: the key is never baked into it
    assert server.srv.token not in page
    assert "apple-touch-icon" in page and "manifest.webmanifest" in page


def test_the_page_offers_his_common_asks_as_taps(server):
    _, body = call(server.srv, "GET", "/")
    page = body.decode("utf-8")
    for label, phrase in wa.QUICK_TAPS:
        assert label in page and phrase in page
    labels = [t[0] for t in wa.QUICK_TAPS]
    for want in ("What's due", "Next exam", "Briefing", "Diagnostics",
                 "Lights up", "Lights down", "Calendar"):
        assert want in labels
    assert any("timer" in p for _, p in wa.QUICK_TAPS)


def test_the_page_is_honest_that_the_microphone_will_not_work(server):
    _, body = call(server.srv, "GET", "/")
    page = body.decode("utf-8")
    assert "isSecureContext" in page, "the check must actually be made"
    assert wa.MIC_NOTE in page
    assert "unavailable here" in page, "a dead button must LOOK dead"
    assert "127.0.0.1" in page, "and must say what would fix it"


def test_add_to_home_screen_has_what_ios_needs(server):
    status, manifest = call(server.srv, "GET", "/manifest.webmanifest")
    assert status == 200
    assert manifest["display"] == "standalone"
    assert {i["sizes"] for i in manifest["icons"]} == {"180x180", "512x512"}
    for path in ("/icon-180.png", "/apple-touch-icon.png", "/icon-512.png"):
        status, png = call(server.srv, "GET", path)
        assert status == 200
        assert png[:8] == b"\x89PNG\r\n\x1a\n", path
    _, body = call(server.srv, "GET", "/")
    page = body.decode("utf-8")
    assert 'name="apple-mobile-web-app-capable" content="yes"' in page


def test_an_unknown_path_is_a_404(server):
    status, out = call(server.srv, "GET", "/wp-login.php")
    assert status == 404


# ------------------------------------------------------------ 9. the link
def test_the_link_file_is_written_for_his_eyes_only(server):
    path = server.tmp / "link.txt"
    assert path.exists()
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    text = path.read_text()
    assert server.srv.token in text
    assert f"127.0.0.1:{server.srv.port}" in text
    assert "Add to Home" in text
    svg = server.tmp / "link.svg"
    assert svg.exists()
    assert stat.S_IMODE(os.stat(svg).st_mode) == 0o600
    assert svg.read_text().startswith("<svg")


def test_the_qr_actually_scans():
    """Verified by DECODING it: a well-formed but unscannable QR is worse
    than none, because it only fails with a phone in his hand."""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    link = "http://192.168.50.11:8765/?t=" + "Zq7x" * 10 + "abc"
    grid = wa._qr_matrix(link)
    scale, quiet = 10, 4
    side = (len(grid) + quiet * 2) * scale
    img = np.full((side, side), 255, dtype=np.uint8)
    for r, row in enumerate(grid):
        for c, on in enumerate(row):
            if on:
                y, x = (r + quiet) * scale, (c + quiet) * scale
                img[y:y + scale, x:x + scale] = 0
    got, _pts, _ = cv2.QRCodeDetector().detectAndDecode(img)
    assert got == link


def test_a_link_too_long_for_a_qr_is_refused_not_drawn_wrong():
    with pytest.raises(ValueError):
        wa.qr_svg("x" * 200)


# ---------------------------------------------------------- 10. lifecycle
def test_stop_gives_the_port_back(tmp_path):
    cfg = make_config(tmp_path)
    srv = wa.PhoneServer(PhoneApp(cfg), cfg=cfg)
    assert srv.start()
    host, port = srv.host, srv.port
    assert call(srv, "GET", "/")[0] == 200
    srv.stop()
    assert not srv.running
    with pytest.raises(OSError):
        s = socket.create_connection((host, port), timeout=2)
        s.close()


def test_binding_does_no_reverse_dns_lookup(server):
    """HTTPServer.server_bind calls getfqdn() for a field nothing reads;
    on a box with a slow resolver that would stall start_assistant."""
    assert server.srv._httpd.server_name == "127.0.0.1"
    assert isinstance(server.srv._httpd, wa._Server)


def test_an_idle_keep_alive_connection_cannot_park_a_thread_forever(server):
    assert wa._Handler.timeout and wa._Handler.timeout <= 60
    assert wa._Handler.protocol_version == "HTTP/1.1"


def test_start_is_idempotent(server):
    port = server.srv.port
    assert server.srv.start() is True
    assert server.srv.port == port


def test_stop_is_safe_when_it_never_started(tmp_path):
    cfg = make_config(tmp_path, enabled=False)
    srv = wa.PhoneServer(PhoneApp(cfg), cfg=cfg)
    srv.stop()              # must not raise
    assert not srv.running


def test_a_port_already_taken_is_a_warning_not_a_crash(tmp_path):
    held = socket.socket()
    held.bind(("127.0.0.1", 0))
    held.listen(1)
    try:
        cfg = make_config(tmp_path, port=held.getsockname()[1])
        srv = wa.PhoneServer(PhoneApp(cfg), cfg=cfg)
        assert srv.start() is False
        assert not srv.running
    finally:
        held.close()


# ------------------------------------------------------------ 11. wiring
def test_the_real_app_builds_the_phone_client_and_leaves_it_off(build):  # noqa: F811
    app = build()
    assert app.webapp is not None
    assert isinstance(app.webapp, wa.PhoneServer)
    app.start_assistant()
    # phone.enabled is false in DEFAULTS, so start_assistant binds nothing
    assert app.webapp.running is False
    assert app.webapp.port == 0


def test_the_real_app_starts_and_stops_it_like_every_other_thread(build,  # noqa: F811
                                                                  tmp_path):
    app = build()
    app.assistant.update({"phone.enabled": True, "phone.bind": "127.0.0.1",
                          "phone.port": 0,
                          "phone.link_file": str(tmp_path / "l.txt"),
                          "phone.qr_file": str(tmp_path / "l.svg")})
    app.start_assistant()
    try:
        assert app.webapp.running, "start_assistant must start it"
        status, body = call(app.webapp, "GET", "/")
        assert status == 200
    finally:
        app.stop_assistant()
    assert app.webapp.running is False


def test_a_real_phone_turn_reaches_the_real_commander(build, tmp_path):  # noqa: F811
    """One round trip through the real JarvisApp: HTTP in, a real Tier-1
    handler, a JarvisReply out -- and nothing spoken, because the phone is
    a quiet source."""
    app = build()
    app.assistant.update({"phone.enabled": True, "phone.bind": "127.0.0.1",
                          "phone.port": 0,
                          "phone.link_file": str(tmp_path / "l.txt"),
                          "phone.qr_file": str(tmp_path / "l.svg")})
    app.start_assistant()
    try:
        status, out = call(app.webapp, "POST", "/api/ask",
                           json.dumps({"text": "what time is it"}),
                           token=app.webapp.token,
                           ctype="application/json")
        assert status == 200
        replies = [m["text"] for m in out["messages"] if m["kind"] == "reply"]
        assert replies and any(ch.isdigit() for ch in " ".join(replies))
        assert app.tts.spoken == [], "a phone turn must not wake the room"
    finally:
        app.stop_assistant()
