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
import threading
from types import SimpleNamespace

import pytest

import jarvis.app as app_mod
import jarvis.intercom as intercom
import jarvis.webapp as wa
from jarvis.assistant_config import SECRET_KEYS, AssistantConfig
from jarvis.commander import CommandResult
from jarvis.events import JarvisReply, Status, bus
from tests.test_cmdsock import FakeApp
from tests.test_app_wiring import (FakeRendition, FakeTTS, build,  # noqa: F401
                                   paths, seams)          # - fixtures


# ------------------------------------------------------------------ helpers
class PhoneApp(FakeApp):
    """cmdsock's fake app plus the two seams the phone adds: an assistant
    config (the intercom switches) and decode_clip (Whisper's stand-in)."""

    def __init__(self, cfg=None):
        super().__init__()
        self.assistant = cfg
        self.heard = "what's due this week"
        self.clips: list = []
        # The voice seam. FakeTTS keeps `spoken` (what the ROOM was asked to
        # say) apart from `rendered` (what went to the phone), which is how
        # every test below proves the soundbar stayed out of it.
        self.tts = FakeTTS()

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


# ------------------------------------------------------- 12. his own voice
def audio(srv, text, token=None, headers=None):
    """POST /api/say and give back (status, headers, body). Not `call`:
    that one parses JSON, and this endpoint is a wav."""
    conn = http.client.HTTPConnection(srv.host, srv.port, timeout=20)
    try:
        head = {"Authorization": "Bearer " + (srv.token if token is None
                                              else token),
                "Content-Type": "application/json"}
        head.update(headers or {})
        conn.request("POST", "/api/say", body=json.dumps({"text": text}),
                     headers=head)
        resp = conn.getresponse()
        return resp.status, dict(resp.getheaders()), resp.read()
    finally:
        conn.close()


def test_a_reply_comes_back_as_his_voice_not_the_browsers(server):
    """The whole point: the audio is rendered by the SAME TTS the room
    uses. A browser's own speech synthesis would be a different assistant
    entirely, and would need no server at all."""
    status, head, body = audio(server.srv, "Two items on Tuesday, sir.")
    assert status == 200
    assert head["Content-Type"] == "audio/wav"
    assert body[:4] == b"RIFF" and body[8:12] == b"WAVE"
    assert server.app.tts.rendered == ["Two items on Tuesday, sir."]


def test_a_cached_reply_is_sent_whole_with_a_length_and_ranges(server):
    """Everything already on disk means no synthesis at all, so the length
    is known -- and a known length is what lets a media element seek."""
    status, head, body = audio(server.srv, "Always, sir.")
    assert status == 200
    assert head["Content-Length"] == str(len(body))
    assert head["Accept-Ranges"] == "bytes"
    assert "Transfer-Encoding" not in head


def test_a_range_request_gets_that_slice_and_says_which(server):
    """Safari probes an audio source with `Range: bytes=0-1` before it will
    commit to it, and a 200 to that probe has been the reason for a silent
    <audio> tag more than once."""
    _, _, whole = audio(server.srv, "Always, sir.")
    status, head, body = audio(server.srv, "Always, sir.",
                               headers={"Range": "bytes=0-1"})
    assert status == 206
    assert body == whole[:2]
    assert head["Content-Range"] == f"bytes 0-1/{len(whole)}"
    status, _, tail = audio(server.srv, "Always, sir.",
                            headers={"Range": "bytes=44-"})
    assert status == 206 and tail == whole[44:]


def test_a_range_it_cannot_honour_sends_the_whole_thing(server):
    """Ignoring a Range is legal; guessing at a multipart one is not."""
    _, _, whole = audio(server.srv, "Always, sir.")
    for spec in ("bytes=0-1,4-5", "chickens=0-1", "bytes=99999-", "bytes=x-y"):
        status, _, body = audio(server.srv, "Always, sir.",
                                headers={"Range": spec})
        assert status == 200 and body == whole, spec


def test_a_reply_that_must_be_rendered_streams_as_it_lands(server):
    """The phone plays sentence one while sentence two is still on the GPU,
    so the bytes must leave before the render is finished."""
    server.app.tts.rendition = lambda text: FakeRendition(
        text, cached=False, chunks=["one.", "two."])
    conn = http.client.HTTPConnection(server.srv.host, server.srv.port,
                                      timeout=20)
    try:
        conn.request("POST", "/api/say", body=json.dumps({"text": "one. two."}),
                     headers={"Authorization": "Bearer " + server.srv.token,
                              "Content-Type": "application/json"})
        resp = conn.getresponse()
        assert resp.status == 200
        assert resp.getheader("Transfer-Encoding") == "chunked"
        assert resp.getheader("Content-Length") is None
        head = resp.read(44)
        assert head[:4] == b"RIFF"
        assert head[40:44] == b"\xff\xff\xff\xff", \
            "a stream cannot know its length yet, and must not claim one"
        assert len(resp.read()) > 0
    finally:
        conn.close()


def test_the_first_bytes_reach_the_phone_before_the_last_chunk_renders(server):
    """Same property, proved rather than inferred: the header and the first
    sentence are readable off the socket while chunk two is still blocked."""
    gate = threading.Event()

    def held(text):
        rend = FakeRendition(text, cached=False,
                                    chunks=["one.", "two."])
        rend.gate = gate
        return rend

    server.app.tts.rendition = held
    conn = http.client.HTTPConnection(server.srv.host, server.srv.port,
                                      timeout=20)
    try:
        conn.request("POST", "/api/say", body=json.dumps({"text": "one. two."}),
                     headers={"Authorization": "Bearer " + server.srv.token,
                              "Content-Type": "application/json"})
        resp = conn.getresponse()
        first = resp.read(44 + len(FakeRendition.PCM))
        assert len(first) == 44 + len(FakeRendition.PCM), \
            "sentence one must arrive while sentence two is still rendering"
        gate.set()
        assert len(resp.read()) == len(FakeRendition.PCM)
    finally:
        gate.set()
        conn.close()


def test_the_voice_needs_the_key_like_everything_else(server):
    status, _, _ = audio(server.srv, "Always, sir.", token="not-the-key")
    assert status == 401
    assert server.app.tts.rendered == []


def test_an_empty_reply_is_refused_without_rendering(server):
    status, _, _ = audio(server.srv, "   ")
    assert status == 400
    assert server.app.tts.rendered == []


def test_a_reply_with_nothing_sayable_in_it_is_silence_not_an_error(server):
    """A reply that cleans down to nothing (all markup, all punctuation)
    should make no sound. A 4xx would put a red line on his screen for a
    reply he can read perfectly well."""
    server.app.tts.rendition = lambda text: FakeRendition(text, chunks=[])
    status, head, body = audio(server.srv, "`x`")
    assert status == 200
    assert body == b""
    assert head["Content-Length"] == "0"


def test_a_box_with_no_speech_engine_says_so_rather_than_going_quiet(server):
    """A dead Voice switch that says it is dead beats one that looks alive
    and answers in silence -- which is what a broken toggle looks like."""
    server.app.tts = None
    status, _, body = audio(server.srv, "Always, sir.")
    assert status == 503
    assert b"speech engine" in body
    status, out = call(server.srv, "GET", "/api/ping", token=server.srv.token)
    assert out["voice"] is False


def test_a_render_that_fails_answers_before_it_commits_to_a_clip(server):
    """The header is pulled before the response line goes out precisely so
    a doomed render is a 503 and not a truncated 200."""
    def doomed(text):
        raise RuntimeError("the sidecar went away")

    server.app.tts.rendition = doomed
    status, _, body = audio(server.srv, "Always, sir.")
    assert status == 503
    assert b"voice" in body


def test_the_ping_says_whether_there_is_a_voice_at_all(server):
    status, out = call(server.srv, "GET", "/api/ping", token=server.srv.token)
    assert status == 200 and out["voice"] is True


# ------------------------------------------- 13. the room stays out of it
def test_asking_for_audio_never_wakes_the_room(server):
    """The Spark's speaker is not on this path at all. `spoken` is what
    app._say puts into the room; `rendered` is what went to the phone."""
    audio(server.srv, "Two items on Tuesday, sir.")
    assert server.app.tts.spoken == [], "the room must stay silent"
    assert server.app.tts.rendered == ["Two items on Tuesday, sir."]


def test_the_two_switches_are_different_rooms_and_say_so(server):
    """`aloud` is the Spark answering the house; `voice` is this phone.
    They are separate controls with separate labels, and both start off."""
    status, page = call(server.srv, "GET", "/")
    assert status == 200
    body = page.decode("utf-8")
    assert 'id="voice"' in body and 'id="aloud"' in body
    assert "Aloud in the room" in body
    assert 'type="checkbox" id="aloud">' in body, \
        "the room switch must not be checked by default"
    assert 'aria-pressed="false"' in body, "and neither must the phone's"


def test_the_page_fetches_nothing_at_all_while_the_voice_is_off(server):
    """Off has to mean nothing is rendered, not that a clip was made and
    discarded -- he uses this in lectures."""
    body = call(server.srv, "GET", "/")[1].decode("utf-8")
    assert "if (!voiceOn || !t) { return; }" in body
    assert 'localStorage.setItem(VOICE_STORE' in body


def test_the_gesture_that_enables_the_voice_is_the_one_ios_needs(server):
    """iOS only lets an AudioContext out of 'suspended' inside a user
    gesture, so the context is built on the enabling TAP -- and re-armed on
    every send, for a switch remembered from last time."""
    body = call(server.srv, "GET", "/")[1].decode("utf-8")
    assert 'voiceBtn.addEventListener("click"' in body
    assert "if (voiceOn) { hush(); unlock(); }" in body
    assert 'note("This phone is holding audio back' in body


# ------------------------------------------------- 14. honest about the link
def test_a_turn_reports_how_long_the_server_half_took(server):
    """Without it the page cannot tell "he was slow" from "the link was
    slow", and neither can he."""
    status, out = ask(server.srv, "what time is it")
    assert status == 200
    assert isinstance(out["ms"], int) and out["ms"] >= 0
    assert out["ms"] < 20000
    status, out = ask(server.srv, "diagnostics")
    assert isinstance(out["ms"], int), "a read is timed too"


def test_the_page_holds_the_path_open_between_questions(server):
    """A ping every 20 s while the page is in front of him keeps this
    server's keep-alive (30 s) and the tailnet's direct path alive, so the
    first question after a pause does not pay for a relay."""
    body = call(server.srv, "GET", "/")[1].decode("utf-8")
    assert "setInterval(beat, BEAT_MS)" in body
    assert 'document.addEventListener("visibilitychange"' in body
    assert "var BEAT_MS = 20000;" in body


# --------------------------------------------------- 15. through the app
def test_the_real_app_speaks_to_the_phone_and_not_to_the_room(build,  # noqa: F811
                                                              tmp_path):
    """The turn and its audio, both through the real JarvisApp: a reply on
    the phone's screen, the same reply in his ear, and a soundbar that
    never made a sound."""
    app = build()
    app.assistant.update({"phone.enabled": True, "phone.bind": "127.0.0.1",
                          "phone.port": 0,
                          "phone.link_file": str(tmp_path / "l.txt"),
                          "phone.qr_file": str(tmp_path / "l.svg")})
    app.start_assistant()
    try:
        status, out = call(app.webapp, "POST", "/api/ask",
                           json.dumps({"text": "what time is it"}),
                           token=app.webapp.token, ctype="application/json")
        assert status == 200
        reply = [m["text"] for m in out["messages"] if m["kind"] == "reply"][0]
        status, head, body = audio(app.webapp, reply, token=app.webapp.token)
        assert status == 200 and body[:4] == b"RIFF"
        assert head["Content-Type"] == "audio/wav"
        assert app.tts.rendered == [reply]
        assert app.tts.spoken == [], "the room must not hear a phone turn"
    finally:
        app.stop_assistant()


def test_a_streamed_clip_leaves_the_connection_reusable(server):
    """A hand-framed chunked body has to terminate correctly or the next
    request on that connection parses audio as a request line. It matters
    here more than usual: holding ONE connection open across a turn is
    what keeps the tailnet from re-negotiating a path for every tap."""
    server.app.tts.rendition = lambda text: FakeRendition(
        text, cached=False, chunks=["one.", "two."])
    conn = http.client.HTTPConnection(server.srv.host, server.srv.port,
                                      timeout=20)
    try:
        for _ in range(2):
            conn.request("POST", "/api/say",
                         body=json.dumps({"text": "one. two."}),
                         headers={"Authorization": "Bearer " + server.srv.token,
                                  "Content-Type": "application/json"})
            resp = conn.getresponse()
            assert resp.status == 200
            assert resp.read()[:4] == b"RIFF"
        # …and a plain JSON call still works on the same socket afterwards
        conn.request("GET", "/api/ping",
                     headers={"Authorization": "Bearer " + server.srv.token})
        resp = conn.getresponse()
        assert resp.status == 200
        assert json.loads(resp.read())["ok"] is True
    finally:
        conn.close()


# ------------------------------------- 16. audio that fails where he can see
# He reported "Jarvis still doesn't talk on the phone" twice, and both times
# the server logs showed the clip going out (`phone say: ... 277036 bytes`).
# Audio that is fetched and then silently dropped -- by a muted phone, by a
# context iOS parked while the screen was off -- leaves him nothing to report
# but the silence. These tests are about the page SAYING what happened.
def page_of(server):
    status, body = call(server.srv, "GET", "/")
    assert status == 200
    return body.decode("utf-8")


def test_the_page_leaves_the_ambient_category_so_the_side_switch_cannot_mute(
        server):
    """The leading suspect, and the one that fits exactly: on iOS a bare
    AudioContext plays in the "ambient" category, which the hardware side
    switch silences -- the fetch succeeds, the PCM is scheduled, state says
    "running", and nothing is heard. `navigator.audioSession.type =
    "playback"` (Safari 16.4+) is what leaves that category, and it has to
    be claimed BEFORE the context makes a sound."""
    page = page_of(server)
    assert "navigator.audioSession" in page, "the claim must be made at all"
    assert 's.type = "playback";' in page
    claim = page.index("function claimSession")
    unlock = page.index("function unlock")
    call_site = page.index("claimSession();", unlock)
    make = page.index("actx = new Ctx()", unlock)
    assert claim < unlock, "declared before it is used"
    assert call_site < make, "claimed before the context exists"


def test_an_interrupted_context_is_resumed_rather_than_walked_past(server):
    """iOS parks a backgrounded context in WebKit's own "interrupted"
    state, not "suspended". A resume guarded on "suspended" alone walks
    straight past it and schedules the next reply into a dead context."""
    page = page_of(server)
    assert 'if (actx.state !== "running" && actx.resume)' in page
    assert 'actx.state === "suspended" && actx.resume' not in page, \
        "the narrow test is exactly the bug"
    assert 'actx.state === "closed"' in page, "a closed context is rebuilt"


def test_coming_back_to_the_phone_re_arms_the_audio(server):
    """Nothing used to wake the context when he returned to the tab or
    came back through the back-forward cache, so the first reply after the
    lock screen was silent."""
    page = page_of(server)
    beat = page.index("function beat()")
    assert "if (voiceOn) { unlock(); }" in page[beat:], \
        "visibilitychange must re-arm the audio, not only ping"
    assert 'window.addEventListener("pageshow"' in page
    assert "if (e.persisted && voiceOn) { hush(); unlock(); }" in page


def test_a_clip_that_makes_no_sound_says_so_instead_of_going_quiet(server):
    """The whole point. Three different silences, three different lines --
    an empty render, bytes that were never scheduled, and a context that is
    not running -- so the next report is a diagnosis."""
    page = page_of(server)
    assert "function report(probe, got, secs)" in page
    assert "report(probe, got, secs);" in page, "and every clip ends in it"
    assert "There was nothing in that one to say aloud." in page
    assert "this phone played none of it." in page
    assert "This phone is holding audio back: it is '" in page
    assert "var got = 0, secs = 0;" in page, "the seconds must be counted"
    assert "secs += buf.duration;" in page, "and only when PCM is scheduled"


def test_the_page_tells_him_the_phone_itself_may_be_muted(server):
    """On a browser too old to claim the media session there is nothing
    the code can do about the side switch -- so it says which case he is
    in, in his own terms, rather than playing to a muted speaker."""
    page = page_of(server)
    assert "your phone is muted: the side switch " in page
    assert "silences this kind of audio" in page
    assert "the side switch CAN mute it." in page
    assert "switch cannot mute it." in page


def test_the_sound_check_is_one_tap_down_the_same_path_as_a_reply(server):
    """A control he can prove the audio with WITHOUT asking a question
    first, through the same endpoint, cache, RIFF walk and scheduler."""
    page = page_of(server)
    assert 'id="check"' in page and ">Test<" in page
    assert 'checkBtn.addEventListener("click"' in page
    assert "fetchSay(CHECK_LINE, true)" in page, \
        "the same function a reply uses, flagged so it always reports"
    assert json.dumps(wa.SOUND_CHECK) in page
    # it lives with the other two switches, not bolted on somewhere else
    switches = page.index('id="switches"')
    assert page.index('id="check"') > switches
    assert page.index('id="check"') < page.index("</div>", switches)
    # ...without squeezing the room switch: a third of a 375 px row would
    # ellipsise "Aloud in the room", so the two audio controls take only
    # the width of their own words.
    assert "#voice, #check { flex: none; }" in page


def test_the_sound_check_line_really_renders_on_this_box(server):
    """A test button that 503s is worse than none: the line has to be one
    the speech path will actually say."""
    status, head, body = audio(server.srv, wa.SOUND_CHECK)
    assert status == 200
    assert body[:4] == b"RIFF" and body[8:12] == b"WAVE"
    assert server.app.tts.rendered == [wa.SOUND_CHECK]


def test_nothing_new_on_the_page_needs_a_relaxed_policy(server):
    """The CSP is the same class of bug as the one being fixed: an inline
    handler or an external asset is refused SILENTLY. Everything added
    here is an addEventListener on markup that is already served."""
    conn = http.client.HTTPConnection(server.srv.host, server.srv.port,
                                      timeout=20)
    try:
        conn.request("GET", "/")
        resp = conn.getresponse()
        page = resp.read().decode("utf-8")
        csp = resp.getheader("Content-Security-Policy")
    finally:
        conn.close()
    assert csp == wa.CSP
    assert "default-src 'none'" in csp
    assert "script-src 'unsafe-inline'" in csp and "https:" not in csp
    assert "connect-src 'self'" in csp
    assert 'src="http' not in page and 'href="http' not in page
    assert "<script src" not in page
    for handler in ("onclick=", "onchange=", "onload=", "onsubmit="):
        assert handler not in page, f"{handler} would be blocked in silence"


# ------------------------------------------------- 10. the address book
# jarvis/contacts.py's third way in. It rides the SAME server: the enabled
# switch, the private-bind rule, the key, the limiter and the CSP are all
# inherited, and every test here proves one of them still applies. Every
# name and address is made up; the book lives in tmp_path.
import stat as _stat                                              # noqa: E402

from jarvis import contacts as contacts_mod                       # noqa: E402


@pytest.fixture
def book(tmp_path, monkeypatch):
    path = tmp_path / "cfg" / "contacts.json"
    monkeypatch.setenv(contacts_mod.ENV_VAR, str(path))
    monkeypatch.setattr(contacts_mod, "_BOOK", None)
    yield path
    monkeypatch.setattr(contacts_mod, "_BOOK", None)


def contacts_post(srv, payload, token=None):
    return call(srv, "POST", "/api/contacts", json.dumps(payload),
                token=srv.token if token is None else token,
                ctype="application/json")


def test_the_book_needs_the_key_like_everything_else(server, book):
    for method, body in (("GET", None),
                         ("POST", '{"op":"add","name":"X","email":"x@example.com"}')):
        status, out = call(server.srv, method, "/api/contacts", body,
                           ctype="application/json")
        assert status == 401 and "key" in out["error"], method
    assert not book.exists(), "nothing is written without the key"
    status, _ = contacts_post(server.srv, {"op": "add", "name": "X",
                                           "email": "x@example.com"},
                              token="not-the-key")
    assert status == 401 and not book.exists()


def test_a_public_peer_is_refused_before_the_book_is_touched(server, book,
                                                              monkeypatch):
    monkeypatch.setattr(wa._Handler, "_peer", lambda self: "8.8.8.8")
    status, out = call(server.srv, "GET", "/api/contacts", token=server.srv.token)
    assert status == 403
    status, _ = contacts_post(server.srv, {"op": "add", "name": "X",
                                           "email": "x@example.com"})
    assert status == 403 and not book.exists()


def test_the_book_page_binds_by_the_same_rule_as_the_phone_page(tmp_path):
    """There is no second server: the page is a route on PhoneServer, so a
    public bind refuses to start exactly as before."""
    cfg = make_config(tmp_path, bind="0.0.0.0")
    srv = wa.PhoneServer(PhoneApp(cfg), cfg=cfg)
    assert srv.start() is False and not srv.running
    assert "/contacts" in wa._Handler.do_GET.__code__.co_consts


def test_the_book_shell_carries_no_data(server, book):
    contacts_post(server.srv, {"op": "add", "name": "Perpetua Quill",
                               "email": "perpetua@example.net"})
    status, body = call(server.srv, "GET", "/contacts")
    assert status == 200
    page = body.decode("utf-8")
    assert "<title>Jarvis address book</title>" in page
    assert server.srv.token not in page
    assert "Perpetua" not in page and "Quill" not in page
    assert "@" not in page, "no address of any kind, not even a placeholder"
    assert "<form" not in page, "CSP form-action is 'none'; a submit fails in silence"
    for handler in ("onclick=", "onchange=", "onload=", "onsubmit="):
        assert handler not in page
    assert 'src="http' not in page and 'href="http' not in page


def test_add_then_list_round_trips_and_the_file_is_his(server, book):
    status, out = contacts_post(server.srv, {
        "op": "add", "name": "Heather Smith", "email": "heather@example.com",
        "honorific": "Dr", "aliases": ["my advisor"], "note": "PhD advisor"})
    assert status == 200 and out["ok"] is True, out
    assert out["contacts"] == [{"name": "Heather Smith", "email": "heather@example.com",
                                "honorific": "Dr", "aliases": ["my advisor"],
                                "note": "PhD advisor"}]
    assert out["path"] == contacts_mod.display_path(book)
    assert _stat.S_IMODE(book.stat().st_mode) == 0o600
    status, out = call(server.srv, "GET", "/api/contacts", token=server.srv.token)
    assert status == 200 and out["contacts"][0]["email"] == "heather@example.com"
    assert out["skipped"] == []
    # and the send lane sees it: same resolver, same file
    assert contacts_mod.current().resolve("my advisor").addr == "heather@example.com"


def test_a_bad_address_is_a_400_and_the_file_is_unchanged(server, book):
    contacts_post(server.srv, {"op": "add", "name": "Heather Smith",
                               "email": "heather@example.com"})
    before = book.read_text()
    for row in ({"name": "Dana Ruiz", "email": "dana at example dot com"},
                {"name": "Dana Ruiz", "email": "dana@example.com junk"},
                {"name": "dana@example.com", "email": "dana@example.com"},
                {"name": "heather smith", "email": "other@example.com"},
                {"name": "Dana Ruiz", "email": "HEATHER@example.com"}):
        status, out = contacts_post(server.srv, dict(row, op="add"))
        assert status == 400 and out["error"], row
    assert book.read_text() == before
    status, out = contacts_post(server.srv, {"op": "edit"})
    assert status == 400 and "op" in out["error"]


def test_remove_needs_the_exact_name_and_the_exact_address(server, book):
    contacts_post(server.srv, {"op": "add", "name": "Heather Smith",
                               "email": "heather@example.com"})
    status, out = contacts_post(server.srv, {"op": "remove", "name": "Heather Smith",
                                             "email": "stale@example.com"})
    assert status == 400 and "not at that address" in out["error"]
    status, out = contacts_post(server.srv, {"op": "remove", "name": "Heather"})
    assert status == 400
    status, out = contacts_post(server.srv, {"op": "remove", "name": "Heather",
                                             "email": "heather@example.com"})
    assert status == 400 and "nothing in the book" in out["error"]
    assert len(json.loads(book.read_text())["contacts"]) == 1
    status, out = contacts_post(server.srv, {"op": "remove", "name": "Heather Smith",
                                             "email": "heather@example.com"})
    assert status == 200 and out["contacts"] == []
    assert json.loads(book.read_text())["contacts"] == []


def test_an_oversized_book_request_is_refused_before_it_is_read(server, book):
    big = {"op": "add", "name": "X", "email": "x@example.com",
           "note": "n" * (wa.MAX_TEXT_BYTES + 10)}
    status, out = contacts_post(server.srv, big)
    assert status == 413 and "cap" in out["error"]
    assert not book.exists()


def test_a_hand_edit_between_two_page_loads_is_reflected(server, book):
    contacts_post(server.srv, {"op": "add", "name": "Heather Smith",
                               "email": "heather@example.com"})
    _, first = call(server.srv, "GET", "/api/contacts", token=server.srv.token)
    assert [c["name"] for c in first["contacts"]] == ["Heather Smith"]
    data = json.loads(book.read_text())
    data["contacts"].append({"name": "Dana Ruiz", "email": "dana@example"})
    data["contacts"].append({"name": "Mum", "email": "linda@example.com"})
    book.write_text(json.dumps(data))
    _, second = call(server.srv, "GET", "/api/contacts", token=server.srv.token)
    assert [c["name"] for c in second["contacts"]] == ["Heather Smith", "Mum"]
    assert second["skipped"] == [{"index": 1, "name": "Dana Ruiz",
                                  "why": second["skipped"][0]["why"]}]
    assert "bad address" in second["skipped"][0]["why"]
    # a page write after the hand edit keeps the hand edit
    contacts_post(server.srv, {"op": "remove", "name": "Mum",
                               "email": "linda@example.com"})
    names = [r["name"] for r in json.loads(book.read_text())["contacts"]]
    assert names == ["Heather Smith", "Dana Ruiz"]


def test_a_book_write_is_a_file_edit_not_a_turn(server, book):
    contacts_post(server.srv, {"op": "add", "name": "Heather Smith",
                               "email": "heather@example.com"})
    call(server.srv, "GET", "/api/contacts", token=server.srv.token)
    assert server.app.calls == [], "never through dispatch_text"
    assert server.app.tts.spoken == []


def test_the_phone_page_links_to_the_book(server):
    _, body = call(server.srv, "GET", "/")
    page = body.decode("utf-8")
    assert 'href="/contacts"' in page and "Address book" in page
    assert "/contacts" not in wa.MANIFEST, "no manifest entry, no home-screen icon"
