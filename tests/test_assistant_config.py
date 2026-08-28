"""Tests for jarvis.assistant_config — the ~/.config/jarvis/assistant.json
loader (spec 2026-08-26 section 10).

Firewall: every test points JARVIS_ASSISTANT_CONFIG at a tmp path and HOME
at a tmp dir; the module-level fixture asserts the user's real
~/.config/jarvis and /tmp/vss_voice are untouched afterwards.  No network,
no Ollama, no Claude.
"""
import json
import os
import stat
import threading
from pathlib import Path

import pytest

import jarvis.assistant_config as ac
from jarvis.assistant_config import (DEFAULTS, MASK, SECRET_KEYS, SECTIONS,
                                     AssistantConfig)

REAL_CONFIG_DIR = Path(os.path.expanduser("~")) / ".config" / "jarvis"
LIVE = Path("/tmp/vss_voice")


def _snapshot(path: Path):
    try:
        st = path.stat()
        return (st.st_mtime_ns, sorted(p.name for p in path.iterdir())
                if path.is_dir() else st.st_size)
    except FileNotFoundError:
        return None


def _queue_bytes():
    """Content of the live speak queue — the /tmp/vss_voice file a stray test
    would actually poison (the running app would speak whatever landed there)."""
    try:
        return (LIVE / "speak_queue.txt").read_bytes()
    except OSError:
        return None


def _log_size():
    try:
        return (LIVE / "jarvis.log").stat().st_size
    except OSError:
        return None


@pytest.fixture(scope="module", autouse=True)
def _real_config_untouched():
    """The suite must not touch the user's real config or the LIVE app's dir.

    2026-08-27: this used to compare a (mtime_ns, size) snapshot of
    /tmp/vss_voice/jarvis.log for equality, which cannot tell "a test wrote"
    from "the running Jarvis app wrote" — the live app appends every few
    seconds, so the teardown assert failed at random in full-suite runs.  The
    intent is kept with two checks that only a test can trip: the speak queue's
    CONTENT must be unchanged, and jarvis.log may only grow by appends (a test
    rewriting or truncating it shrinks the file)."""
    before = _snapshot(REAL_CONFIG_DIR)
    queue_before = _queue_bytes()
    log_before = _log_size()
    yield
    assert _snapshot(REAL_CONFIG_DIR) == before, "test touched the real ~/.config/jarvis"
    assert _queue_bytes() == queue_before, "test wrote to the live speak queue"
    log_after = _log_size()
    if log_before is not None and log_after is not None:
        assert log_after >= log_before, "test rewrote /tmp/vss_voice/jarvis.log"


@pytest.fixture
def cfg_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    path = tmp_path / "cfg" / "assistant.json"
    monkeypatch.setenv(ac.ENV_VAR, str(path))
    monkeypatch.setattr(ac, "_claude_bin", lambda: "/fake/bin/claude")
    return path


@pytest.fixture
def cfg(cfg_path):
    return AssistantConfig.load()


# ------------------------------------------------------------ creation
def test_load_creates_file_with_placeholders_and_mode_600(cfg_path):
    assert not cfg_path.exists()
    cfg = AssistantConfig.load()
    assert cfg.path == cfg_path
    assert cfg_path.exists()
    mode = stat.S_IMODE(cfg_path.stat().st_mode)
    assert mode == 0o600, oct(mode)
    on_disk = json.loads(cfg_path.read_text())
    assert on_disk == DEFAULTS
    # placeholders, never a real value
    assert on_disk["icloud"]["app_password"] == ""
    assert on_disk["gmail"]["app_password"] == ""
    assert on_disk["discord"]["bot_token"] == ""
    assert on_disk["home_location"] == {"city": "", "region": "", "lat": None, "lon": None}


def test_defaults_match_spec_10_1():
    assert DEFAULTS["version"] == 1
    assert DEFAULTS["user"] == {"name": "Hunter"}
    assert DEFAULTS["units"] == "us"
    assert DEFAULTS["local_model"] == "gemma4:26b"
    assert DEFAULTS["location_lookup"] is True
    assert DEFAULTS["google_ical_urls"] == []
    assert DEFAULTS["icloud"]["url"] == "https://caldav.icloud.com"
    assert DEFAULTS["gmail"]["imap_host"] == "imap.gmail.com"
    c = DEFAULTS["claude"]
    assert c["allowed_dirs"] == ["/home/hunterp/Jarvis", "/home/hunterp/haymaker-digest"]
    assert c["projects_root"] == "/home/hunterp/projects"
    assert c["permission_mode"] == "acceptEdits"
    assert c["dangerously_skip_permissions"] is False
    # user decision 2026-08-26: auto-approve outside allowed_dirs too
    assert c["auto_approve_anywhere"] is True
    # CLI contract verified against 2.1.247 on 2026-08-27, so it is on
    assert c["permission_prompt_tool"] is True
    assert (c["model"], c["big_model"], c["fast_mode"], c["effort"]) == \
        ("opus", "fable", False, "")
    assert len(c["skill_phrases"]) == 6
    assert c["skill_phrases"]["^run a ralph loop on (.+)$"] == "/ralph-loop $1"
    assert c["skill_phrases"]["^plan a feature (.+)$"] == "/feature-dev $1"
    b = DEFAULTS["briefing"]
    assert b["enabled"] is False and b["hn_items"] == 3
    assert b["news_feeds"] == ["https://www.theverge.com/rss/index.xml",
                               "https://feeds.arstechnica.com/arstechnica/index"]
    assert b["sports_feeds"] == [] and b["stock_symbols"] == []
    assert DEFAULTS["alarms"] == {"sound": "", "volume": 0.8, "escalate": True,
                                  "max_ring_s": 300, "snooze_min": 10}
    assert DEFAULTS["discord"] == {"bot_token": "", "channel_id": "", "user_id": ""}
    assert DEFAULTS["autostart"] == {"enabled": False}
    assert SECRET_KEYS == ("icloud.app_password", "gmail.app_password", "discord.bot_token")


def test_defaults_are_not_shared_with_instances(cfg):
    cfg.set("alarms.snooze_min", 99)
    assert DEFAULTS["alarms"]["snooze_min"] == 10
    assert AssistantConfig().get("alarms.snooze_min") == 10


def test_env_override_wins_over_default_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    env_path = tmp_path / "env" / "a.json"
    monkeypatch.setenv(ac.ENV_VAR, str(env_path))
    assert ac.config_path() == env_path
    cfg = AssistantConfig.load()
    assert cfg.path == env_path and env_path.exists()
    assert not (tmp_path / "home" / ".config" / "jarvis").exists()


def test_default_path_is_under_home_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(ac.ENV_VAR, raising=False)
    assert ac.config_path() == tmp_path / ".config" / "jarvis" / "assistant.json"
    cfg = AssistantConfig.load()
    assert cfg.path == tmp_path / ".config" / "jarvis" / "assistant.json"
    assert cfg.path.exists()
    assert stat.S_IMODE(cfg.path.stat().st_mode) == 0o600


def test_explicit_path_arg_wins_over_env(tmp_path, cfg_path):
    other = tmp_path / "other.json"
    cfg = AssistantConfig.load(other)
    assert cfg.path == other and other.exists()
    assert not cfg_path.exists()


def test_tilde_in_env_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(ac.ENV_VAR, "~/x/assistant.json")
    assert ac.config_path() == tmp_path / "x" / "assistant.json"


def test_load_keeps_user_values_and_fills_new_keys(cfg_path):
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(json.dumps({"units": "metric", "user": {"name": "H"},
                                    "custom_key": {"kept": True},
                                    "claude": {"model": "sonnet"}}))
    cfg = AssistantConfig.load()
    assert cfg.get("units") == "metric"
    assert cfg.get("user.name") == "H"
    assert cfg.get("custom_key.kept") is True
    assert cfg.get("claude.model") == "sonnet"
    assert cfg.get("claude.big_model") == "fable"           # filled from DEFAULTS
    assert cfg.get("alarms.snooze_min") == 10
    on_disk = json.loads(cfg_path.read_text())
    assert on_disk["alarms"]["snooze_min"] == 10            # written back
    assert on_disk["custom_key"] == {"kept": True}
    assert on_disk["claude"]["model"] == "sonnet"


def test_load_tightens_loose_mode(cfg_path):
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(json.dumps(DEFAULTS))
    os.chmod(cfg_path, 0o644)
    AssistantConfig.load()
    assert stat.S_IMODE(cfg_path.stat().st_mode) == 0o600


def test_load_never_raises_on_unwritable_dir(tmp_path, monkeypatch):
    if os.geteuid() == 0:
        pytest.skip("root ignores directory modes")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    locked = tmp_path / "locked"
    locked.mkdir()
    os.chmod(locked, 0o500)
    try:
        monkeypatch.setenv(ac.ENV_VAR, str(locked / "assistant.json"))
        cfg = AssistantConfig.load()
        assert cfg.get("local_model") == "gemma4:26b"
        assert cfg.save() is False
        assert cfg.set("units", "metric") is False
        assert cfg.get("units") == "metric"       # in memory still works
    finally:
        os.chmod(locked, 0o700)


# --------------------------------------------------------- corrupt file
@pytest.mark.parametrize("content", ["{not json", "", "[1, 2, 3]", '"a string"', "\xff\xfe"])
def test_corrupt_file_is_moved_aside_and_recreated(cfg_path, content):
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(content, encoding="utf-8", errors="surrogateescape") \
        if content != "\xff\xfe" else cfg_path.write_bytes(b"\xff\xfe\x00")
    cfg = AssistantConfig.load()
    bad = cfg_path.with_name("assistant.json.bad")
    assert bad.exists(), "corrupt file must be preserved as .bad"
    assert cfg_path.exists()
    assert json.loads(cfg_path.read_text()) == DEFAULTS
    assert stat.S_IMODE(cfg_path.stat().st_mode) == 0o600
    assert cfg.get("version") == 1


# ------------------------------------------------------------- get/set
def test_get_dotted_and_defaults(cfg):
    assert cfg.get("claude.model") == "opus"
    assert cfg.get("claude.skill_phrases.^commit (this|it|that)$") == "/commit"
    assert cfg.get("nope") is None
    assert cfg.get("nope.deeper", "dflt") == "dflt"
    assert cfg.get("claude.model.too_deep", 7) == 7
    assert cfg.get("units.x") is None


def test_get_returns_copies(cfg):
    dirs = cfg.get("claude.allowed_dirs")
    dirs.append("/evil")
    assert "/evil" not in cfg.get("claude.allowed_dirs")
    sec = cfg.get("alarms")
    sec["volume"] = 0
    assert cfg.get("alarms.volume") == 0.8


def test_set_saves_atomically_with_mode_600(cfg, cfg_path):
    assert cfg.set("briefing.enabled", True) is True
    assert cfg.set("new.section.key", [1, 2]) is True
    on_disk = json.loads(cfg_path.read_text())
    assert on_disk["briefing"]["enabled"] is True
    assert on_disk["new"]["section"]["key"] == [1, 2]
    assert stat.S_IMODE(cfg_path.stat().st_mode) == 0o600
    leftovers = [p for p in cfg_path.parent.iterdir() if p.name != "assistant.json"]
    assert leftovers == [], f"temp files left behind: {leftovers}"


def test_set_through_a_scalar_makes_a_section(cfg):
    cfg.set("units.sub", 1)
    assert cfg.get("units") == {"sub": 1}


def test_update_many_one_save(cfg, cfg_path, monkeypatch):
    calls = []
    real = ac._write_private
    monkeypatch.setattr(ac, "_write_private", lambda p, t: calls.append(p) or real(p, t))
    cfg.update({"alarms.volume": 0.5, "user.name": "Hunter P"})
    assert len(calls) == 1
    on_disk = json.loads(cfg_path.read_text())
    assert on_disk["alarms"]["volume"] == 0.5 and on_disk["user"]["name"] == "Hunter P"


def test_save_is_thread_safe(cfg):
    def worker(i):
        for j in range(20):
            cfg.set(f"t.{i}", j)
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    on_disk = json.loads(cfg.path.read_text())
    assert on_disk["t"] == {str(i): 19 for i in range(4)}


def test_attribute_views_are_live(cfg):
    assert cfg.units == "us"
    assert cfg.google_ical_urls == []
    assert cfg.claude.model == "opus"
    assert cfg.claude.skill_phrases["^security review$"] == "/security-review"
    assert cfg.alarms.snooze_min == 10
    assert cfg.discord.bot_token == ""
    cfg.set("alarms.snooze_min", 5)
    assert cfg.alarms.snooze_min == 5
    cfg.alarms.volume = 0.3                    # assignment saves
    assert json.loads(cfg.path.read_text())["alarms"]["volume"] == 0.3
    assert "volume" in cfg.alarms and cfg.alarms["volume"] == 0.3
    assert cfg.alarms.get("missing", "d") == "d"
    assert cfg.alarms.as_dict()["escalate"] is True
    with pytest.raises(AttributeError):
        cfg.alarms.no_such_key
    with pytest.raises(AttributeError):
        cfg.no_such_section
    assert "•••" not in repr(cfg.alarms)


# ------------------------------------------------------------- reload
def test_reload_if_changed(cfg, cfg_path):
    assert cfg.reload_if_changed() is False
    data = json.loads(cfg_path.read_text())
    data["units"] = "metric"
    cfg_path.write_text(json.dumps(data))
    os.utime(cfg_path, ns=(cfg_path.stat().st_atime_ns, cfg_path.stat().st_mtime_ns + 10_000_000))
    assert cfg.reload_if_changed() is True
    assert cfg.get("units") == "metric"
    assert cfg.get("alarms.snooze_min") == 10          # defaults still merged
    assert cfg.reload_if_changed() is False


def test_reload_keeps_memory_copy_when_edit_is_corrupt(cfg, cfg_path):
    cfg.set("units", "metric")
    cfg_path.write_text("{oops")
    os.utime(cfg_path, ns=(cfg_path.stat().st_atime_ns, cfg_path.stat().st_mtime_ns + 10_000_000))
    assert cfg.reload_if_changed() is False
    assert cfg.get("units") == "metric"
    assert cfg.reload_if_changed() is False              # not re-reported


# ------------------------------------------------- sections / setup lines
def test_nothing_configured_on_fresh_file_except_claude(cfg):
    assert cfg.is_configured("claude") is True          # binary faked + dirs
    for section in SECTIONS:
        if section != "claude":
            assert cfg.is_configured(section) is False, section
    assert cfg.missing_sections() == [s for s in SECTIONS if s != "claude"]
    assert cfg.is_configured("unknown") is False


@pytest.mark.parametrize("section, expect", [
    ("google_ical", "Google calendar link"),
    ("icloud", "iCloud calendar"),
    ("gmail", "Gmail app password"),
    ("discord", "Discord bot"),
    ("claude", "Claude command line"),
    ("home_location", "home location"),
])
def test_setup_lines_are_persona(cfg, section, expect):
    line = cfg.setup_line(section)
    assert line.startswith("I'll need ")
    assert expect in line
    assert " sir;" in line
    assert line.endswith("the notes are in docs/assistant-setup.md.")
    assert line.count(".") == 2         # one sentence: ".md" + the full stop
    assert "\n" not in line and "*" not in line


def test_setup_line_google_matches_spec_example(cfg):
    assert cfg.setup_line("google_ical") == ("I'll need your Google calendar link set up, sir; "
                                             "the notes are in docs/assistant-setup.md.")


def test_setup_line_unknown_section_still_persona(cfg):
    assert cfg.setup_line("spotify") == \
        "I'll need spotify set up, sir; the notes are in docs/assistant-setup.md."
    assert set(AssistantConfig.setup_lines()) == {cfg.setup_line(s) for s in SECTIONS}


def test_home_location_configured_and_property(cfg):
    assert cfg.home_location is None
    cfg.set("home_location.lat", 41.88)
    assert cfg.is_configured("home_location") is False   # lon still missing
    cfg.set("home_location.lon", -87.63)
    cfg.set("home_location.city", "Chicago")
    assert cfg.is_configured("home_location") is True
    assert cfg.home_location == {"city": "Chicago", "region": "", "lat": 41.88, "lon": -87.63}
    cfg.set("home_location.lat", "41.88")                 # a string is not a coordinate
    assert cfg.home_location is None


def test_google_ical_configured(cfg):
    cfg.set("google_ical_urls", ["", "<paste the secret address here>"])
    assert cfg.is_configured("google_ical") is False
    cfg.set("google_ical_urls", ["https://calendar.google.com/calendar/ical/x/private-abc/basic.ics"])
    assert cfg.is_configured("google_ical") is True
    cfg.set("google_ical_urls", "not-a-list")
    assert cfg.is_configured("google_ical") is False


@pytest.mark.parametrize("placeholder", ["", "   ", "<app password>", "PASTE-HERE",
                                         "your-app-password", "changeme", "xxxx", None])
def test_placeholders_do_not_count(cfg, placeholder):
    cfg.set("icloud.apple_id", "hunter@example.com")
    cfg.set("icloud.app_password", placeholder)
    assert cfg.is_configured("icloud") is False
    cfg.set("icloud.app_password", "abcd-efgh-ijkl-mnop")
    assert cfg.is_configured("icloud") is True


def test_gmail_discord_claude_configured(cfg, monkeypatch):
    cfg.update({"gmail.address": "h@gmail.com", "gmail.app_password": "abcd efgh ijkl mnop"})
    assert cfg.is_configured("gmail") is True
    cfg.set("discord.bot_token", "MTIz.fake.token")
    assert cfg.is_configured("discord") is False          # channel id missing
    cfg.set("discord.channel_id", "123456789012345678")
    assert cfg.is_configured("discord") is True
    monkeypatch.setattr(ac, "_claude_bin", lambda: "")
    assert cfg.is_configured("claude") is False
    monkeypatch.setattr(ac, "_claude_bin", lambda: "/x/claude")
    cfg.set("claude.allowed_dirs", [])
    assert cfg.is_configured("claude") is False


def test_claude_bin_seam_reads_env(monkeypatch):
    monkeypatch.setenv("JARVIS_CLAUDE_BIN", "/opt/claude")
    assert ac._claude_bin() == "/opt/claude"


def test_claude_bin_falls_back_to_local_bin(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("JARVIS_CLAUDE_BIN", raising=False)
    monkeypatch.setattr(ac.shutil, "which", lambda name: None)     # not on PATH
    assert ac._claude_bin() == ""
    local = tmp_path / ".local" / "bin" / "claude"
    local.parent.mkdir(parents=True)
    local.write_text("#!/bin/sh\n")
    assert ac._claude_bin() == ""                                    # not executable yet
    local.chmod(0o755)
    assert ac._claude_bin() == str(local)


# ----------------------------------------------------------- redaction
def test_redacted_masks_secrets_only_when_set(cfg):
    assert cfg.redacted()["icloud"]["app_password"] == ""     # unset stays visible-empty
    cfg.update({"icloud.app_password": "s3cret-icloud", "gmail.app_password": "s3cret-gmail",
                "discord.bot_token": "s3cret-token", "gmail.address": "h@gmail.com"})
    red = cfg.redacted()
    assert red["icloud"]["app_password"] == MASK == "•••"
    assert red["gmail"]["app_password"] == MASK
    assert red["discord"]["bot_token"] == MASK
    assert red["gmail"]["address"] == "h@gmail.com"           # not a secret
    dumped = json.dumps(red)
    for secret in ("s3cret-icloud", "s3cret-gmail", "s3cret-token"):
        assert secret not in dumped
    # the real data is untouched
    assert cfg.get("discord.bot_token") == "s3cret-token"


def test_repr_and_scrub_never_leak(cfg):
    cfg.update({"discord.bot_token": "tok-ABC123", "gmail.app_password": "pw-XYZ"})
    text = repr(cfg) + str(cfg) + repr(cfg.discord) + repr(cfg.gmail)
    assert "tok-ABC123" not in text and "pw-XYZ" not in text
    assert MASK in text
    assert cfg.scrub("Authorization: Bot tok-ABC123 / pw-XYZ") == \
        f"Authorization: Bot {MASK} / {MASK}"
    assert cfg.scrub(None) == ""
    assert cfg.secret_values() == ["pw-XYZ", "tok-ABC123"] or \
        set(cfg.secret_values()) == {"pw-XYZ", "tok-ABC123"}


# ------------------------------------------------------------ properties
def test_simple_properties(cfg):
    assert cfg.local_model == "gemma4:26b"
    assert cfg.user_name == "Hunter"
    assert cfg.units == "us"
    assert cfg.allowed_dirs == ["/home/hunterp/Jarvis", "/home/hunterp/haymaker-digest"]
    assert cfg.projects_root == "/home/hunterp/projects"
    assert cfg.skill_phrases == DEFAULTS["claude"]["skill_phrases"]
    cfg.skill_phrases["^x$"] = "/x"                           # a copy
    assert "^x$" not in cfg.skill_phrases
    cfg.set("local_model", "")
    assert cfg.local_model == "gemma4:26b"


def test_allowed_dirs_expand_tilde_and_skip_junk(cfg, tmp_path, monkeypatch):
    home = Path(os.environ["HOME"])
    cfg.set("claude.allowed_dirs", ["~/proj", "", 3, "/abs/dir/"])
    assert cfg.allowed_dirs == [str(home / "proj"), "/abs/dir"]


# ------------------------------------------------------ allowed paths
@pytest.fixture
def sandbox(cfg, tmp_path):
    allowed = tmp_path / "allowed"
    (allowed / "sub").mkdir(parents=True)
    (allowed / "sub" / "f.py").write_text("x")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("s")
    (allowed / "escape").symlink_to(outside)                  # symlink out
    (allowed / "escape_file").symlink_to(outside / "secret.txt")
    (tmp_path / "alias").symlink_to(allowed)                  # symlink in
    cfg.set("claude.allowed_dirs", [str(allowed)])
    return {"allowed": allowed, "outside": outside, "tmp": tmp_path}


def test_is_allowed_path_inside(cfg, sandbox):
    a = sandbox["allowed"]
    assert cfg.is_allowed_path(a) is True
    assert cfg.is_allowed_path(str(a) + "/") is True
    assert cfg.is_allowed_path(a / "sub" / "f.py") is True
    assert cfg.is_allowed_path(a / "sub" / "new_file.py") is True     # need not exist
    assert cfg.is_allowed_path(a / "sub" / ".." / "sub" / "f.py") is True
    assert cfg.is_allowed_path(sandbox["tmp"] / "alias" / "sub" / "f.py") is True  # via symlink in
    assert cfg.is_allowed_path("sub/f.py", base=a) is True             # relative to project


def test_is_allowed_path_outside_and_traversal(cfg, sandbox):
    a, tmp = sandbox["allowed"], sandbox["tmp"]
    assert cfg.is_allowed_path(sandbox["outside"] / "secret.txt") is False
    assert cfg.is_allowed_path(a / ".." / "outside" / "secret.txt") is False
    assert cfg.is_allowed_path(a / "sub" / ".." / ".." / "outside") is False
    assert cfg.is_allowed_path(a / "escape" / "secret.txt") is False    # symlink out
    assert cfg.is_allowed_path(a / "escape_file") is False
    assert cfg.is_allowed_path(Path(str(a) + "_sibling") / "x") is False  # prefix trick
    assert cfg.is_allowed_path(tmp) is False
    assert cfg.is_allowed_path("/") is False
    assert cfg.is_allowed_path("../outside/secret.txt", base=a) is False


def test_is_allowed_path_when_allowed_dir_is_a_symlink(cfg, sandbox):
    cfg.set("claude.allowed_dirs", [str(sandbox["tmp"] / "alias")])
    assert cfg.is_allowed_path(sandbox["allowed"] / "sub" / "f.py") is True
    assert cfg.is_allowed_path(sandbox["outside"]) is False


def test_is_allowed_path_with_no_dirs_or_tilde(cfg, tmp_path):
    cfg.set("claude.allowed_dirs", [])
    assert cfg.is_allowed_path(tmp_path) is False
    home = Path(os.environ["HOME"])
    (home / "p").mkdir(parents=True)
    cfg.set("claude.allowed_dirs", ["~/p"])
    assert cfg.is_allowed_path("~/p/x.py") is True
    assert cfg.is_allowed_path(home / "p2") is False


def test_add_and_remove_allowed_dir(cfg, cfg_path, tmp_path):
    new = tmp_path / "newproj"
    new.mkdir()
    assert cfg.add_allowed_dir(new) is True
    assert cfg.add_allowed_dir(str(new) + "/") is False                 # already there
    (tmp_path / "link").symlink_to(new)
    assert cfg.add_allowed_dir(tmp_path / "link") is False              # same real dir
    assert str(new) in json.loads(cfg_path.read_text())["claude"]["allowed_dirs"]
    assert cfg.is_allowed_path(new / "a.py") is True
    assert cfg.add_allowed_dir("~/viatilde") is True
    assert str(Path(os.environ["HOME"]) / "viatilde") in cfg.allowed_dirs
    assert cfg.remove_allowed_dir(tmp_path / "link") is True            # removes `new`
    assert str(new) not in cfg.allowed_dirs
    assert cfg.remove_allowed_dir(tmp_path / "link") is False


# -------------------------------------------------------- misc / hygiene
def test_constructor_without_path_is_memory_only():
    cfg = AssistantConfig({"units": "metric"})
    assert cfg.path is None
    assert cfg.get("units") == "metric" and cfg.get("claude.model") == "opus"
    assert cfg.save() is False and cfg.reload_if_changed() is False
    assert cfg.set("units", "us") is False and cfg.get("units") == "us"


def test_no_real_secret_strings_in_module_source():
    src = Path(ac.__file__).read_text()
    for marker in ("ghp_", "sk-ant", "xoxb-", "MTIz"):
        assert marker not in src


def test_gmail_counts_as_configured_when_accounts_list_is_used(tmp_path):
    """Multi-account configs have no top-level address/app_password, so the
    old check reported gmail unconfigured and the setup line kept appearing."""
    from jarvis.assistant_config import AssistantConfig
    cfg = AssistantConfig(data={"gmail": {"accounts": [
        {"label": "personal", "address": "me@gmail.com",
         "app_password": "aaaa bbbb cccc dddd"}]}})
    assert cfg.is_configured("gmail") is True


def test_gmail_with_only_empty_accounts_is_not_configured():
    from jarvis.assistant_config import AssistantConfig
    assert AssistantConfig(data={"gmail": {"accounts": []}}).is_configured("gmail") is False
    assert AssistantConfig(data={"gmail": {"accounts": [
        {"address": "", "app_password": ""}]}}).is_configured("gmail") is False
