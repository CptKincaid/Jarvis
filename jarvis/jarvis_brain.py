"""Jarvis Brain — hybrid intelligence with Ollama + Claude Peers.

Fast path: Ollama (local LLM) for simple questions, system commands, quick facts.
Smart path: Claude Peers (sends message to running Claude session with full context).

The Claude session receives messages via the claude-peers MCP network and responds
through the speak queue file.
"""

import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("/tmp/vss_voice")
SPEAK_QUEUE = Path("/tmp/vss_voice/speak_queue.txt")
PEERS_RESPONSE_FILE = Path("/tmp/vss_voice/claude_response.txt")

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:latest"

# Questions Ollama can handle locally (fast, no Claude needed)
LOCAL_PATTERNS = [
    "what time", "what day", "what date", "what's the weather",
    "what's my ip", "how long has", "uptime", "temperature",
    "check gpu", "check disk", "system status", "what's running",
    "git status", "what's changed", "am i online", "check network",
    "find file", "recent files", "count lines",
    "show clipboard", "show notes", "list notes",
]

JARVIS_SYSTEM = """You are Jarvis, a voice AI assistant (MCU Jarvis personality).
Keep responses under 2 sentences. Be direct, professional, slightly warm.
You run on a Linux desktop with 2x RTX 3090 GPUs.
User's name is Hunter. Current time: {time}.
Respond conversationally — this will be read aloud through TTS."""


def _log(msg):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"{ts} [Brain] {msg}"
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / "gui_debug.log", "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


class JarvisBrain:
    """Hybrid brain: Ollama (fast) + Claude Peers (smart)."""

    def __init__(self):
        self._busy = False
        self._history = []
        self._ollama_available = None

    def think(self, user_input, callback=None):
        """Route user input to the best backend."""
        if self._busy:
            if callback:
                callback([("SPEAK", "I'm still processing sir. One moment.")])
            return

        self._busy = True

        def _process():
            try:
                # Decide: local (Ollama) or smart (Claude)
                if self._is_local_question(user_input):
                    actions = self._query_ollama(user_input)
                else:
                    actions = self._query_claude_peers(user_input)

                self._history.append({"role": "user", "text": user_input})
                self._history.append({"role": "jarvis", "actions": actions})
                self._history = self._history[-20:]

                if callback:
                    callback(actions)
            except Exception as e:
                _log(f"Brain error: {e}")
                if callback:
                    callback([("SPEAK", f"I encountered an error. {str(e)[:50]}")])
            finally:
                self._busy = False

        threading.Thread(target=_process, daemon=True).start()

    def _is_local_question(self, text):
        """Check if this can be handled locally by Ollama."""
        lower = text.lower()
        # Simple questions, system commands, quick facts
        for pattern in LOCAL_PATTERNS:
            if pattern in lower:
                return True
        # Short conversational responses
        if len(lower.split()) <= 5:
            return True
        return False

    # ------------------------------------------------------------------
    # Ollama (fast, local)
    # ------------------------------------------------------------------
    def _query_ollama(self, user_input):
        """Query local Ollama for fast responses."""
        _log(f"Ollama query: {user_input[:60]}")

        system = JARVIS_SYSTEM.format(
            time=datetime.now().strftime("%I:%M %p, %A %B %d"),
        )

        try:
            import urllib.request
            req_data = json.dumps({
                "model": OLLAMA_MODEL,
                "prompt": user_input,
                "system": system,
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 100},
            }).encode()
            req = urllib.request.Request(
                f"{OLLAMA_URL}/api/generate",
                data=req_data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            text = data.get("response", "").strip()
            _log(f"Ollama response: {text[:80]}")

            if text:
                return self._parse_response(text)
            return [("SPEAK", "I couldn't generate a response locally.")]

        except Exception as e:
            _log(f"Ollama error: {e}")
            # Fall back to Claude
            return self._query_claude_peers(user_input)

    # ------------------------------------------------------------------
    # Claude Peers (smart, full context)
    # ------------------------------------------------------------------
    def _query_claude_peers(self, user_input):
        """Send message to the running Claude session via peers network."""
        _log(f"Claude Peers query: {user_input[:60]}")

        # Write the question to a request file
        request_file = Path("/tmp/vss_voice/claude_request.txt")
        response_file = PEERS_RESPONSE_FILE

        # Clear old response
        if response_file.exists():
            response_file.unlink()

        # Write request
        request_file.write_text(json.dumps({
            "text": user_input,
            "timestamp": datetime.now().isoformat(),
            "respond_to": str(response_file),
        }))

        _log("Waiting for Claude peer response...")

        # Also try claude -p as fallback with better context
        try:
            claude_bin = "/home/hunterp/.npm-global/bin/claude"
            prompt = (
                f"You are Jarvis. Respond in 1-2 sentences maximum. "
                f"Be concise and professional like MCU Jarvis. "
                f"User (Hunter) said: {user_input}"
            )
            result = subprocess.run(
                [claude_bin, "-p", "--output-format", "text"],
                input=prompt,
                capture_output=True, text=True,
                timeout=60,
                env={**os.environ,
                     "PATH": f"/home/hunterp/.npm-global/bin:{os.environ.get('PATH', '')}"},
            )
            response = result.stdout.strip()
            _log(f"Claude response: {response[:80]}")

            if response:
                return self._parse_response(response)
        except subprocess.TimeoutExpired:
            return [("SPEAK",
                     "That request is taking a while sir. Could you simplify it?")]
        except Exception as e:
            _log(f"Claude error: {e}")

        return [("SPEAK", "I wasn't able to process that request sir.")]

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------
    def _parse_response(self, response):
        """Parse response into structured actions."""
        actions = []
        has_structured = False

        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue
            for tag in ("SPEAK", "RUN", "TYPE", "CLICK", "WINDOW", "SILENT"):
                if line.startswith(f"[{tag}]"):
                    content = line[len(tag) + 2:].strip()
                    if content:
                        actions.append((tag, content))
                        has_structured = True
                    break

        if not has_structured and response.strip():
            # Clean markdown
            clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', response)
            clean = re.sub(r'`([^`]+)`', r'\1', clean)
            clean = re.sub(r'#{1,6}\s+', '', clean)
            clean = re.sub(r'https?://\S+', '', clean)
            clean = re.sub(r'\s+', ' ', clean).strip()
            # Truncate for speech
            if len(clean) > 250:
                cut = clean[:250].rfind('.')
                clean = clean[:cut + 1] if cut > 100 else clean[:250]
            actions.append(("SPEAK", clean))

        return actions

    @property
    def is_busy(self):
        return self._busy
