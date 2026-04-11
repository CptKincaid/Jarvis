"""Jarvis Brain — Claude-powered intelligence for voice assistant.

Routes user requests to Claude CLI, parses structured responses,
and executes actions + speaks results. Claude IS Jarvis's brain.

Response format from Claude:
  [SPEAK] text to say aloud
  [RUN] shell command to execute
  [TYPE] text to type into active window
  [CLICK] text to find and click on screen
  [WINDOW] window name to switch to
  [SILENT] text only shown in GUI, not spoken
"""

import json
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("/tmp/vss_voice")

SYSTEM_PROMPT = """You are Jarvis, an AI voice assistant running on a Linux desktop with 2x RTX 3090 GPUs.
You respond to voice commands from your user (Hunter). You are helpful, concise, and professional — like the MCU Jarvis.

You can control the desktop. Respond with structured commands:
- [SPEAK] text — read this aloud through TTS
- [RUN] command — execute this shell command
- [TYPE] text — type this into the active window
- [WINDOW] name — switch to this window
- [SILENT] text — show in GUI only, don't speak

Keep spoken responses SHORT (1-3 sentences). Be direct. Use [RUN] for anything that needs system info.
If the user asks something conversational, just [SPEAK] your response.
If they ask for system info, [RUN] the command first, then [SPEAK] a summary.

Current working directory: /home/hunterp/jarvis
Project: Jarvis AI Voice Assistant
User: Hunter"""


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
    """Claude-powered brain for Jarvis voice assistant."""

    def __init__(self):
        self._busy = False
        self._history = []  # Conversation history for context

    def think(self, user_input, callback=None):
        """Send user input to Claude and get structured response.

        Args:
            user_input: what the user said
            callback: function(actions) called with parsed actions on completion
        """
        if self._busy:
            if callback:
                callback([("SPEAK", "I'm still thinking about your last request sir.")])
            return

        self._busy = True

        def _process():
            try:
                actions = self._query_claude(user_input)
                self._history.append({"role": "user", "text": user_input})
                self._history.append({"role": "jarvis", "actions": actions})
                # Keep last 10 exchanges for context
                self._history = self._history[-20:]

                if callback:
                    callback(actions)
            except Exception as e:
                _log(f"Brain error: {e}")
                if callback:
                    callback([("SPEAK", f"I encountered an error sir. {str(e)[:50]}")])
            finally:
                self._busy = False

        threading.Thread(target=_process, daemon=True).start()

    def _query_claude(self, user_input):
        """Query Claude CLI and parse structured response."""
        # Build prompt with conversation context
        context = ""
        if self._history:
            recent = self._history[-6:]
            for entry in recent:
                if entry["role"] == "user":
                    context += f"User: {entry['text']}\n"
                else:
                    for action_type, action_data in entry.get("actions", []):
                        if action_type == "SPEAK":
                            context += f"Jarvis: {action_data}\n"

        prompt = f"{SYSTEM_PROMPT}\n\n"
        if context:
            prompt += f"Recent conversation:\n{context}\n"
        prompt += f"User just said: {user_input}\n\nRespond with structured commands:"

        _log(f"Querying Claude: {user_input[:60]}")

        try:
            result = subprocess.run(
                ["claude", "-p", "--output-format", "text"],
                input=prompt,
                capture_output=True, text=True,
                timeout=30,
                cwd="/home/hunterp/jarvis",
            )
            response = result.stdout.strip()
            _log(f"Claude response: {response[:100]}")
            return self._parse_response(response)
        except subprocess.TimeoutExpired:
            return [("SPEAK", "I took too long to think about that. Could you try again?")]
        except Exception as e:
            _log(f"Claude CLI error: {e}")
            return [("SPEAK", f"I had trouble processing that. {str(e)[:30]}")]

    def _parse_response(self, response):
        """Parse Claude's response into structured actions."""
        actions = []

        # Look for structured commands
        lines = response.split("\n")
        has_structured = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            for tag in ("SPEAK", "RUN", "TYPE", "CLICK", "WINDOW", "SILENT"):
                pattern = f"[{tag}]"
                if line.startswith(pattern):
                    content = line[len(pattern):].strip()
                    if content:
                        actions.append((tag, content))
                        has_structured = True
                    break

        # If no structured commands found, treat entire response as speech
        if not has_structured and response.strip():
            # Clean up markdown
            clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', response)
            clean = re.sub(r'`([^`]+)`', r'\1', clean)
            clean = re.sub(r'#{1,6}\s+', '', clean)
            # Truncate for speech
            if len(clean) > 300:
                cut = clean[:300].rfind('.')
                if cut > 150:
                    clean = clean[:cut + 1]
                else:
                    clean = clean[:300] + "..."
            actions.append(("SPEAK", clean))
            actions.append(("SILENT", response))

        return actions

    @property
    def is_busy(self):
        return self._busy
