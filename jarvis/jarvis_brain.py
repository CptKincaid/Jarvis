"""Jarvis Brain — hybrid intelligence with Context + Memory.

Tier 1: Local commands (handled by commander, not brain)
Tier 2: Ollama (fast, local, context-aware)
Tier 3: Claude CLI (deep reasoning, full context, autonomous tasks)

Both tiers receive rich context from ContextEngine and persistent
memory from JarvisMemory.
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

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:latest"
CLAUDE_BIN = str(Path.home() / ".npm-global" / "bin" / "claude")
_NPM_BIN_PATH = str(Path.home() / ".npm-global" / "bin")

# Simple questions Ollama can handle
LOCAL_PATTERNS = [
    "what time", "what day", "what date", "what's the weather",
    "what's my ip", "how long has", "uptime", "temperature",
    "how are you", "hello", "hey", "good morning", "good evening",
    "thank you", "thanks", "good night",
]

JARVIS_SYSTEM = """You are Jarvis, a voice AI assistant (MCU Jarvis personality).
Keep responses under 2 sentences. Be direct, professional, slightly warm.
You run on a Linux desktop with 2x RTX 3090 GPUs. User's name is Hunter.

{context}

Respond conversationally — this will be read aloud through TTS."""

CLAUDE_SYSTEM = """You are Jarvis, an AI voice assistant. Respond with structured commands:
[SPEAK] text — read aloud (max 2 sentences)
[RUN] command — execute shell command
[TYPE] text — type into active window
[WINDOW] name — switch to window
[SILENT] text — show in GUI only
[DONE] text — task complete, speak this

Be concise. [SPEAK] lines are read through TTS so keep them short.
For multi-step tasks, execute one step at a time.

{context}

User (Hunter) said: {input}"""


from jarvis.jarvis_logging import get_logger
_log = get_logger("BRAIN")


class JarvisBrain:
    """Hybrid brain: Ollama (fast) + Claude (smart), with context + memory."""

    def __init__(self):
        self._busy = False
        self._context = None
        self._memory = None
        self._init_modules()

    def _init_modules(self):
        """Lazy-init context and memory."""
        try:
            from jarvis.context import ContextEngine
            self._context = ContextEngine()
        except Exception as e:
            _log(f"Context init error: {e}")

        try:
            from jarvis.memory import JarvisMemory
            self._memory = JarvisMemory()
        except Exception as e:
            _log(f"Memory init error: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def think(self, user_input, callback=None):
        """Route user input to the best backend."""
        if self._busy:
            if callback:
                callback([("SPEAK", "I'm still processing sir. One moment.")])
            return

        self._busy = True

        def _process():
            try:
                if self._is_local_question(user_input):
                    actions = self._query_ollama(user_input)
                else:
                    actions = self._query_claude(user_input)

                # Log to memory
                spoken = " ".join(d for t, d in actions if t == "SPEAK")
                if self._memory:
                    self._memory.log_habit(user_input[:50])
                if self._context:
                    self._context.add_exchange(user_input, spoken)

                if callback:
                    callback(actions)
            except Exception as e:
                _log(f"Brain error: {e}")
                if callback:
                    callback([("SPEAK", f"I encountered an error. {str(e)[:40]}")])
            finally:
                self._busy = False

        threading.Thread(target=_process, daemon=True).start()

    def execute_autonomous(self, task_description, callback=None):
        """Execute a multi-step task autonomously.

        Queries Claude repeatedly, executing [RUN] commands and feeding
        results back until [DONE] or max steps reached.
        """
        if self._busy:
            if callback:
                callback([("SPEAK", "I'm still working on something else sir.")])
            return

        self._busy = True

        def _run():
            try:
                results = self._autonomous_loop(task_description, callback)
                _log(f"Autonomous task complete: {len(results)} steps")
            except Exception as e:
                _log(f"Autonomous error: {e}")
                if callback:
                    callback([("SPEAK", f"Task failed. {str(e)[:40]}")])
            finally:
                self._busy = False

        threading.Thread(target=_run, daemon=True).start()

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def _is_local_question(self, text):
        lower = text.lower()
        for pattern in LOCAL_PATTERNS:
            if pattern in lower:
                return True
        if len(lower.split()) <= 5:
            return True
        return False

    # ------------------------------------------------------------------
    # Tier 2: Ollama (fast, local)
    # ------------------------------------------------------------------
    def _query_ollama(self, user_input):
        _log(f"Ollama: {user_input[:60]}")

        ctx_text = ""
        if self._context:
            ctx = self._context.get_context("standard")
            ctx_text = self._context.format_for_prompt(ctx)

        mem_text = ""
        if self._memory:
            mem_text = self._memory.format_for_context()

        full_context = ctx_text
        if mem_text:
            full_context += f"\n{mem_text}"

        system = JARVIS_SYSTEM.format(context=full_context)

        try:
            import urllib.request
            req_data = json.dumps({
                "model": OLLAMA_MODEL,
                "prompt": user_input,
                "system": system,
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 150},
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
            _log(f"Ollama error: {e}, falling back to Claude")
            return self._query_claude(user_input)

    # ------------------------------------------------------------------
    # Tier 3: Claude CLI (deep reasoning)
    # ------------------------------------------------------------------
    def _query_claude(self, user_input):
        _log(f"Claude: {user_input[:60]}")

        ctx_text = ""
        if self._context:
            ctx = self._context.get_context("full")
            ctx_text = self._context.format_for_prompt(ctx)

        mem_text = ""
        if self._memory:
            mem_text = self._memory.format_for_context()
            sessions = self._memory.format_sessions_for_prompt()
            if sessions:
                mem_text += f"\n{sessions}"

        full_context = ctx_text
        if mem_text:
            full_context += f"\n\nMemory:\n{mem_text}"

        prompt = CLAUDE_SYSTEM.format(
            context=full_context,
            input=user_input,
        )

        try:
            result = subprocess.run(
                [CLAUDE_BIN, "-p", "--output-format", "text"],
                input=prompt,
                capture_output=True, text=True,
                timeout=120,
                env={**os.environ,
                     "PATH": f"{_NPM_BIN_PATH}:{os.environ.get('PATH', '')}"},
            )
            response = result.stdout.strip()
            _log(f"Claude response: {response[:80]}")

            if response:
                return self._parse_response(response)
        except subprocess.TimeoutExpired:
            return [("SPEAK",
                     "That request is taking too long sir. Could you simplify it?")]
        except Exception as e:
            _log(f"Claude error: {e}")

        return [("SPEAK", "I wasn't able to process that request sir.")]

    # ------------------------------------------------------------------
    # Autonomous multi-step execution
    # ------------------------------------------------------------------
    def _autonomous_loop(self, task, callback, max_steps=10):
        results = []

        for step in range(max_steps):
            ctx_text = ""
            if self._context:
                ctx = self._context.get_context("full")
                ctx_text = self._context.format_for_prompt(ctx)

            prompt = f"""You are Jarvis executing an autonomous task.

Task: {task}
Step {step + 1}/{max_steps}

Previous results:
{json.dumps(results[-3:], indent=2) if results else 'None yet'}

{ctx_text}

Respond with structured commands. Use [RUN] to execute shell commands.
Use [SPEAK] to update the user on progress.
When the task is complete, use [DONE] with a summary.
If something fails, use [SPEAK] to explain and suggest alternatives."""

            try:
                result = subprocess.run(
                    [CLAUDE_BIN, "-p", "--output-format", "text"],
                    input=prompt,
                    capture_output=True, text=True,
                    timeout=60,
                    env={**os.environ,
                         "PATH": f"{_NPM_BIN_PATH}:{os.environ.get('PATH', '')}"},
                )
                actions = self._parse_response(result.stdout.strip())
            except Exception as e:
                _log(f"Autonomous step {step} error: {e}")
                if callback:
                    callback([("SPEAK", f"Step {step + 1} failed. {str(e)[:30]}")])
                break

            for action_type, action_data in actions:
                if action_type == "DONE":
                    if callback:
                        callback([("SPEAK", action_data)])
                    return results
                elif action_type == "RUN":
                    _log(f"Auto-run: {action_data[:50]}")
                    try:
                        r = subprocess.run(
                            action_data, shell=True,
                            capture_output=True, text=True, timeout=30,
                        )
                        output = r.stdout.strip()[:500]
                        if r.returncode != 0:
                            output += f"\nSTDERR: {r.stderr.strip()[:200]}"
                        results.append({
                            "step": step, "command": action_data,
                            "output": output, "rc": r.returncode,
                        })
                    except Exception as e:
                        results.append({
                            "step": step, "command": action_data,
                            "output": f"ERROR: {e}", "rc": -1,
                        })
                elif action_type == "SPEAK":
                    if callback:
                        callback([("SPEAK", action_data)])

            time.sleep(0.5)

        # Max steps reached
        if callback:
            callback([("SPEAK",
                       f"Task completed after {len(results)} steps sir.")])
        return results

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------
    def _parse_response(self, response):
        actions = []
        has_structured = False

        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue
            for tag in ("SPEAK", "RUN", "TYPE", "CLICK", "WINDOW",
                        "SILENT", "DONE"):
                if line.startswith(f"[{tag}]"):
                    content = line[len(tag) + 2:].strip()
                    if content:
                        actions.append((tag, content))
                        has_structured = True
                    break

        if not has_structured and response.strip():
            clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', response)
            clean = re.sub(r'`([^`]+)`', r'\1', clean)
            clean = re.sub(r'#{1,6}\s+', '', clean)
            clean = re.sub(r'https?://\S+', '', clean)
            clean = re.sub(r'\s+', ' ', clean).strip()
            if len(clean) > 250:
                cut = clean[:250].rfind('.')
                clean = clean[:cut + 1] if cut > 100 else clean[:250]
            actions.append(("SPEAK", clean))

        return actions

    @property
    def is_busy(self):
        return self._busy
