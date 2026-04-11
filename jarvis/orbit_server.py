"""Tiny HTTP server for the orbit animation.

Serves the animation HTML and exposes /amp endpoint for amplitude data.
Designed to run in a background thread from the voice input GUI.
"""

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

_ANIMATION_HTML = Path(__file__).parent / "orbit_animation.html"

# Shared state — updated by the GUI
_state = {"amp": 0.25, "recording": False}
_state_lock = threading.Lock()


def set_state(amp=None, recording=None):
    with _state_lock:
        if amp is not None:
            _state["amp"] = amp
        if recording is not None:
            _state["recording"] = recording


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/amp"):
            with _state_lock:
                data = json.dumps(_state)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data.encode())
        elif self.path == "/" or self.path.startswith("/index"):
            html = _ANIMATION_HTML.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress request logging


def start_server(port=0):
    """Start the orbit animation server in a background thread.

    Args:
        port: Port to listen on (0 = auto-pick)

    Returns:
        (server, port) tuple
    """
    server = HTTPServer(("127.0.0.1", port), _Handler)
    actual_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, actual_port
