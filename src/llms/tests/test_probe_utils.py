from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from llms.probe.dirs import probe_dirs
from llms.probe.health import wait_for_health


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/healthz":
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass


def test_wait_for_health_true():
    server = HTTPServer(("127.0.0.1", 0), _HealthHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert wait_for_health(port, timeout_s=5) is True
    finally:
        server.shutdown()


def test_wait_for_health_false_on_closed_port():
    assert wait_for_health(47911, timeout_s=1) is False


def test_probe_dirs_creates_home_and_workspace(tmp_path):
    with probe_dirs("probe-x", root=str(tmp_path)) as (home, workspace):
        assert home.is_dir()
        assert workspace.is_dir()
        assert home.parent.name == "probe-x"
