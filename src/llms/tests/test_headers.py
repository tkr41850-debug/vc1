from __future__ import annotations

from llms.proxy.config import Settings
from llms.proxy.zen_headers import build_zen_headers, new_request_id, stable_session_id
from tests.conftest import make_settings


def test_stable_session_id_anonymous():
    assert stable_session_id("") == stable_session_id("")
    assert stable_session_id("").startswith("ses_")


def test_stable_session_id_differs_per_key():
    assert stable_session_id("key-a") != stable_session_id("key-b")


def test_request_ids_unique():
    assert new_request_id() != new_request_id()


def test_free_tier_sends_no_auth_but_identity_headers():
    headers = build_zen_headers(make_settings(zen_api_key=""))
    assert "Authorization" not in headers
    assert headers["User-Agent"].startswith("opencode/")
    assert headers["x-opencode-client"] == "cli"
    assert headers["x-opencode-project"] == "global"
    assert headers["x-opencode-session"].startswith("ses_")
    assert headers["x-opencode-request"].startswith("msg_")


def test_user_agent_matches_genuine_v2_shape():
    headers = build_zen_headers(
        make_settings(
            zen_api_key="",
            opencode_channel="latest",
            opencode_version="2.0.12",
            opencode_client="cli",
        )
    )
    assert headers["User-Agent"] == "opencode/latest/2.0.12/cli"


def test_session_id_matches_genuine_shape():
    import re

    sid = stable_session_id("key-a")
    assert re.fullmatch(r"ses_[0-9a-f]{12}[0-9A-Za-z]{14}", sid) is not None


def test_affinity_headers_mirror_session():
    headers = build_zen_headers(make_settings(zen_api_key=""))
    assert headers["x-session-affinity"] == headers["x-opencode-session"]
    assert headers["x-session-id"] == headers["x-opencode-session"]


def test_incoming_credentials_never_forwarded():
    # sk- secrets, harness dummy keys, anything client-sent: never upstream.
    settings: Settings = make_settings(zen_api_key="")
    for incoming in ("Bearer sk-client", "Bearer dummy", "Bearer incoming-key"):
        headers = build_zen_headers(settings, incoming)
        assert "Authorization" not in headers


def test_env_key_used_when_set():
    settings: Settings = make_settings(zen_api_key="env-key")
    headers = build_zen_headers(settings, "Bearer sk-client")
    assert headers["Authorization"] == "Bearer env-key"


def test_env_key_used_when_no_incoming_auth():
    headers = build_zen_headers(make_settings(zen_api_key="env-key"))
    assert headers["Authorization"] == "Bearer env-key"
