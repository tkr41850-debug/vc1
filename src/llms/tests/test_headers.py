from __future__ import annotations

from proxy.config import Settings
from proxy.zen_headers import build_zen_headers, new_request_id, stable_session_id
from tests.conftest import make_settings


def test_stable_session_id_anonymous():
    assert stable_session_id("") == stable_session_id("")
    assert stable_session_id("").startswith("ses_")


def test_stable_session_id_differs_per_key():
    assert stable_session_id("key-a") != stable_session_id("key-b")


def test_request_ids_unique():
    assert new_request_id() != new_request_id()


def test_free_tier_sends_no_auth_but_identity_headers():
    headers = build_zen_headers(make_settings(zen_api_key=""), None)
    assert "Authorization" not in headers
    assert headers["User-Agent"] == "opencode/1.18.4"
    assert headers["x-opencode-client"] == "cli"
    assert headers["x-opencode-project"] == "global"
    assert headers["x-opencode-session"].startswith("ses_")
    assert headers["x-opencode-request"].startswith("msg_")


def test_incoming_bearer_preferred_over_env():
    settings: Settings = make_settings(zen_api_key="env-key")
    headers = build_zen_headers(settings, "Bearer incoming-key")
    assert headers["Authorization"] == "Bearer incoming-key"


def test_env_key_used_when_no_incoming_auth():
    headers = build_zen_headers(make_settings(zen_api_key="env-key"), None)
    assert headers["Authorization"] == "Bearer env-key"
