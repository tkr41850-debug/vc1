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
    headers = build_zen_headers(make_settings(zen_api_key=""), None)
    assert "Authorization" not in headers
    assert headers["User-Agent"].startswith("opencode/")
    assert headers["x-opencode-client"] == "cli"
    assert headers["x-opencode-project"] == "global"
    assert headers["x-opencode-session"].startswith("ses_")
    assert headers["x-opencode-request"].startswith("msg_")


def test_incoming_bearer_ignored_by_default():
    settings: Settings = make_settings(zen_api_key="")
    headers = build_zen_headers(settings, "Bearer incoming-key")
    assert "Authorization" not in headers


def test_env_key_wins_over_incoming():
    settings: Settings = make_settings(zen_api_key="env-key")
    headers = build_zen_headers(settings, "Bearer incoming-key")
    assert headers["Authorization"] == "Bearer env-key"


def test_incoming_forwarded_when_allowed():
    settings: Settings = make_settings(zen_api_key="", allow_client_keys=True)
    headers = build_zen_headers(settings, "Bearer incoming-key")
    assert headers["Authorization"] == "Bearer incoming-key"


def test_env_key_used_when_no_incoming_auth():
    headers = build_zen_headers(make_settings(zen_api_key="env-key"), None)
    assert headers["Authorization"] == "Bearer env-key"
