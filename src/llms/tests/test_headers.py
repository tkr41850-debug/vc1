from __future__ import annotations

from llms.proxy.config import Settings
from llms.proxy.zen_headers import (
    build_zen_headers,
    descending_session_id,
    stable_session_id,
)
from tests.conftest import make_settings


def _head_time_mod(session_id: str) -> int:
    # Inverse of descending_session_id (counter=1): the 6 head bytes
    # carry the timestamp mod 2**36 ms (~2.2y window) — the full stamp
    # does not fit 48 bits with the 0x1000 multiplier.
    value = int(session_id.removeprefix("ses_")[:12], 16)
    return (((~value) & ((1 << 48) - 1)) - 1) // 0x1000


def test_stable_session_id_anonymous():
    assert stable_session_id("") == stable_session_id("")
    assert stable_session_id("").startswith("ses_")


def test_stable_session_id_differs_per_key():
    assert stable_session_id("key-a") != stable_session_id("key-b")


def test_descending_session_id_head_is_current_time():
    import time

    assert descending_session_id(now_ms=1791877187547, rand=bytes(range(14))) == (
        "ses_ecbcbf824ffe0123456789ABCD"
    )
    before = int(time.time() * 1000)
    sid = descending_session_id()
    after = int(time.time() * 1000)
    mod = 2**36
    for stamp in (before, after):
        delta = (_head_time_mod(sid) - stamp) % mod
        assert min(delta, mod - delta) <= 5000


def test_stable_session_id_rotates_daily_but_same_shape():
    import re

    a = stable_session_id("key-a", now_s=1760390400.0)
    assert a == stable_session_id("key-a", now_s=1760390400.0 + 3600.0)
    assert a != stable_session_id("key-a", now_s=1760390400.0 + 86400.0)
    assert re.fullmatch(r"ses_[0-9a-f]{12}[0-9A-Za-z]{14}", a)


def test_free_tier_sends_public_bearer_but_identity_headers():
    headers = build_zen_headers(make_settings(zen_api_key=""))
    assert headers["Authorization"] == "Bearer public"
    assert headers["User-Agent"].startswith("opencode/")
    assert headers["x-opencode-client"] == "cli"
    assert headers["x-opencode-project"] == "global"
    assert headers["x-opencode-session"].startswith("ses_")
    assert headers["x-opencode-session-id"] == headers["x-opencode-session"]
    # Deliberately omitted: genuine sends the user id here, but the gate
    # accepts omission (probed Oct 2026) — never fabricate a user id.
    assert "x-opencode-request" not in headers


def test_session_id_override_shared_with_body():
    headers = build_zen_headers(make_settings(zen_api_key=""), session_id="ses_custom")
    assert headers["x-opencode-session"] == "ses_custom"
    assert headers["x-opencode-session-id"] == "ses_custom"
    assert headers["x-session-affinity"] == "ses_custom"
    assert headers["x-session-id"] == "ses_custom"


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
    assert headers["x-opencode-session-id"] == headers["x-opencode-session"]


def test_incoming_credentials_never_forwarded():
    # sk- secrets, harness dummy keys, anything client-sent: never upstream.
    # The operator key (or the public free-tier marker) is the only
    # Authorization that reaches Zen.
    settings: Settings = make_settings(zen_api_key="")
    for incoming in ("Bearer sk-client", "Bearer dummy", "Bearer incoming-key"):
        headers = build_zen_headers(settings, incoming)
        assert headers["Authorization"] == "Bearer public"


def test_env_key_used_when_set():
    settings: Settings = make_settings(zen_api_key="env-key")
    headers = build_zen_headers(settings, "Bearer sk-client")
    assert headers["Authorization"] == "Bearer env-key"


def test_env_key_used_when_no_incoming_auth():
    headers = build_zen_headers(make_settings(zen_api_key="env-key"))
    assert headers["Authorization"] == "Bearer env-key"
