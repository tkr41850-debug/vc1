from __future__ import annotations

import re
import urllib.error

from llms.proxy import zen_fingerprint as fp
from llms.proxy.zen_prompts import TITLE_PREFIX

IDENTIFIER_FIXTURE = """const length = 26
const chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
export function descending() {
  return create(true)
}
"""

SESSION_ID_FIXTURE = """export const SessionID = Schema.String.check(Schema.isStartsWith("ses")).pipe(
  statics((schema) => {
    const create = () => schema.make("ses_" + descending())
  }),
)
"""

REQUEST_FIXTURE = """    ...(input.model.providerID.startsWith("opencode")
      ? {
          "x-opencode-project": opencodeProjectID,
          "x-opencode-session": input.sessionID,
          "x-session-affinity": input.sessionID,
          "x-session-id": input.sessionID,
          "x-opencode-client": input.flags.client,
          "User-Agent": USER_AGENT,
        }
      : {}),
"""

TAGS_FIXTURE = (
    '[{"name": "v2.0.12"}, {"name": "v2.0.9"}, {"name": "v1.18.32"},'
    ' {"name": "not-a-version"}]'
)


def _sources(**overrides):
    base = {
        "request": REQUEST_FIXTURE,
        "session_id": SESSION_ID_FIXTURE,
        "identifier": IDENTIFIER_FIXTURE,
        "title_prompt": TITLE_PREFIX + "\n<rest of the upstream prompt>",
        "tags": TAGS_FIXTURE,
    }
    base.update(overrides)
    return base


def test_parse_expected_matches_snapshot():
    parsed = fp.parse_expected(_sources())
    assert not fp.diff_fingerprints(fp.EXPECTED, parsed), parsed


def test_parse_expected_session_pattern_shape():
    parsed = fp.parse_expected(_sources())
    assert re.fullmatch(parsed["session_pattern"], "ses_f37eb59b5ffemW0CO4BXGkHGfb")
    assert not re.fullmatch(parsed["session_pattern"], "ses_short")


def test_parse_expected_detects_request_header():
    parsed = fp.parse_expected(
        _sources(request=REQUEST_FIXTURE + '\n"x-opencode-request": input.user.id,')
    )
    assert parsed["forbidden_headers"] == ()


def test_list_v2_versions_descending():
    assert fp.list_v2_versions(TAGS_FIXTURE) == ["v2.0.12", "v2.0.9"]


def test_mint_probe_session_shape():
    assert (
        re.fullmatch(fp.EXPECTED["session_pattern"], fp.mint_probe_session())
        is not None
    )


def test_probe_version_accepted_status(monkeypatch):
    class Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(fp.urllib.request, "urlopen", lambda *a, **k: Resp())
    assert fp.probe_version_accepted("2.0.12", "title") is True

    def denied(*a, **k):
        raise urllib.error.HTTPError("https://x", 403, "Forbidden", {}, None)

    monkeypatch.setattr(fp.urllib.request, "urlopen", denied)
    assert fp.probe_version_accepted("9.9.99", "title") is False


def test_apply_version_bumps_newer_only(tmp_path):
    proxy = tmp_path / "llms" / "proxy"
    proxy.mkdir(parents=True)
    (proxy / "config.py").write_text(
        'x = os.getenv("ZEN_GATEWAY_OPENCODE_VERSION", "2.0.12")\n'
    )
    (proxy / "zen_fingerprint.py").write_text('"ua_version": "2.0.12"\n')
    fp.apply_version("2.0.16", root=tmp_path)
    assert '"2.0.16"' in (proxy / "config.py").read_text()
    assert '"2.0.16"' in (proxy / "zen_fingerprint.py").read_text()
    try:
        fp.apply_version("1.0.0", root=tmp_path)
    except ValueError:
        pass
    else:
        raise AssertionError("downgrade should refuse")
    assert '"1.0.0"' not in (proxy / "config.py").read_text()


def test_run_refresh_pins_newest_accepted(tmp_path, monkeypatch):
    proxy = tmp_path / "llms" / "proxy"
    proxy.mkdir(parents=True)
    (proxy / "config.py").write_text(
        'x = os.getenv("ZEN_GATEWAY_OPENCODE_VERSION", "2.0.12")\n'
    )
    (proxy / "zen_fingerprint.py").write_text('"ua_version": "2.0.12"\n')
    import llms.proxy.zen_fingerprint as _fp

    monkeypatch.setattr(_fp, "EXPECTED", dict(fp.EXPECTED))
    tags = '[{"name": "v2.0.16"}, {"name": "v2.0.15"}, {"name": "v2.0.12"}]'
    monkeypatch.setattr(_fp, "fetch_sources", lambda *a, **k: _sources(tags=tags))
    monkeypatch.setattr(_fp, "probe_version_accepted", lambda v, *a, **k: v == "2.0.15")
    report = _fp.run_refresh(root=tmp_path)
    assert report["version_applied"] == "2.0.15"
    assert '"2.0.15"' in (proxy / "config.py").read_text()


def test_free_tier_error_triggers_gated_recheck(tmp_path, monkeypatch):
    import llms.proxy.zen_fingerprint as _fp

    calls: list = []
    monkeypatch.setattr(_fp, "CHECK_COOLDOWN_S", 3600.0)
    monkeypatch.setattr(_fp, "run_refresh", lambda *a, **k: calls.append(1) or {})
    _fp.note_free_tier_error(str(tmp_path), "trace-1", "https://x")
    assert (tmp_path / ".zen_fingerprint_check").exists()
    for _ in range(100):
        if calls:
            break
        import time as _t

        _t.sleep(0.05)
    assert len(calls) == 1
    _fp.note_free_tier_error(str(tmp_path), "trace-2", "https://x")
    import time as _t

    _t.sleep(0.2)
    assert len(calls) == 1
