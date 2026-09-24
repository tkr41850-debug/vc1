"""Checked-in Zen fingerprint + GitHub-source refresh.

Genuine opencode wire identity (captured + source-traced, Sep 2026):

- ``Authorization: Bearer public`` (anonymous) — omitting it 403s.
- ``User-Agent: opencode/<channel>/<version>/<client>`` — only released
  versions pass; stale (1.18.4) and unreleased (9.9.99) 403.
- ``x-opencode-session`` / ``x-session-affinity`` / ``x-session-id`` all
  carry a ``ses_`` + 12-hex + 14-base62 id (identifier.ts descending IDs).
- responses body ``prompt_cache_key`` must equal the session id.
- No ``x-opencode-request`` — genuine v2 omits it and its presence 403s.

``EXPECTED`` is the checked-in snapshot. ``scripts/refresh_zen_fingerprint.py``
re-fetches the upstream sources below and updates the snapshot (and the
config version default) when opencode drifts. The gateway calls
:func:`note_free_tier_error` whenever Zen answers ``FreeTierError`` so a
stale fingerprint triggers a background re-check instead of silent 403s.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger("zen_proxy")

EXPECTED = {
    # ua_version must be a version Zen currently ACCEPTS (allowlist lags
    # releases: 2.0.16 exists yet 403s). Only bump after a live probe 200s.
    "ua_version": "2.0.12",
    "ua_channel": "latest",
    "required_headers": (
        "authorization",
        "user-agent",
        "x-opencode-client",
        "x-opencode-project",
        "x-opencode-session",
        "x-session-affinity",
        "x-session-id",
    ),
    "forbidden_headers": ("x-opencode-request",),
    "session_pattern": r"ses_[0-9a-f]{12}[0-9A-Za-z]{14}",
    "prompt_cache_key": "mirror-session",
}

SOURCES = {
    "request": "https://raw.githubusercontent.com/sst/opencode/dev/packages/opencode/src/session/llm/request.ts",
    "session_id": "https://raw.githubusercontent.com/sst/opencode/dev/packages/schema/src/session-id.ts",
    "identifier": "https://raw.githubusercontent.com/sst/opencode/dev/packages/schema/src/identifier.ts",
    "title_prompt": "https://raw.githubusercontent.com/sst/opencode/dev/packages/opencode/src/agent/prompt/title.txt",
    "tags": "https://api.github.com/repos/sst/opencode/tags?per_page=100",
}

FETCH_TIMEOUT_S = float(os.getenv("ZEN_FINGERPRINT_FETCH_TIMEOUT_S", "20"))
CHECK_COOLDOWN_S = float(os.getenv("ZEN_FINGERPRINT_CHECK_COOLDOWN_S", "21600"))
MARKER_NAME = ".zen_fingerprint_check"


def fetch_sources(timeout: float = FETCH_TIMEOUT_S) -> dict[str, str]:
    """Download the upstream fingerprint sources; raises on network failure."""
    out: dict[str, str] = {}
    for name, url in SOURCES.items():
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "llms-fingerprint-refresh", "Accept": "*/*"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out[name] = r.read().decode("utf-8", "replace")
    return out


def _parse_identifier_params(text: str) -> tuple[int, str]:
    """(total id length, base62-ish alphabet) from identifier.ts."""
    length = int(re.search(r"const length\s*=\s*(\d+)", text).group(1))  # type: ignore[union-attr]
    alphabet = re.search(r'const chars\s*=\s*"([^"]+)"', text).group(1)  # type: ignore[union-attr]
    return length, alphabet


def parse_expected(sources: dict[str, str]) -> dict:
    """Derive the expected fingerprint from upstream source text."""
    length, alphabet = _parse_identifier_params(sources["identifier"])
    if '"ses_"' not in sources["session_id"] and "'ses_'" not in sources["session_id"]:
        raise ValueError("session-id.ts no longer mints ses_ ids")
    if "descending()" not in sources["session_id"]:
        raise ValueError("session-id.ts no longer uses descending IDs")
    # Header block for the opencode provider branch in request.ts.
    # Authorization comes from provider auth (always Bearer), User-Agent and
    # the x- headers are literals in the branch — normalize and union.
    branch = sources["request"]
    marker = 'startsWith("opencode")'
    at = branch.find(marker)
    if at < 0:
        raise ValueError("request.ts opencode branch not found")
    window = branch[at : at + 1500]
    headers = {h.lower() for h in re.findall(r'"(x-[a-z0-9-]+|User-Agent)"', window)}
    headers.add("authorization")
    if "x-opencode-session" not in headers:
        raise ValueError("request.ts opencode branch lost x-opencode-session")
    try:
        tags = json.loads(sources["tags"])
    except ValueError as exc:
        raise ValueError(f"tags API unparseable: {exc}") from exc
    versions = sorted(
        (
            _version_tuple(t.get("name", "").removeprefix("v"))
            for t in tags
            if isinstance(t, dict)
            and (t.get("name") or "").startswith("v2.")
            and _version_tuple(t.get("name", "").removeprefix("v"))
        ),
        reverse=True,
    )
    if not versions:
        raise ValueError("tags API returned no v2 tags")
    version = ".".join(str(p) for p in versions[0])
    # identifier.create emits 12 hex time chars + (length-12) alphabet chars.
    rand_len = length - 12
    alpha_class = (
        "0-9A-Za-z"
        if alphabet == "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        else re.escape(alphabet)
    )
    return {
        "ua_version": version,
        "ua_channel": EXPECTED["ua_channel"],
        "required_headers": tuple(sorted(headers)),
        "forbidden_headers": (
            ("x-opencode-request",) if '"x-opencode-request"' not in window else ()
        ),
        "session_pattern": rf"ses_[0-9a-f]{{12}}[{alpha_class}]{{{rand_len}}}",
        "prompt_cache_key": "mirror-session",
    }


def diff_fingerprints(old: dict, new: dict) -> dict[str, tuple]:
    """{key: (old, new)} for changed keys (header tuples order-insensitive)."""
    drift: dict[str, tuple] = {}
    for key, new_value in new.items():
        old_value = old.get(key)
        if isinstance(new_value, tuple) and isinstance(old_value, tuple):
            if set(new_value) != set(old_value):
                drift[key] = (old_value, new_value)
        elif old_value != new_value:
            drift[key] = (old_value, new_value)
    return drift


def _version_tuple(value: str) -> tuple:
    try:
        return tuple(int(p) for p in value.split("."))
    except ValueError:
        return ()


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def list_v2_versions(tags_text: str) -> list[str]:
    """Newest-first v2 tags from the GitHub tags API payload."""
    try:
        tags = json.loads(tags_text)
    except ValueError as exc:
        raise ValueError(f"tags API unparseable: {exc}") from exc
    ordered = sorted(
        {
            t.get("name", "")
            for t in tags
            if isinstance(t, dict) and (t.get("name") or "").startswith("v2.")
        },
        key=lambda name: _version_tuple(name.removeprefix("v")),
        reverse=True,
    )
    if not ordered:
        raise ValueError("tags API returned no v2 tags")
    return ordered


def mint_probe_session() -> str:
    """Well-formed ses_ id for acceptance probes (shape only, random)."""
    import os as _os

    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    raw = _os.urandom(20)
    return "ses_" + raw[:6].hex() + "".join(alphabet[b % 62] for b in raw[6:])


def probe_version_accepted(
    version: str,
    title_prompt: str,
    base_url: str = "https://opencode.ai/zen/v1",
    timeout: float = 30.0,
) -> bool:
    """Live-probe whether Zen accepts a UA version (canonical recipe).

    Sends the known-good shape (well-formed session + matching
    prompt_cache_key + canonical title instructions, tiny output cap).
    A 403 costs nothing (no inference runs); a 200 burns a few tokens.
    Network errors raise; only an HTTP status answers the question.
    """
    import urllib.error

    session = mint_probe_session()
    body = json.dumps(
        {
            "model": "muse-spark-1.3-contributor-free",
            "prompt_cache_key": session,
            "stream": True,
            "max_output_tokens": 16,
            "instructions": title_prompt,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                }
            ],
        }
    ).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/responses",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer public",
            "User-Agent": f"opencode/latest/{version}/cli",
            "x-opencode-client": "cli",
            "x-opencode-project": "global",
            "x-opencode-session": session,
            "x-session-affinity": session,
            "x-session-id": session,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except urllib.error.HTTPError as exc:
        logger.debug("version probe %s -> HTTP %s", version, exc.code)
        return False


def apply_version(new_version: str, root: Path | None = None) -> None:
    """Pin a probe-accepted UA version in the config default + snapshot."""
    root = root or _repo_root()
    old_version = EXPECTED.get("ua_version", "")
    if not (_version_tuple(new_version) > _version_tuple(old_version)):
        raise ValueError(f"refusing to pin non-newer version {new_version}")
    config_path = root / "llms" / "proxy" / "config.py"
    text = config_path.read_text()
    needle = f'ZEN_GATEWAY_OPENCODE_VERSION", "{old_version}"'
    if needle not in text:
        raise ValueError("config.py version default not found; refusing to patch")
    config_path.write_text(
        text.replace(needle, f'ZEN_GATEWAY_OPENCODE_VERSION", "{new_version}"')
    )
    fp_path = root / "llms" / "proxy" / "zen_fingerprint.py"
    fp_text = fp_path.read_text()
    fp_needle = f'"ua_version": "{old_version}"'
    if fp_needle not in fp_text:
        raise ValueError("fingerprint snapshot version not found")
    fp_path.write_text(fp_text.replace(fp_needle, f'"ua_version": "{new_version}"'))


def run_refresh(
    base_url: str = "https://opencode.ai/zen/v1",
    timeout: float = FETCH_TIMEOUT_S,
    root: Path | None = None,
) -> dict:
    """One full refresh pass; returns a report dict.

    Structural drift is reported for a human (never auto-rewritten).
    A newer v2 tag is pinned only after a live probe accepts it — the
    allowlist lags releases, so newest is not automatically safe.
    """
    report: dict = {"structural": None, "version_applied": None, "note": ""}
    sources = fetch_sources(timeout)
    fetched = parse_expected(sources)
    structural = {
        k: v
        for k, v in diff_fingerprints(EXPECTED, fetched).items()
        if k != "ua_version"
    }
    if structural:
        report["structural"] = structural
        return report
    pinned = EXPECTED.get("ua_version", "")
    candidates = [
        v.removeprefix("v")
        for v in list_v2_versions(sources["tags"])
        if _version_tuple(v.removeprefix("v")) > _version_tuple(pinned)
    ]
    if not candidates:
        report["note"] = f"pinned ua {pinned} is newest known"
        return report
    title_prompt = sources.get("title_prompt", "")
    if not title_prompt.strip():
        report["note"] = "title prompt source empty; cannot probe versions"
        return report
    for candidate in candidates:
        try:
            accepted = probe_version_accepted(
                candidate, title_prompt, base_url, timeout
            )
        except Exception as exc:
            report["note"] = f"version probe network failure: {exc!r}"
            return report
        if accepted:
            apply_version(candidate, root)
            report["version_applied"] = candidate
            return report
    report["note"] = (
        f"newer tags exist {candidates} but Zen accepts none yet; staying pinned"
    )
    return report


def check(timeout: float = FETCH_TIMEOUT_S) -> dict[str, tuple]:
    """Fetch upstream and diff against the snapshot; {} when in sync."""
    return diff_fingerprints(EXPECTED, parse_expected(fetch_sources(timeout)))


def _marker_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / MARKER_NAME


def should_check_now(data_dir: str | Path) -> bool:
    """Cooldown gate so one bad hour doesn't hammer GitHub per request."""
    try:
        age = time.time() - _marker_path(data_dir).stat().st_mtime
    except OSError:
        return True
    return age >= CHECK_COOLDOWN_S


def note_free_tier_error(
    data_dir: str | Path, trace_id: str, base_url: str = "https://opencode.ai/zen/v1"
) -> None:
    """Background a fingerprint re-check after a Zen FreeTierError (cooldown-gated)."""
    try:
        if not should_check_now(data_dir):
            return
        _marker_path(data_dir).touch()
    except OSError as exc:
        logger.debug("[%s] fingerprint marker unwritable: %s", trace_id, exc)
        return
    logger.warning(
        "[%s] Zen FreeTierError: fingerprint may be stale, "
        "re-checking against GitHub source in the background",
        trace_id,
    )

    def _run() -> None:
        try:
            report = run_refresh(base_url)
        except Exception as exc:
            logger.warning("[%s] fingerprint re-check failed: %r", trace_id, exc)
            return
        if report.get("structural"):
            logger.warning(
                "[%s] Zen fingerprint structure drifted: %s — needs a human "
                "(run just llms refresh-fingerprint for details)",
                trace_id,
                report["structural"],
            )
            return
        if report.get("version_applied"):
            logger.warning(
                "[%s] Zen fingerprint: UA %s probe-accepted and pinned; "
                "restart the gateway to apply",
                trace_id,
                report["version_applied"],
            )
            return
        logger.info(
            "[%s] fingerprint re-check: %s",
            trace_id,
            report.get("note") or "in sync with upstream",
        )

    thread = threading.Thread(target=_run, name="zen-fingerprint-check", daemon=True)
    thread.start()
