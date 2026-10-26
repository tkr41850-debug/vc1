"""Checked-in Zen fingerprint + GitHub-source refresh.

Genuine opencode wire identity (captured + source-traced, Oct 2026 —
post-restructure: ``packages/opencode/`` is gone, the header block
lives in ``packages/core/src/session/model-request.ts`` and the title
prompt in ``packages/core/src/plugin/agent.ts``):

- ``Authorization: Bearer public`` (anonymous) — omitting it 403s.
- ``User-Agent: opencode/<channel>/<version>/<client>`` — only released
  versions pass; stale (1.18.4) and unreleased (9.9.99) 403.
- ``x-opencode-session`` / ``x-opencode-session-id`` /
  ``x-session-affinity`` / ``x-session-id`` all carry a ``ses_`` +
  12-hex + 14-base62 id (identifier.ts descending IDs) whose time part
  must decode to a plausible timestamp — random-head ids 403 even
  with a perfect wire shape (live-bisected Oct 2026). Genuine mints
  them fresh per session; the gateway mirrors that (time-ordered
  mint per conversation, day-bucketed stable fallback).
- responses body ``prompt_cache_key`` must equal the session id.
- ``x-opencode-request`` is gone upstream (the model-request.ts block
  no longer sends it), so its presence is forbidden again and the
  gateway leaves it out. Parent-session ids only exist on subagent
  turns; single-session traffic omits them, matching genuine.

``EXPECTED`` is the checked-in snapshot. ``scripts/refresh_zen_fingerprint.py``
re-fetches the upstream sources below and updates the snapshot (and the
config version default) when opencode drifts. The gateway calls
:func:`note_free_tier_error` whenever Zen answers ``FreeTierError`` so a
stale fingerprint triggers a background re-check instead of silent 403s.

Shape rejections are ``403`` + ``FreeTierError`` since the v2.1.x update
(they were 5xx before). Never key the staleness signal on the status
class — ``error.type`` is the stable contract, and the fingerprint hook
fires for 403 exactly like it did for 5xx.
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
from typing import NamedTuple

logger = logging.getLogger("zen_proxy")

from llms.proxy.zen_prompts import TITLE_PREFIX

EXPECTED = {
    # ua_version must be a version Zen currently ACCEPTS (newest is not
    # automatically safe — the allowlist lags releases. Only bump after
    # a live probe 200s.
    # Upstream tags ran to v2.1.4 on 2026-10-26; the pin stays at 2.1.0
    # until a probe can run (the anonymous tier was quota-exhausted that
    # day, and a 429 cannot tell acceptance from rejection).
    "ua_version": "2.1.0",
    "ua_channel": "latest",
    "required_headers": (
        "authorization",
        "user-agent",
        "x-opencode-client",
        # Conditional upstream (subagent turns only): single-session
        # traffic omits them, matching genuine — listed so the parser
        # union stays drift-free, not because the gateway sends them.
        "x-opencode-parent-session-id",
        "x-opencode-project",
        "x-opencode-session",
        "x-opencode-session-id",
        "x-parent-session-id",
        "x-session-affinity",
        "x-session-id",
    ),
    # Gone upstream (model-request.ts no longer sends it): presence
    # 403s per the original bisect, so the gateway leaves it out.
    "forbidden_headers": ("x-opencode-request",),
    "session_pattern": r"ses_[0-9a-f]{12}[0-9A-Za-z]{14}",
    "prompt_cache_key": "mirror-session",
    "title_prefix": TITLE_PREFIX,
}

SOURCES = {
    "request": "https://raw.githubusercontent.com/sst/opencode/dev/packages/core/src/session/model-request.ts",
    "session_id": "https://raw.githubusercontent.com/sst/opencode/dev/packages/schema/src/session-id.ts",
    "identifier": "https://raw.githubusercontent.com/sst/opencode/dev/packages/schema/src/identifier.ts",
    "title_prompt": "https://raw.githubusercontent.com/sst/opencode/dev/packages/core/src/plugin/agent.ts",
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


def _parse_title_prompt(text: str) -> str:
    """PROMPT_TITLE body from agent.ts (moved from title.txt).

    The const template carries no backticks or ${} interpolation
    (verified against upstream), so the first closing backtick ends
    it. Raises when the shape changes — a silent empty would
    disable version probing downstream.
    """
    m = re.search(r"const PROMPT_TITLE\s*=\s*`", text)
    if m is None:
        raise ValueError("agent.ts PROMPT_TITLE not found")
    end = text.find("`", m.end())
    if end < 0:
        raise ValueError("agent.ts PROMPT_TITLE unterminated")
    return text[m.end() : end]


def parse_expected(sources: dict[str, str]) -> dict:
    """Derive the expected fingerprint from upstream source text."""
    length, alphabet = _parse_identifier_params(sources["identifier"])
    if '"ses_"' not in sources["session_id"] and "'ses_'" not in sources["session_id"]:
        raise ValueError("session-id.ts no longer mints ses_ ids")
    if "descending()" not in sources["session_id"]:
        raise ValueError("session-id.ts no longer uses descending IDs")
    # Header block in model-request.ts: the http.headers literal for the
    # model request. Authorization comes from provider auth (always
    # Bearer); User-Agent and the x- headers are literals in the block —
    # normalize and union. Anchored on x-opencode-session-id (unique in
    # the file); the window covers the whole literal either side.
    body = sources["request"]
    anchor = '"x-opencode-session-id"'
    at = body.find(anchor)
    if at < 0:
        raise ValueError("model-request.ts header block not found")
    window = body[max(0, at - 500) : at + 1500]
    # Case-insensitive: upstream spells one header "X-Session-Id"
    # (same wire header as x-session-id — HTTP names are
    # case-insensitive, and our gateway sends it lowercase).
    headers = {
        h.lower()
        for h in re.findall(r'"(x-[a-z0-9-]+|User-Agent)"', window, re.IGNORECASE)
    }
    headers.add("authorization")
    if "x-opencode-session" not in headers:
        raise ValueError("model-request.ts header block lost x-opencode-session")
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
            () if '"x-opencode-request"' in window else ("x-opencode-request",)
        ),
        "session_pattern": rf"ses_[0-9a-f]{{12}}[{alpha_class}]{{{rand_len}}}",
        "prompt_cache_key": "mirror-session",
        "title_prefix": _parse_title_prompt(sources["title_prompt"])[:1500],
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
    """Well-formed ses_ id for acceptance probes (time-ordered, random)."""
    from llms.proxy.zen_headers import descending_session_id

    return descending_session_id()


class ProbeOutcome(NamedTuple):
    """One version-acceptance probe: observed status, verdict, and why.

    ``accepted`` is tri-state and ``status`` carries the evidence the
    verdict was derived from, so a report can name the actual cause
    instead of guessing "quota or 5xx".
    """

    status: int | None
    accepted: bool | None
    detail: str = ""

    @property
    def transient(self) -> bool:
        """The gate never evaluated the version (quota, burst, incident)."""
        return self.accepted is None and (
            self.status is None or self.status == 429 or self.status >= 500
        )


def _probe_error_outcome(version: str, exc) -> ProbeOutcome:
    status = getattr(exc, "code", None) or 0
    retry_after = ""
    # HTTPMessage lookups are case-insensitive; never `dict()` the headers
    # first (that lowercases nothing and silently drops `Retry-After`).
    try:
        ra = (exc.headers or {}).get("retry-after")
    except Exception:
        ra = None
    if ra:
        retry_after = f", retry-after {ra}s"
    body = ""
    try:
        body = exc.read().decode(errors="replace")
    except Exception:
        pass
    etype = ""
    emsg = ""
    try:
        payload = json.loads(body) if body else None
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            etype = str(error.get("type") or "")
            emsg = str(error.get("message") or "")
    except Exception:
        pass
    logger.debug(
        "version probe %s -> HTTP %s %s %s", version, status, etype, body[:160].strip()
    )
    label = f"HTTP {status}{f' {etype}' if etype else ''}{retry_after}"
    if status == 403:
        return ProbeOutcome(status, False, label)
    if status == 429 or status >= 500:
        return ProbeOutcome(status, None, label)
    # 400/401/404/422: the gate did not reject the *version* — the probe
    # recipe itself no longer fits (model renamed, endpoint moved, body
    # shape tightened). Surface it as needing a human, not as quota.
    snippet = " ".join((emsg or body).split())[:120]
    detail = label
    if snippet:
        detail += f": {snippet}"
    return ProbeOutcome(status, None, detail)


def probe_version_outcome(
    version: str,
    title_prompt: str,
    base_url: str = "https://opencode.ai/inference/openai/v1",
    timeout: float = 30.0,
) -> ProbeOutcome:
    """Live-probe a UA version and report the evidence, not just a verdict.

    Sends the known-good shape (well-formed session + matching
    prompt_cache_key + canonical title instructions, tiny output cap).
    A 403 costs nothing (no inference runs); a 200 burns a few tokens.
    Network errors raise.
    """
    import urllib.error

    session = mint_probe_session()
    body = json.dumps(
        {
            "model": "glm-5.3-flash-free",
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
            "x-opencode-session-id": session,
            "x-session-affinity": session,
            "x-session-id": session,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return ProbeOutcome(r.status, r.status == 200, f"HTTP {r.status}")
    except urllib.error.HTTPError as exc:
        return _probe_error_outcome(version, exc)


def probe_version_accepted(
    version: str,
    title_prompt: str,
    base_url: str = "https://opencode.ai/inference/openai/v1",
    timeout: float = 30.0,
) -> bool | None:
    """Verdict-only view of :func:`probe_version_outcome`.

    ``True`` (accepted), ``False`` (403: version rejected) or ``None``
    (inconclusive — quota 429, upstream 5xx, or a probe recipe that no
    longer fits). Inconclusive must never be read as rejection: the
    quota gate runs *before* the version gate, so a spent quota makes
    every version look refused.
    """
    return probe_version_outcome(version, title_prompt, base_url, timeout).accepted


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
    base_url: str = "https://opencode.ai/inference/openai/v1",
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
    title_source = sources.get("title_prompt", "")
    try:
        title_prompt = _parse_title_prompt(title_source)
    except ValueError:
        title_prompt = ""
    if not title_prompt.strip():
        report["note"] = "title prompt source empty; cannot probe versions"
        return report
    for candidate in candidates:
        try:
            outcome = probe_version_outcome(candidate, title_prompt, base_url, timeout)
        except Exception as exc:
            report["note"] = f"version probe network failure: {exc!r}"
            return report
        report["probe"] = {
            "version": candidate,
            "status": outcome.status,
            "detail": outcome.detail,
        }
        if outcome.accepted is None:
            if outcome.transient:
                if "FreeUsageLimitError" in outcome.detail:
                    why = (
                        "this egress IP's daily free-tier quota is spent"
                        " (resets with the retry-after above)"
                    )
                else:
                    why = "the gate never evaluated the version (quota or upstream incident)"
                report["note"] = (
                    f"version probe inconclusive for {candidate} "
                    f"({outcome.detail}): {why}; staying pinned"
                )
            else:
                report["note"] = (
                    f"version probe for {candidate} failed with "
                    f"{outcome.detail}: the probe recipe no longer fits "
                    "upstream (model renamed or endpoint moved?) — needs a "
                    "human, staying pinned"
                )
            return report
        if outcome.accepted:
            apply_version(candidate, root)
            report["version_applied"] = candidate
            return report
        logger.info("version probe %s -> %s (rejected)", candidate, outcome.detail)
    report["note"] = (
        f"newer tags exist {candidates} but Zen 403s them all; staying pinned"
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
    data_dir: str | Path,
    trace_id: str,
    base_url: str = "https://opencode.ai/inference/openai/v1",
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
