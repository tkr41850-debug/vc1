"""Refresh the checked-in Zen fingerprint from GitHub source.

Structural drift (headers/session mint) is reported for a human — exit 2.
A newer v2 tag is pinned only after a live probe shows Zen accepts it
(the UA allowlist lags releases); applied pins need a gateway restart.

Exit 0 in sync or auto-applied, 1 on fetch/parse failure, 2 on
unapplied drift.
"""

from __future__ import annotations

from llms.proxy import zen_fingerprint as fp


def main() -> int:
    try:
        report = fp.run_refresh()
    except Exception as exc:
        print(f"fingerprint refresh failed: {exc}")
        return 1
    if report.get("structural"):
        print(f"shape drift needs a human: {report['structural']}")
        return 2
    if report.get("version_applied"):
        print(
            f"ua_version {report['version_applied']} probe-accepted and pinned; "
            "restart the gateway to pick it up"
        )
        return 0
    print(f"fingerprint check: {report.get('note') or 'in sync'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
