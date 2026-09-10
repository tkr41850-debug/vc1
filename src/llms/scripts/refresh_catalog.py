from __future__ import annotations

import json
import urllib.request

from llms.proxy.catalog import BY_ID


def main() -> int:
    request = urllib.request.Request(
        "https://opencode.ai/zen/v1/models", headers={"User-Agent": "llms-catalog-refresh/1.0"}
    )
    with urllib.request.urlopen(request, timeout=20) as r:
        live = json.load(r)
    live_free = sorted(
        m["id"] for m in live["data"] if "free" in m["id"] or m["id"] == "big-pickle"
    )
    seed_free = sorted(BY_ID)
    only_live = [m for m in live_free if m not in BY_ID]
    only_seed = [m for m in seed_free if m not in live_free]
    print(f"live free: {len(live_free)}, seed: {len(seed_free)}")
    if only_live:
        print(f"missing from seed: {only_live}")
    if only_seed:
        print(f"stale in seed: {only_seed}")
    if only_live or only_seed:
        return 1
    print("catalog matches live free set")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
