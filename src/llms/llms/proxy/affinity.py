from __future__ import annotations

import hashlib
import re

AFFINITY_PATTERN = re.compile(r"^ak-[A-Za-z0-9_-]+$")

DEFAULT_BUCKETS = 1024


def parse_affinity_prefix(path: str) -> tuple[str | None, str]:
    parts = path.split("/", 2)
    if len(parts) >= 3 and AFFINITY_PATTERN.fullmatch(parts[1]):
        return parts[1], "/" + parts[2]
    return None, path


def bucket_for(
    affinity: str | None,
    model: str,
    num_buckets: int = DEFAULT_BUCKETS,
    secret_key: str | None = None,
) -> int:
    # affinity (ak- path prefix) and the secret key (sk- header) are separate
    # namespaces but both enter the bucket hash, so different teams sharing a
    # model still spread across pools — while neither leaks upstream.
    key = f"{affinity or ''}\x00{secret_key or ''}\x00{model.strip().lower()}"
    digest = hashlib.sha256(key.encode()).hexdigest()
    return int(digest, 16) % num_buckets
