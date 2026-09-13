from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
PORT = os.getenv("ZEN_GATEWAY_PORT", "8789")
BASE_URL = f"http://127.0.0.1:{PORT}"


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def load_key() -> str:
    import yaml

    path = DATA_DIR / "keys.yaml"
    if not path.exists():
        raise SystemExit("no data/keys.yaml — run: just llms keygen")
    keys = yaml.safe_load(path.read_text()) or []
    for k in keys:
        if isinstance(k, dict) and k.get("key") and k.get("enabled", True):
            return str(k["key"])
    raise SystemExit("no enabled key in data/keys.yaml — run: just llms keygen")


def cmd_chat(args: list[str]) -> int:
    import httpx

    message = args[0] if args else "Reply with exactly: local-ok."
    model = os.getenv("PROBE_MODEL", "muse-spark-1.3-contributor-free")
    r = httpx.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "max_tokens": 256,
        },
        headers={"Authorization": f"Bearer {load_key()}"},
        timeout=180.0,
    )
    if r.status_code != 200:
        return fail(f"chat failed: {r.status_code} {r.text[:300]}")
    message = r.json()["choices"][0]["message"]
    print(message.get("content") or json.dumps(message)[:500])
    return 0


def cmd_stream(args: list[str]) -> int:
    import httpx

    message = args[0] if args else "Reply with exactly: local-ok."
    model = os.getenv("PROBE_MODEL", "muse-spark-1.3-contributor-free")
    text = ""
    with httpx.stream(
        "POST",
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "max_tokens": 256,
            "stream": True,
        },
        headers={"Authorization": f"Bearer {load_key()}"},
        timeout=180.0,
    ) as r:
        if r.status_code != 200:
            return fail(f"stream failed: {r.status_code} {r.read().decode()[:300]}")
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            try:
                delta = json.loads(data)["choices"][0]["delta"].get("content", "")
            except (KeyError, ValueError, IndexError):
                continue
            text += delta
            print(delta, end="", flush=True)
    print()
    return 0


def cmd_models(_args: list[str]) -> int:
    import httpx

    r = httpx.get(
        f"{BASE_URL}/v1/models",
        headers={"Authorization": f"Bearer {load_key()}"},
        timeout=30.0,
    )
    if r.status_code != 200:
        return fail(f"models failed: {r.status_code} {r.text[:300]}")
    for m in r.json().get("data", []):
        print(m.get("id"))
    return 0


def cmd_usage(_args: list[str]) -> int:
    path = DATA_DIR / "usage.json"
    if not path.exists():
        return fail("no usage.json yet — send traffic first")
    snapshot = json.loads(path.read_text())
    for key, entry in snapshot.get("keys", {}).items():
        print(
            f"{key[:11]}… requests={entry.get('requests')} "
            f"in={entry.get('input_tokens')} out={entry.get('output_tokens')} "
            f"cached={entry.get('cached_tokens')}"
        )
    return 0


def cmd_providers(_args: list[str]) -> int:
    import yaml

    path = DATA_DIR / "providers.yaml"
    if not path.exists():
        print("no providers.yaml — direct egress only")
        return 0
    for p in yaml.safe_load(path.read_text()) or []:
        if not isinstance(p, dict):
            continue
        pid = p.get("id", "?")
        status_path = DATA_DIR / "warps" / pid / "status.json"
        ready: str | int = "n/a"
        if status_path.exists():
            try:
                exits = json.loads(status_path.read_text()).get("exits") or []
                ready = sum(1 for w in exits if isinstance(w, dict) and w.get("ready"))
            except ValueError:
                ready = "unparsable"
        print(
            f"{pid} kind={p.get('kind')} enabled={p.get('enabled', True)} "
            f"exits={p.get('exits')} ready={ready} models={p.get('models')}"
        )
    return 0


def cmd_add_provider(args: list[str]) -> int:
    import yaml

    if not args:
        return fail("usage: add-provider <id> [exits=1] [models=*] [label=]")
    pid, exits, models, label = args[0], 1, ["*"], ""
    for a in args[1:]:
        if a.isdigit():
            exits = max(1, int(a))
        elif a.startswith("models="):
            models = [m for m in a[len("models=") :].split(",") if m] or ["*"]
        elif a.startswith("label="):
            label = a[len("label=") :]
        else:
            return fail(f"unknown arg: {a} (want [exits=N] [models=a,b] [label=x])")
    path = DATA_DIR / "providers.yaml"
    providers = yaml.safe_load(path.read_text()) if path.exists() else []
    providers = [
        p for p in (providers or []) if not (isinstance(p, dict) and p.get("id") == pid)
    ]
    providers.append(
        {
            "id": pid,
            "label": label or pid,
            "kind": "warp",
            "models": models,
            "enabled": True,
            "exits": exits,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(providers, sort_keys=False))
    (DATA_DIR / "warps" / pid).mkdir(parents=True, exist_ok=True)
    print(f"provider {pid} saved (exits={exits} models={models})")
    print(f"activate: just llms local reconnect-pool {pid}  (or restart: just llms up)")
    return 0


def cmd_reconnect_pool(args: list[str]) -> int:
    import httpx

    if not args:
        return fail("usage: reconnect-pool <id>")
    pid = args[0]
    r = httpx.post(
        f"{BASE_URL}/api/providers/{pid}/reconnect",
        headers={"Authorization": f"Bearer {load_key()}"},
        timeout=90.0,
    )
    if r.status_code != 200:
        return fail(f"reconnect-pool failed: {r.status_code} {r.text[:300]}")
    print(json.dumps(r.json(), indent=2))
    return 0


def cmd_logs(args: list[str]) -> int:
    import os

    logfile = os.getenv("LLMS_LOG", "/tmp/llms.log")
    lines = "100"
    if args:
        if args[0].isdigit():
            lines = args[0]
        else:
            return fail("usage: logs [lines]")
    if not Path(logfile).exists():
        return fail(f"no {logfile} — run: just llms up")
    os.execvp("tail", ["tail", "-n", lines, "-F", logfile])


COMMANDS = {
    "chat": cmd_chat,
    "stream": cmd_stream,
    "models": cmd_models,
    "usage": cmd_usage,
    "providers": cmd_providers,
    "add-provider": cmd_add_provider,
    "reconnect-pool": cmd_reconnect_pool,
    "logs": cmd_logs,
}


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in COMMANDS:
        return fail(f"usage: local_call.py {{{'|'.join(COMMANDS)}}} [...]")
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
