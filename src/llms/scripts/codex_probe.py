from __future__ import annotations

import os
import subprocess

from llms.probe.dirs import probe_dirs
from llms.probe.proc import fail, running_proxy

PORT = int(os.getenv("PROBE_PORT", "8793"))
BASE_URL = f"http://127.0.0.1:{PORT}/v1"
MODEL = os.getenv("PROBE_MODEL", "muse-spark-1.3-contributor-free")


def main() -> int:
    try:
        with probe_dirs("codex-probe") as (home, workspace):
            (home / "config.toml").write_text(
                f'model = "{MODEL}"\n'
                'model_provider = "zen-proxy"\n'
                "\n"
                "[model_providers.zen-proxy]\n"
                'name = "zen-proxy"\n'
                f'base_url = "{BASE_URL}"\n'
                'wire_api = "responses"\n'
            )
            with running_proxy(
                PORT,
                os.getenv("PROXY_LOG", "/tmp/codex-probe-proxy.log"),
                data_dir=os.getenv("PROBE_DATA_DIR", "/tmp/codex-probe-data"),
                probe_secret=os.getenv("PROBE_SECRET"),
            ) as (_, secret):
                env = dict(
                    os.environ,
                    CODEX_HOME=str(home),
                    OPENAI_API_KEY=secret,
                )
                completed = subprocess.run(
                    [
                        "codex",
                        "exec",
                        "--skip-git-repo-check",
                        "Reply with exactly: codex-ok.",
                    ],
                    cwd=str(workspace),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=False,
                )
                print(completed.stdout[-2000:])
                if completed.returncode != 0:
                    return fail(completed.stderr[-2000:])
                tools = subprocess.run(
                    [
                        "codex",
                        "exec",
                        "--skip-git-repo-check",
                        "Use the shell to run echo tool-ok, then reply with exactly its output.",
                    ],
                    cwd=str(workspace),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=False,
                )
                print(tools.stdout[-2000:])
                if tools.returncode != 0 or "tool-ok" not in tools.stdout:
                    return fail(
                        f"tool turn failed: {tools.returncode} {tools.stderr[-2000:]}"
                    )
                import json as _json

                schema_file = workspace / "schema.json"
                schema_file.write_text(
                    _json.dumps(
                        {
                            "type": "object",
                            "properties": {"ok": {"type": "string"}},
                            "required": ["ok"],
                            "additionalProperties": False,
                        }
                    )
                )
                structured = subprocess.run(
                    [
                        "codex",
                        "exec",
                        "--skip-git-repo-check",
                        "--output-schema",
                        str(schema_file),
                        "Reply with a JSON object with ok set to schema-ok.",
                    ],
                    cwd=str(workspace),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=False,
                )
                print(structured.stdout[-2000:])
                try:
                    parsed = _json.loads(structured.stdout.strip().splitlines()[-1])
                except Exception:
                    parsed = {}
                if structured.returncode != 0 or parsed.get("ok") != "schema-ok":
                    return fail(
                        f"schema turn failed: {structured.returncode} {structured.stderr[-2000:]}"
                    )
        return 0
    except Exception as exc:
        return fail(f"probe failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
