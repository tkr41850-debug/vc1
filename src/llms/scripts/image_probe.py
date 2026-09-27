from __future__ import annotations

import base64
import io
import os

import httpx

from llms.probe.dirs import probe_dirs
from llms.probe.proc import fail, running_proxy

PORT = int(os.getenv("PROBE_PORT", "8793"))
BASE_URL = f"http://127.0.0.1:{PORT}"
QUESTION = "What two solid colors fill the left and right halves of this image? Reply with just the color names."


def red_blue_png() -> bytes:
    """240x120 test image (red left, blue right) generated with PIL.

    Generated at runtime so no binary fixture is committed; the colors
    give a vision check with an unambiguous expected answer.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (240, 120), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 119, 119], fill=(255, 0, 0))
    draw.rectangle([120, 0, 239, 119], fill=(0, 0, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _sees_colors(text: str) -> bool:
    low = text.lower()
    return "red" in low and "blue" in low


def main() -> int:
    try:
        data = base64.b64encode(red_blue_png()).decode()
        with (
            probe_dirs("image-probe"),
            running_proxy(
                PORT,
                os.getenv("PROXY_LOG", "/tmp/image-probe-proxy.log"),
                data_dir=os.getenv("PROBE_DATA_DIR", "/tmp/image-probe-data"),
                probe_secret=os.getenv("PROBE_SECRET"),
            ) as (_, secret),
        ):
            headers = {"Authorization": f"Bearer {secret}"}
            # Messages leg (Claude path): base64 image block.
            r = httpx.post(
                f"{BASE_URL}/v1/messages",
                json={
                    "model": os.getenv("PROBE_MESSAGES_MODEL", "claude-haiku-4-5"),
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": data,
                                    },
                                },
                                {"type": "text", "text": QUESTION},
                            ],
                        }
                    ],
                    "max_tokens": 512,
                },
                headers=headers,
                timeout=180.0,
            )
            if r.status_code != 200:
                return fail(f"messages leg failed: {r.status_code} {r.text[:500]}")
            texts = [
                b.get("text", "")
                for b in r.json().get("content", [])
                if b.get("type") == "text"
            ]
            answer = "".join(texts)
            print(f"messages leg answer: {answer!r}")
            if not _sees_colors(answer):
                return fail(f"messages leg blind: {answer!r}")
            # Responses leg (Codex path): input_image data URL.
            r = httpx.post(
                f"{BASE_URL}/v1/responses",
                json={
                    "model": os.getenv(
                        "PROBE_MODEL", "muse-spark-1.3-contributor-free"
                    ),
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/png;base64,{data}",
                                },
                                {"type": "input_text", "text": QUESTION},
                            ],
                        }
                    ],
                },
                headers=headers,
                timeout=180.0,
            )
            if r.status_code != 200:
                return fail(f"responses leg failed: {r.status_code} {r.text[:500]}")
            payload = r.json()
            texts = [
                p.get("text", "")
                for m in payload.get("output", [])
                if m.get("type") == "message"
                for p in m.get("content", [])
            ]
            answer = "".join(texts)
            print(f"responses leg answer: {answer!r}")
            if not _sees_colors(answer):
                return fail(f"responses leg blind: {answer!r}")
        return 0
    except Exception as exc:
        return fail(f"probe failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
