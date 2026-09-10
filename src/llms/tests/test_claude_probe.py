from __future__ import annotations

import pytest


@pytest.mark.skip(reason="live probe; run via just probe-claude")
def test_claude_code_headless_through_proxy():
    raise AssertionError("run via just probe-claude")
