from __future__ import annotations

import pytest

from scripts.claude_probe import check_prereqs


def test_messages_live_probe_guarded():
    with pytest.raises(NotImplementedError):
        check_prereqs()


@pytest.mark.skip(reason="no free messages-capable models on Zen yet")
def test_claude_code_headless_through_proxy():
    raise AssertionError(
        "enable with PROBE_MESSAGES_LIVE=1 once a messages model is available"
    )
