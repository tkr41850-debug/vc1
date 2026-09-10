from __future__ import annotations

import pytest

from llms.proxy.egress import WarpPoolEgress


@pytest.mark.skip(reason="vsp per-bucket warp selection API not implemented yet")
def test_warp_pool_serves_bucket_through_leased_warp():
    egress = WarpPoolEgress("http://127.0.0.1:8080", num_slots=8)
    client = egress.client_for(bucket=17, slot=3)
    assert client is not None
